"""Retrieval-augmented generation helpers for the SME documentation corpus.

Two backends are supported:

``local`` (default)
    A dependency-free TF-IDF vector retriever implemented with ``numpy`` only.
    It works out of the box - no external service, no embedding API key.

``weaviate`` (optional)
    The original Weaviate ``near_text`` path. It is only used when the env var
    ``RAG_BACKEND=weaviate`` is set *and* the ``weaviate`` package is importable.
    Any failure in that path degrades gracefully to the local backend.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root, resolved by walking up from this file (``<root>/tools/rag.py``).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default documentation directory, independent of the current working directory.
DEFAULT_DOCS_DIR: Path = REPO_ROOT / "docs"


def _resolve_docs_dir(docs_dir: str | Path | None) -> Path:
    """Resolve *docs_dir* to an absolute path.

    A relative path that does not exist relative to the current working
    directory is retried relative to the repository root, so callers can pass
    ``"docs"`` from anywhere.
    """
    if docs_dir is None:
        return DEFAULT_DOCS_DIR
    path = Path(docs_dir)
    if path.is_absolute():
        return path
    if path.is_dir():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.is_dir():
        return candidate
    return path


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

#: Small English stopword list - deliberately short so domain words survive.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren as at be because
    been before being below between both but by can cannot could did do does doing
    don down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most my
    myself no nor not of off on once only or other ought our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours yourself
    yourselves s t don t will
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase *text*, extract ``\\b\\w+\\b`` tokens and drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t and t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Loading and chunking
# --------------------------------------------------------------------------- #


def load_documents(docs_dir: str | Path | None = DEFAULT_DOCS_DIR) -> list[dict[str, str]]:
    """Load every markdown file in *docs_dir* (recursively) as ``{"source", "content"}`` dicts."""
    resolved = _resolve_docs_dir(docs_dir)
    docs: list[dict[str, str]] = []
    if not resolved.is_dir():
        logger.warning("RAG docs directory not found: %s", resolved)
        return docs
    for path in sorted(resolved.rglob("*.md")):
        try:
            docs.append({"source": path.name, "content": path.read_text(encoding="utf-8")})
        except OSError as exc:  # pragma: no cover - unreadable file
            logger.warning("Could not read %s: %s", path, exc)
    return docs


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown *text* into ``(heading, body)`` sections.

    Text appearing before the first heading is returned with an empty heading.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if buffer and "".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    if buffer and "".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    if not sections and text.strip():
        sections.append((heading, text.strip()))
    return sections


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Break an over-long paragraph on word boundaries into <= *max_chars* pieces."""
    words = paragraph.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        pieces.append(" ".join(current))
    return pieces or [paragraph[:max_chars]]


def chunk_documents(docs: list[dict[str, str]], max_chars: int = 1500) -> list[dict[str, Any]]:
    """Chunk documents on heading / paragraph boundaries.

    Each chunk is a dict with ``source``, ``chunk_index`` (per source),
    ``content`` and ``heading`` (empty string when the text had no heading).
    No chunk exceeds *max_chars* characters of body text.
    """
    max_chars = max(1, int(max_chars))
    chunks: list[dict[str, Any]] = []

    for doc in docs:
        source = doc.get("source", "")
        index = 0

        for heading, body in _split_sections(doc.get("content", "")):
            paragraphs: list[str] = []
            for raw in re.split(r"\n\s*\n", body):
                para = raw.strip()
                if not para:
                    continue
                if len(para) > max_chars:
                    paragraphs.extend(_split_long_paragraph(para, max_chars))
                else:
                    paragraphs.append(para)

            buffer: list[str] = []
            size = 0
            for para in paragraphs:
                extra = len(para) + (2 if buffer else 0)
                if buffer and size + extra > max_chars:
                    chunks.append({
                        "source": source,
                        "chunk_index": index,
                        "content": "\n\n".join(buffer),
                        "heading": heading,
                    })
                    index += 1
                    buffer = [para]
                    size = len(para)
                else:
                    buffer.append(para)
                    size += extra
            if buffer:
                chunks.append({
                    "source": source,
                    "chunk_index": index,
                    "content": "\n\n".join(buffer),
                    "heading": heading,
                })
                index += 1

    return chunks


# --------------------------------------------------------------------------- #
# Local TF-IDF index
# --------------------------------------------------------------------------- #


class TfidfIndex:
    """A tiny pure-NumPy TF-IDF vector store with cosine-similarity search."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks: list[dict[str, Any]] = chunks
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float64)
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self._build()

    # -- construction ------------------------------------------------------- #

    def _build(self) -> None:
        tokenized: list[list[str]] = [
            tokenize(f"{c.get('heading', '')} {c.get('content', '')}") for c in self.chunks
        ]
        vocabulary: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self.vocabulary = vocabulary

        n_docs = len(tokenized)
        n_terms = len(vocabulary)
        if n_docs == 0 or n_terms == 0:
            self.idf = np.zeros(n_terms, dtype=np.float64)
            self.matrix = np.zeros((n_docs, n_terms), dtype=np.float64)
            return

        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for row, tokens in enumerate(tokenized):
            for token in tokens:
                tf[row, vocabulary[token]] += 1.0

        df = np.count_nonzero(tf, axis=0).astype(np.float64)
        # Smoothed IDF: log((1 + N) / (1 + df)) + 1
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        self.matrix = _l2_normalize(tf * self.idf)

    # -- query -------------------------------------------------------------- #

    def _vectorize_query(self, query: str) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary), dtype=np.float64)
        for token in tokenize(query):
            position = self.vocabulary.get(token)
            if position is not None:
                vector[position] += 1.0
        vector *= self.idf
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the top *limit* chunks ranked by cosine similarity."""
        limit = max(0, int(limit))
        if limit == 0 or not self.chunks or self.matrix.size == 0:
            return []

        scores = self.matrix @ self._vectorize_query(query)
        top = min(limit, scores.shape[0])
        order = np.argsort(-scores, kind="stable")[:top]

        results: list[dict[str, Any]] = []
        for position in order:
            chunk = self.chunks[int(position)]
            score = float(scores[int(position)])
            if not math.isfinite(score):
                score = 0.0
            results.append({
                "source": chunk.get("source", ""),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "content": chunk.get("content", ""),
                "score": score,
                "heading": chunk.get("heading", ""),
                "distance": 1.0 - score,
            })
        return results


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of *matrix*, leaving all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


# Module-level singleton so the index is only computed once per process.
_INDEX: TfidfIndex | None = None
_INDEX_DIR: Path | None = None


def reset_index() -> None:
    """Clear the cached in-memory index (used by tests and re-ingestion)."""
    global _INDEX, _INDEX_DIR
    _INDEX = None
    _INDEX_DIR = None


def get_index(docs_dir: str | Path | None = None, rebuild: bool = False) -> TfidfIndex:
    """Return the cached :class:`TfidfIndex`, building it lazily on first use."""
    global _INDEX, _INDEX_DIR
    resolved = _resolve_docs_dir(docs_dir)
    if rebuild or _INDEX is None or _INDEX_DIR != resolved:
        chunks = chunk_documents(load_documents(resolved))
        _INDEX = TfidfIndex(chunks)
        _INDEX_DIR = resolved
        logger.info("Built local TF-IDF index: %d chunks from %s", len(chunks), resolved)
    return _INDEX


def _local_ingest(docs_dir: str | Path | None = None) -> int:
    """(Re)build the local index and return the number of indexed chunks."""
    return len(get_index(docs_dir, rebuild=True).chunks)


def _local_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search the local TF-IDF index."""
    return get_index().search(query, limit=limit)


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def get_backend() -> str:
    """Return ``"weaviate"`` if explicitly requested and importable, else ``"local"``."""
    if os.getenv("RAG_BACKEND", "").strip().lower() != "weaviate":
        return "local"
    try:
        import weaviate  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("RAG_BACKEND=weaviate but the weaviate package is unavailable: %s", exc)
        return "local"
    return "weaviate"


# --------------------------------------------------------------------------- #
# Weaviate backend (optional)
# --------------------------------------------------------------------------- #


def _client() -> Any:
    """Connect to the configured local Weaviate instance."""
    import weaviate

    from src.config.settings import get_settings

    settings = get_settings()
    host = settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(settings.weaviate_url.split(":")[-1]) if ":" in settings.weaviate_url else 8080
    return weaviate.connect_to_local(host=host, port=port)


def _weaviate_ensure_schema() -> None:
    """Create the Weaviate collection if it does not yet exist."""
    from weaviate.classes.config import Configure, DataType, Property

    from src.config.settings import get_settings

    settings = get_settings()
    with _client() as client:
        if client.collections.exists(settings.weaviate_class):
            return
        client.collections.create(
            name=settings.weaviate_class,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="heading", data_type=DataType.TEXT),
            ],
        )


def _weaviate_ingest(docs_dir: str | Path | None = None) -> int:
    """Push all chunks into the Weaviate collection and return the chunk count."""
    from src.config.settings import get_settings

    settings = get_settings()
    _weaviate_ensure_schema()
    chunks = chunk_documents(load_documents(docs_dir))
    with _client() as client:
        collection = client.collections.get(settings.weaviate_class)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(chunk)
    logger.info("Ingested %d chunks into %s", len(chunks), settings.weaviate_class)
    return len(chunks)


def _weaviate_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Run a ``near_text`` query against Weaviate."""
    from weaviate.classes.query import MetadataQuery

    from src.config.settings import get_settings

    settings = get_settings()
    with _client() as client:
        collection = client.collections.get(settings.weaviate_class)
        response = collection.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        results: list[dict[str, Any]] = []
        for obj in response.objects:
            distance = obj.metadata.distance
            distance = 1.0 if distance is None else float(distance)
            results.append({
                "source": obj.properties.get("source", ""),
                "chunk_index": int(obj.properties.get("chunk_index", 0) or 0),
                "content": obj.properties.get("content", ""),
                "score": 1.0 - distance,
                "heading": obj.properties.get("heading", "") or "",
                "distance": distance,
            })
        return results


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def ensure_schema() -> None:
    """Ensure the backing store is ready. A no-op for the local backend."""
    if get_backend() == "weaviate":
        try:
            _weaviate_ensure_schema()
            return
        except Exception as exc:
            logger.warning("Weaviate ensure_schema failed, using local backend: %s", exc)
    logger.debug("Local RAG backend requires no schema.")


def ingest_documents(docs_dir: str | Path | None = DEFAULT_DOCS_DIR) -> int:
    """Index every markdown document in *docs_dir*; returns the chunk count."""
    if get_backend() == "weaviate":
        try:
            return _weaviate_ingest(docs_dir)
        except Exception as exc:
            logger.warning("Weaviate ingest failed, falling back to local backend: %s", exc)
    return _local_ingest(docs_dir)


def search_documents(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the *limit* most relevant chunks for *query*.

    Each result has ``source``, ``chunk_index``, ``content``, ``score``
    (higher = more relevant), ``heading`` and ``distance`` (``1 - score``).
    """
    if get_backend() == "weaviate":
        try:
            return _weaviate_search(query, limit=limit)
        except Exception as exc:
            logger.warning("Weaviate search failed, falling back to local backend: %s", exc)
    return _local_search(query, limit=limit)
