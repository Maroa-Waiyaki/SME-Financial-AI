from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_documents(docs_dir: str | Path) -> list[dict[str, str]]:
    docs_dir = Path(docs_dir)
    docs: list[dict[str, str]] = []
    for path in docs_dir.glob("*.md"):
        docs.append({
            "source": path.name,
            "content": path.read_text(encoding="utf-8"),
        })
    return docs


def chunk_documents(docs: list[dict[str, str]], max_chars: int = 1500) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        text = doc["content"]
        for i in range(0, len(text), max_chars):
            chunk = text[i : i + max_chars]
            chunks.append({
                "source": doc["source"],
                "chunk_index": i // max_chars,
                "content": chunk.strip(),
            })
    return chunks


def _client():
    import weaviate
    from src.config.settings import get_settings

    settings = get_settings()
    return weaviate.connect_to_local(
        host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
        port=int(settings.weaviate_url.split(":")[-1]) if ":" in settings.weaviate_url else 8080,
    )


def ensure_schema() -> None:
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
            ],
        )


def ingest_documents(docs_dir: str | Path) -> int:
    from src.config.settings import get_settings

    settings = get_settings()
    ensure_schema()
    chunks = chunk_documents(load_documents(docs_dir))
    with _client() as client:
        collection = client.collections.get(settings.weaviate_class)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(chunk)
    logger.info(f"Ingested {len(chunks)} into {settings.weaviate_class}")
    return len(chunks)


def search_documents(query: str, limit: int = 5) -> list[dict[str, Any]]:
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
            results.append({
                "source": obj.properties.get("source", ""),
                "chunk_index": obj.properties.get("chunk_index", 0),
                "content": obj.properties.get("content", ""),
                "distance": obj.metadata.distance,
            })
        return results
