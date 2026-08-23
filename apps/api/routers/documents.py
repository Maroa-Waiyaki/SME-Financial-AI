from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from apps.api.auth import get_current_user
from tools.document_parser import extract_text_from_file
from tools.rag import get_index

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
UPLOAD_DIR = DOCS_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    business_id: str = Form(None),
    _: str = Depends(get_current_user),
) -> dict[str, str | int]:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    content = await file.read()
    text = extract_text_from_file(content, file.filename)

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract any readable text from this file format or file is empty.",
        )

    # Save as Markdown file in docs directory
    prefix = f"{business_id}_" if business_id else ""
    safe_name = f"{prefix}{Path(file.filename).stem}.md"
    target_path = UPLOAD_DIR / safe_name
    
    header = f"# Document: {file.filename}\n"
    if business_id:
        header += f"**Business ID:** {business_id}\n\n"
    
    target_path.write_text(header + text, encoding="utf-8")

    # Trigger automatic index rebuild for RAG
    get_index(docs_dir=DOCS_DIR, rebuild=True)
    get_index(docs_dir=UPLOAD_DIR, rebuild=True)

    return {
        "status": "success",
        "filename": file.filename,
        "saved_as": safe_name,
        "characters_extracted": len(text),
        "message": "Document uploaded and indexed into RAG successfully. You can now query it in AI Chat.",
    }
