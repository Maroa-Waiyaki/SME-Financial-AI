from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from agents import supervisor
from apps.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = supervisor.chat(
            question=request.question,
            history=[(m.role, m.content) for m in request.history],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing failed: {exc}",
        )
    return ChatResponse(
        response=result.get("final_response", ""),
        intent=result.get("current_intent", "GENERAL"),
        result=result.get("result", {}),
    )
