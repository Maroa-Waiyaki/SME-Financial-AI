from __future__ import annotations

from fastapi import APIRouter

from apps.api.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health_check() -> HealthOut:
    return HealthOut(status="ok")
