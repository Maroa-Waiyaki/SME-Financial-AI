from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import businesses, chat, documents, health, reports


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kenya SME Financial Intelligence API",
        version="0.1.0",
        description="Agentic AI financial intelligence for Kenyan SMEs.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(businesses.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(reports.router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": "Kenya SME Financial Intelligence API", "docs": "/docs"}

    return app


app = create_app()
