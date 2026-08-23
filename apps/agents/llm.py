from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config.settings import get_settings


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    settings = get_settings()
    provider = (settings.llm_provider or "openai").lower()

    # Ollama exposes an OpenAI-compatible endpoint at /v1 by default.
    # When using a local Ollama instance, no real API key is required; the
    # key field must still be a non-empty string to satisfy the OpenAI client.
    if provider == "ollama":
        base_url = settings.llm_base_url or "http://localhost:11434/v1"
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or "ollama",
            base_url=base_url,
            temperature=temperature if temperature is not None else settings.llm_temperature,
        )

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
    )
