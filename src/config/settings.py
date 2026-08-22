from __future__ import annotations

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = True
    default_currency: str = "KES"
    data_dir: str = "data/synthetic"
    models_dir: str = "models"

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql://sme_user:sme_pass@localhost:5432/sme_financial_intelligence"
    )
    database_pool_size: int = 10

    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    celery_broker_url: RedisDsn = Field(default="redis://localhost:6379/1")
    celery_result_backend: RedisDsn = Field(default="redis://localhost:6379/2")

    # LLM / Agentic AI
    openai_api_key: str | None = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_temperature: float = 0.0

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_class: str = "SmeDocument"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_username: str | None = None
    mlflow_password: str | None = None
    mlflow_experiment: str = "kenya-sme-financial-intelligence"

    # Security
    secret_key: str = "change-me-in-production"
    jwt_secret: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60


def get_settings() -> Settings:
    return Settings()
