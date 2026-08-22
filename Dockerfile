FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# System dependencies for compiled Python packages (XGBoost, LightGBM, SHAP, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
COPY apps/ ./apps/
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY scripts/ ./scripts/

RUN pip install --upgrade pip && \
    pip install -e .

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

ENV PYTHONPATH="${APP_HOME}:${APP_HOME}/src:${APP_HOME}/apps"

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
