# Project Reference

## Kenya SME Financial Intelligence Platform

This file tracks project-specific conventions, commands, and configuration for AI assistants.

## Primary deployment

Docker Compose is the primary local deployment method. The main orchestration file is `docker-compose.yml` in the repository root.

## Important environment files

- `.env.example` - template for all environment variables; never commit real values.
- `.env` - ignored by git; contains local secrets and service settings.

## Directories

- `apps/api` - FastAPI application
- `apps/django_app` - Django dashboard
- `apps/agents` - LLM configuration helpers
- `src/data` - synthetic data generator and ETL
- `src/database` - SQLAlchemy models and engine
- `src/features` - ML feature engineering
- `src/financial` - deterministic calculations
- `src/forecasting` - forecasting baseline and training
- `src/ml` - credit-risk training and MLflow tracking
- `agents/` - LangGraph agent implementations
- `tools/` - deterministic tool definitions
- `docs/` - RAG documents
- `tests/` - pytest suite
- `scripts/` - helper scripts

## Package layout

`src/` layout is used. Source packages are discovered from `src/` by `pyproject.toml`. The image and local Python path also include the repository root so `agents/`, `tools/`, `apps/`, and `src/` are importable.

## Data generation

- Sample: `python -m src.data.generate --n-businesses 100 --n-months 6`
- Full: `python -m src.data.generate --n-businesses 1000 --n-months 18`
- Output: `data/synthetic/*.csv`
- Validate: `python scripts/validate_synthetic.py`

## ETL / Database

- Create tables: `Base.metadata.create_all` is called by `src.data.etl.load_to_postgres`
- Dry-run validation: `python -m src.data.ingest --dry-run`
- Load into PostgreSQL: `python -m src.data.ingest --database-url <URL>`

## API

- Run FastAPI: `uvicorn apps.api.main:app --reload`
- Health endpoint: `GET /api/v1/health`
- Business profile: `GET /api/v1/businesses/{business_id}`
- Financial summary: `GET /api/v1/businesses/{business_id}/financial-summary?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- Chat: `POST /api/v1/chat`

## Agent routing

The LangGraph supervisor in `agents/supervisor.py` routes to:

- `financial` for revenue, expenses, profit, cash flow, summaries
- `transaction` for transaction analysis and top transactions
- `anomaly` for suspicious transaction detection
- `credit` for credit-risk assessment
- `forecast` for 30/60/90-day revenue, expense, cash-flow forecasts
- `rag` for document Q&A
- `general` for greetings and small talk

## Testing

- Smoke tests: `python scripts/smoke_tests.py`
- Full suite: `pytest` (requires all dependencies)
- Syntax check: `python -m compileall -q .`

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR:

- lint: `ruff check .` and `ruff format --check .`
- compile: `python -m compileall -q .`
- smoke: `python scripts/smoke_tests.py`
- test: `pytest`
- docker: `docker build . -t kenya-sme-fi:latest`

## Notes

- All financial calculations must be deterministic Python/SQL functions; never rely on the LLM for arithmetic.
- No real M-Pesa or customer data is used; data is synthetic and clearly labeled.
- Never commit secrets, API keys, or `.env`.
