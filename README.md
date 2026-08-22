# Kenya SME Financial Intelligence Platform

A production-grade, agentic AI platform that provides financial intelligence for small and medium-sized businesses in Kenya.

The platform combines **LangGraph-based multi-agent orchestration**, **deterministic financial analytics**, **machine learning** (credit risk, anomaly detection, forecasting), **RAG over financial documentation**, and a **Django + FastAPI + Docker** deployment.

## Problem Statement

Kenyan SMEs often lack accessible, data-driven financial insight. Business owners need to understand cash flow, expenses, revenue, credit risk, and anomalies without manually combing through M-Pesa, bank, sales, and invoice records. This project demonstrates how agentic AI can answer natural-language questions over realistic SME data while keeping calculations deterministic, explainable, and safe.

## Architecture

```
Django Dashboard
       ↓
FastAPI API
       ↓
LangGraph Supervisor
       ↓
┌─────────────────────────────────────────────────┐
│  Financial Agent                                │
│  Transaction Agent                              │
│  Credit Risk Agent                              │
│  Forecasting Agent                              │
│  Anomaly Detection Agent                        │
│  Document / RAG Agent                           │
└─────────────────────────────────────────────────┘
       ↓
Tool Layer (PostgreSQL, ML models, Weaviate, Forecasting)
       ↓
Response with evidence
```

## Technology Stack

- **Backend:** Python 3.12+, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Redis, Celery
- **Frontend:** Django 5
- **Agentic AI:** LangGraph, LangChain, OpenAI-compatible LLMs, structured tool calling
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost, Prophet, SHAP, MLflow
- **Vector DB:** Weaviate
- **MLOps:** MLflow with optional DagsHub remote
- **Infra:** Docker, Docker Compose, GitHub Actions
- **Testing:** pytest, ruff, mypy

## Project Structure

```
kenya-sme-financial-intelligence/
├── apps/                  # FastAPI, Django, shared agent entrypoints
│   ├── api/
│   ├── django_app/
│   └── agents/
├── src/                   # Core libraries
│   ├── config/
│   ├── data/              # generator, ETL, validation
│   ├── database/          # SQLAlchemy base, models, engine
│   ├── features/          # ML feature engineering
│   ├── financial/         # deterministic calculations
│   ├── forecasting/       # forecasting baseline + training
│   └── ml/                # credit-risk training + MLflow tracking
├── agents/                # LangGraph specialist agents
├── tools/                 # deterministic tool definitions
├── docs/                  # RAG documents
├── data/                  # raw, processed, synthetic data
├── tests/                 # pytest suite
├── scripts/               # helper scripts
├── models/                # trained model artifacts
├── .github/workflows/     # CI/CD
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Getting Started

1. Copy `.env.example` to `.env` and set your keys:
   ```bash
   cp .env.example .env
   ```

2. Build and start all services:
   ```bash
   docker compose up --build -d
   ```

3. Generate synthetic data (inside the `fastapi` container or locally after installing deps):
   ```bash
   python -m src.data.generate --n-businesses 100 --n-months 6
   python -m src.data.ingest --input-dir data/synthetic
   ```

4. Run the smoke tests without the full stack:
   ```bash
   python scripts/smoke_tests.py
   ```

5. Run the full test suite in Docker:
   ```bash
   docker compose run --rm fastapi pytest
   ```

## API (FastAPI)

Implemented endpoints:

- `GET  /api/v1/health`
- `GET  /api/v1/businesses/{business_id}`
- `GET  /api/v1/businesses/{business_id}/financial-summary`
- `POST /api/v1/chat`

Additional endpoints planned:

- `GET  /api/v1/businesses/{business_id}/forecast`
- `GET  /api/v1/businesses/{business_id}/anomalies`
- `GET  /api/v1/businesses/{business_id}/credit-risk`
- `POST /api/v1/reports`

## Agentic AI

A LangGraph supervisor routes user questions to specialist agents. Each agent calls deterministic tools that query PostgreSQL or invoke ML models. No agent is called unless the supervisor determines it is required.

Implemented agents:

- **Financial Agent** — revenue, expenses, profit, profit margin, cash flow, summary, comparison
- **Transaction Agent** — transaction summary, top transactions, volume, period comparison
- **Credit Risk Agent** — XGBoost model with SHAP-style heuristic fallback
- **Forecasting Agent** — 30/60/90-day revenue, expense, cash-flow baseline
- **Anomaly Detection Agent** — z-score, off-hours, and frequency anomalies
- **RAG Agent** — Weaviate-backed document Q&A with citations

## MLOps

MLflow tracks credit-risk, anomaly, and forecasting experiments, metrics, and artifacts. The best credit-risk model is promoted through `candidate → staging → production` aliases and loaded at runtime.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR:

- `ruff check` and `ruff format --check`
- `python -m compileall -q .`
- `python scripts/smoke_tests.py`
- `pytest` (full suite in container)
- `docker build .`

## Safety & Disclaimer

- AI-generated recommendations are informational only.
- Credit-risk outputs are model predictions, not lending decisions.
- Forecasts are estimates and should be verified.
- The platform never uses real private financial data; all data is synthetic.

## License

MIT
