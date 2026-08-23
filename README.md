# Kenya SME Financial Intelligence Platform

A production-grade, agentic AI platform that provides financial intelligence for small and medium-sized businesses in Kenya.

The platform combines **LangGraph-based multi-agent orchestration**, **deterministic financial analytics**, **machine learning** (credit risk, anomaly detection, forecasting), **RAG over financial documentation**, and a **Django + FastAPI + Docker** deployment.

## Problem Statement

Kenyan SMEs often lack accessible, data-driven financial insight. Business owners need to understand cash flow, expenses, revenue, credit risk, and anomalies without manually combing through M-Pesa, bank, sales, and invoice records. This project demonstrates how agentic AI can answer natural-language questions over realistic SME data while keeping calculations deterministic, explainable, and safe.

## Architecture

```mermaid
flowchart TB
    subgraph UI [Frontend]
        D[Django Dashboard]
    end

    D -->|HTTP| F[FastAPI API]
    F -->|JSON| S[LangGraph Supervisor]
    S -->|routes to| FA[Financial Agent]
    S -->|routes to| TA[Transaction Agent]
    S -->|routes to| CA[Credit Risk Agent]
    S -->|routes to| AA[Anomaly Agent]
    S -->|routes to| FO[Forecasting Agent]
    S -->|routes to| RA[RAG / Policy Agent]
    S -->|routes to| IA[Invoice Agent]

    CA -->|inference| SC[Credit Scorecard<br/>(logistic regression)]
    FA -->|queries| PG[(PostgreSQL)]
    TA -->|queries| PG
    AA -->|statistical| DET[Anomaly Detection]
    FO -->|forecasts| BASE[Baseline Forecaster]
    RA -->|near_text| WV[(Weaviate)]

    CEL[Celery Workers] -->|proactive monitoring| SC
    CEL -->|writes| PG

    PG -->|serves data| F
    SC -->|writes predictions/alerts| PG

    F -->|agent result| D
```

## Technology Stack

- **Backend:** Python 3.12+, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, Redis, Celery
- **Frontend:** Django 5
- **Agentic AI:** LangGraph, LangChain, OpenAI-compatible / Ollama LLMs, structured tool calling
- **Machine Learning:** NumPy/Pandas scorecard (core), optional scikit-learn, XGBoost, SHAP, MLflow
- **Vector DB:** Weaviate
- **MLOps:** MLflow with optional DagsHub remote
- **Infra:** Docker, Docker Compose, GitHub Actions
- **Testing:** pytest, ruff

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
- `GET  /api/v1/businesses/{business_id}/credit-risk`
- `POST /api/v1/chat`

Additional endpoints planned:

- `GET  /api/v1/businesses/{business_id}/forecast`
- `GET  /api/v1/businesses/{business_id}/anomalies`
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

## Local LLM (Ollama)

The platform can use any OpenAI-compatible endpoint, including a local [Ollama](https://ollama.com) server:

```bash
# 1. Install Ollama and pull a model
ollama pull llama3.2

# 2. Set your .env (host mode)
OPENAI_API_KEY=ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_BASE_URL=http://localhost:11434/v1

# 3. When running inside Docker Desktop, use the host gateway
LLM_BASE_URL=http://host.docker.internal:11434/v1
```

## MLOps

MLflow tracks credit-risk, anomaly, and forecasting experiments, metrics, and artifacts. The best credit-risk model is promoted through `candidate → staging → production` aliases and loaded at runtime.

### Credit-risk scorecard

The default credit-risk model is an L2-regularised logistic-regression scorecard implemented in NumPy. It runs in the core runtime image with no extra ML dependencies and produces exact, additive reason codes for every prediction.

Train it after generating synthetic data:

```bash
python scripts/train_scorecard.py
```

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
