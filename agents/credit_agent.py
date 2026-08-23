from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.config.settings import get_settings
from src.features.credit_features import get_business_features, load_data

logger = logging.getLogger(__name__)


def _serialise(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def _heuristic_risk(features: dict[str, Any]) -> dict[str, Any]:
    """Rule-based fallback when the trained scorecard is not available."""
    score = 0.0
    drivers = []

    if features.get("expense_ratio", 0) > 0.8:
        score += 25
        drivers.append("High expense-to-revenue ratio")
    if features.get("repayment_burden", 0) > 0.3:
        score += 20
        drivers.append("High loan repayment burden")
    if features.get("overdue_rate", 0) > 0.3:
        score += 20
        drivers.append("Large share of overdue invoices")
    if features.get("defaulted", 0) == 1:
        score += 20
        drivers.append("Existing defaulted loan")
    if features.get("negative_months", 0) > 2:
        score += 15
        drivers.append("Multiple negative cash-flow months")
    if features.get("anomaly_count", 0) > 10:
        score += 10
        drivers.append("Unusual transaction patterns")

    score = min(100, max(0, score))
    probability = round(score / 100, 2)
    level = "low" if score < 40 else "medium" if score < 70 else "high"
    return {
        "risk_score": score,
        "probability_of_default": probability,
        "risk_level": level,
        "model_version": "heuristic",
        "explanation_drivers": drivers,
        "note": "No trained scorecard found; using a heuristic risk score.",
    }


def _model_risk(features: dict[str, Any]) -> dict[str, Any]:
    """Assess risk using the L2-regularised logistic-regression scorecard."""
    from src.ml.scorecard import get_scorecard

    card = get_scorecard()
    assessment = card.assess(features, top_n=5)
    return {
        "risk_score": assessment["risk_score"],
        "probability_of_default": assessment["probability_of_default"],
        "risk_level": assessment["risk_level"],
        "model_version": f"scorecard-{assessment['model_version']}",
        "explanation_drivers": assessment["reason_codes"],
    }


def credit_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    question = state["messages"][-1].content

    if not business_id:
        message = "Please provide a business ID so I can assess credit risk."
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    settings = get_settings()
    data_dir = Path(settings.data_dir) if hasattr(settings, "data_dir") else Path("data/synthetic")
    dfs = load_data(data_dir)
    features = _serialise(get_business_features(dfs, business_id))

    try:
        risk = _serialise(_model_risk(features))
    except FileNotFoundError:
        logger.warning("Credit-risk scorecard not found; using heuristic fallback")
        risk = _serialise(_heuristic_risk(features))

    system = (
        "You are a credit-risk analyst for a Kenyan SME. "
        "Explain the risk result and the main drivers using the provided data. "
        "Clearly state whether the result comes from the ML model or a heuristic. "
        "Do not guarantee lending approval."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Risk assessment: {risk}\n\n"
        f"Features: {features}\n\n"
        "Provide a concise, evidence-backed explanation."
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": risk,
        "messages": [AIMessage(content=response.content)],
    }
