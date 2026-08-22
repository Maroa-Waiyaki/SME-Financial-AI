from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.database.engine import get_db
from tools.anomaly import detect_anomalies


def _serialise(obj: Any) -> Any:
    from decimal import Decimal

    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def anomaly_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    start = state.get("start_date")
    end = state.get("end_date")
    question = state["messages"][-1].content

    if not business_id or not start or not end:
        message = (
            "Please provide a business ID and a start/end date "
            "so I can scan for anomalous transactions."
        )
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    with get_db() as db:
        anomalies = _serialise(detect_anomalies(db, business_id, start, end))

    data = {"anomalies": anomalies, "count": len(anomalies)}
    period = f"{start} to {end}"
    system = (
        "You are an anomaly-detection analyst for a Kenyan SME. "
        "Explain each flagged transaction as a 'potential anomaly', not as confirmed fraud. "
        "Use KES with two decimal places and normal ASCII punctuation. "
        "Ground every statement in the provided data. If no anomalies are found, say so clearly."
    )
    prompt = (
        f"User question: {question}\n"
        f"Period: {period}\n\n"
        f"Data: {data}\n\n"
        "Format:\n"
        f"**Anomaly Scan ({period})**\n"
        "- Number of potential anomalies found\n"
        "- Concise description of each, with amounts and reasons\n\n"
        "**Recommendation** (one short next step, only if relevant)"
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
