from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.database.engine import get_db
from tools.transactions import (
    compare_transaction_periods,
    get_top_transactions,
    get_transaction_volume,
    summarize_transactions,
)


def _serialise(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def _extract_count(question: str) -> int:
    import re
    m = re.search(r"(\d+)\s*transactions", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if "largest" in question.lower() or "biggest" in question.lower():
        return 5
    return 5


def transaction_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    start = state.get("start_date")
    end = state.get("end_date")
    question = state["messages"][-1].content

    if not business_id or not start or not end:
        message = (
            "Please provide a business ID and a start/end date "
            "so I can look up the transactions."
        )
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    with get_db() as db:
        summary = _serialise(summarize_transactions(db, business_id, start, end))
        volume = get_transaction_volume(db, business_id, start, end)
        top_n = _extract_count(question)
        top = _serialise(get_top_transactions(db, business_id, start, end, n=top_n))

    data: dict[str, Any] = {"summary": summary, "volume": volume, "top_transactions": top}
    system = (
        "You are a Kenyan SME transaction analyst. Answer using the provided transaction data. "
        "Use KES for currency. Distinguish facts from recommendations. "
        "Do not call every anomaly fraud; use 'potential anomaly' when relevant."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Data: {data}\n\n"
        "Provide a concise, evidence-backed answer."
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
