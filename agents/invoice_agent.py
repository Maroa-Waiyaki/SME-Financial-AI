from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.database.engine import get_db
from tools.invoices import get_last_invoices, get_outstanding_invoices


def _serialise(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def _extract_count(question: str) -> int:
    m = re.search(r"(\d+)\s*invoices?", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 5


def invoice_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    question = state["messages"][-1].content

    if not business_id:
        message = "Please provide a business ID (for example B000001) so I can look up invoices."
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    n = _extract_count(question)
    with get_db() as db:
        last_invoices = _serialise(get_last_invoices(db, business_id, n=n))
        outstanding = _serialise(get_outstanding_invoices(db, business_id))

    total_outstanding = sum(i["outstanding_amount"] for i in outstanding)
    data = {
        "last_invoices": last_invoices,
        "outstanding_count": len(outstanding),
        "total_outstanding": total_outstanding,
    }

    system = (
        "You are a Kenyan SME accounts-receivable assistant. "
        "Use only the provided invoice data. Use KES with two decimal places. "
        "Use normal ASCII punctuation. "
        "Clearly separate facts from recommendations. If no invoices are found, say so."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Data: {data}\n\n"
        "Format:\n"
        "**Invoices**\n"
        "- List each invoice with ID, date, amount, amount paid, outstanding, and status\n\n"
        "**Outstanding summary**\n"
        "- Total outstanding and count\n\n"
        "**Recommendation** (one short collection or follow-up suggestion, only if relevant)"
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
