from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.database.engine import get_db
from tools.financial_reports import generate_balance_sheet, generate_pnl_statement

logger = logging.getLogger(__name__)


def _serialise(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def report_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    start = state.get("start_date") or "2023-01-01"
    end = state.get("end_date") or "2023-12-31"
    question = state["messages"][-1].content

    if not business_id:
        message = "Please provide a business ID (e.g. B000001) to generate financial statements."
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    with get_db() as db:
        pnl = _serialise(generate_pnl_statement(db, business_id, start, end))
        bs = _serialise(generate_balance_sheet(db, business_id, end))

    data = {"pnl": pnl, "balance_sheet": bs}

    system = (
        "You are a professional Kenyan SME corporate accountant and financial advisor. "
        "Use the provided deterministic Profit & Loss (P&L) and Balance Sheet data. "
        "Format cleanly in markdown tables with KES currency and percentages. "
        "Highlight gross margin, operating expenses, net profit/EBITDA, total assets, debt liabilities, and owner's net worth. "
        "Inform the user they can also download the full PowerPoint (.pptx) deck and Excel (.xlsx) financial model from the 'P&L & Balance Sheet' dashboard tab."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Statement Data for {business_id}:\n{data}\n\n"
        "Provide a complete, professional Executive Financial Statement summary."
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
