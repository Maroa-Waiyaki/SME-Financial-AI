from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from src.database.engine import get_db
from tools.financial import compare_periods, get_financial_summary


def _previous_period(start: str, end: str) -> tuple[str, str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = e - s
    prev_e = s - timedelta(days=1)
    prev_s = prev_e - delta
    return prev_s.strftime("%Y-%m-%d"), prev_e.strftime("%Y-%m-%d")


def _serialise(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def financial_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    start = state.get("start_date")
    end = state.get("end_date")
    question = state["messages"][-1].content

    if not business_id or not start or not end:
        message = (
            "Please provide a business ID and a start/end date "
            "(for example: '2023-01-01' to '2023-01-31') so I can look up the data."
        )
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    with get_db() as db:
        summary = _serialise(get_financial_summary(db, business_id, start, end))
        prev_start, prev_end = _previous_period(start, end)
        comparison = _serialise(compare_periods(db, business_id, start, end, prev_start, prev_end))

    data = {"summary": summary, "comparison": comparison}
    period = f"{start} to {end}"
    system = (
        "You are a professional Kenyan SME financial analyst. Use only the provided data. "
        "Respond in clear, plain English using normal ASCII punctuation (standard hyphens and spaces). "
        "Use KES with two decimal places. Clearly label the period. "
        "Keep facts and recommendations in separate sections. If no recommendation is warranted, omit it."
    )
    prompt = (
        f"User question: {question}\n"
        f"Period: {period}\n\n"
        f"Data: {data}\n\n"
        "Format:\n"
        f"**Facts ({period})**\n"
        "- Revenue: KES ...\n"
        "- Expenses: KES ...\n"
        "- Profit: KES ... (profit margin ...%)\n"
        "- Other relevant metrics\n\n"
        "**Recommendation** (one short, practical suggestion for a Kenyan SME, only if relevant)"
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
