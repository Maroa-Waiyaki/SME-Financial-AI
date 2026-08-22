from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from tools.forecasting import forecast_cashflow, forecast_revenue


def _serialise(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def forecasting_agent(state: AgentState) -> dict:
    business_id = state.get("business_id")
    question = state["messages"][-1].content

    if not business_id:
        message = (
            "Please provide a business ID so I can forecast revenue, expenses, and cash flow."
        )
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    revenue_forecast = _serialise(forecast_revenue(business_id))
    cashflow_forecast = _serialise(forecast_cashflow(business_id))

    data: dict[str, Any] = {
        "revenue_forecast": revenue_forecast,
        "cash_flow_forecast": cashflow_forecast,
    }
    system = (
        "You are a financial forecasting analyst for a Kenyan SME. "
        "Present the 30, 60, and 90-day projections clearly. "
        "Use KES for currency. Explain the assumptions and confidence bands. "
        "Make it clear that forecasts are estimates, not guarantees."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Forecasts: {data}\n\n"
        "Provide a concise answer with the projected numbers and the key assumptions."
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": response.content,
        "result": data,
        "messages": [AIMessage(content=response.content)],
    }
