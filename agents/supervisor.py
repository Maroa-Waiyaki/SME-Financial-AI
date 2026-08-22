from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.anomaly_agent import anomaly_agent
from agents.credit_agent import credit_agent
from agents.financial_agent import financial_agent
from agents.forecasting_agent import forecasting_agent
from agents.rag_agent import rag_agent
from agents.state import AgentState
from agents.transaction_agent import transaction_agent
from apps.agents.llm import get_llm

INTENTS = [
    "FINANCIAL_ANALYSIS",
    "TRANSACTION_ANALYSIS",
    "CREDIT_RISK",
    "FORECAST",
    "ANOMALY_DETECTION",
    "DOCUMENT_QUERY",
    "REPORT_GENERATION",
    "GENERAL",
    "CLARIFICATION",
]


class ClassifierOutput:
    """We use a Pydantic model at runtime; this stub keeps typing minimal."""

    def __init__(self, **kwargs):  # pragma: no cover
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_classifier():
    from pydantic import BaseModel, Field

    class _ClassifierOutput(BaseModel):
        intent: str = Field(
            ...,
            description=(
                "One of FINANCIAL_ANALYSIS, TRANSACTION_ANALYSIS, CREDIT_RISK, "
                "FORECAST, ANOMALY_DETECTION, DOCUMENT_QUERY, REPORT_GENERATION, "
                "GENERAL, or CLARIFICATION."
            ),
        )
        business_id: str | None = Field(None, description="Business ID if known; e.g. B000001")
        start_date: str | None = Field(None, description="Start date in YYYY-MM-DD format")
        end_date: str | None = Field(None, description="End date in YYYY-MM-DD format")
        clarification: str = Field(
            "",
            description="If clarification is needed, ask a short follow-up question.",
        )

    return _ClassifierOutput


def _clean_text(text: str) -> str:
    # Remove trailing spaces per line and collapse more than two newlines
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_business_id(text: str) -> str | None:
    m = re.search(r"(B\d{6})\b", text)
    return m.group(1) if m else None


def _extract_dates(text: str) -> tuple[str | None, str | None]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if len(dates) >= 2:
        return dates[0], dates[1]
    if len(dates) == 1:
        return dates[0], "2023-12-31"
    return None, None


def classify(state: AgentState) -> dict:
    system = (
        "You are the supervisor for a Kenyan SME financial intelligence platform. "
        "Analyse the user's question and extract the intent, business ID, and date range. "
        "Use YYYY-MM-DD for dates. If a date is missing but a relative term is used "
        "(e.g. 'this month'), use the closest reasonable range. "
        "If you cannot identify the required information, set intent to CLARIFICATION and ask. "
        f"Allowed intents: {', '.join(INTENTS)}"
    )
    llm = get_llm().with_structured_output(_build_classifier())
    last_message = state["messages"][-1]
    out = llm.invoke([SystemMessage(content=system), last_message])

    question = last_message.content
    conversation = " ".join(m.content for m in state["messages"])
    business_id = out.business_id or _extract_business_id(question) or _extract_business_id(conversation)
    start_date, end_date = _extract_dates(question)
    if not start_date:
        start_date = out.start_date or _extract_dates(conversation)[0]
    if not end_date:
        end_date = out.end_date or _extract_dates(conversation)[1]

    # Default to the full 2023 range if the specialist still needs dates
    if business_id and not start_date:
        start_date = "2023-01-01"
    if business_id and not end_date:
        end_date = "2023-12-31"

    if not business_id:
        return {
            "current_intent": "CLARIFICATION",
            "final_response": "Please provide a business ID (for example B000001) so I can look up the data.",
        }

    return {
        "current_intent": out.intent if out.intent != "CLARIFICATION" else "FINANCIAL_ANALYSIS",
        "business_id": business_id,
        "start_date": start_date,
        "end_date": end_date,
        "final_response": out.clarification if out.intent == "CLARIFICATION" else "",
    }


def route(state: AgentState) -> str:
    intent = state.get("current_intent", "GENERAL")
    if intent == "FINANCIAL_ANALYSIS":
        return "financial"
    if intent == "TRANSACTION_ANALYSIS":
        return "transaction"
    if intent == "ANOMALY_DETECTION":
        return "anomaly"
    if intent == "CREDIT_RISK":
        return "credit"
    if intent == "FORECAST":
        return "forecast"
    if intent == "DOCUMENT_QUERY":
        return "rag"
    if intent == "CLARIFICATION":
        return "general"
    return "general"


def general_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    system = (
        "You are a helpful Kenyan SME financial assistant. "
        "You do not have access to live data unless the question has been routed to a "
        "specialist agent. Answer general questions, ask for clarification, or explain "
        "what you can do. Be concise, professional, and safe: never invent financial numbers. "
        "Use normal ASCII punctuation."
    )
    if state.get("final_response"):
        response_text = state["final_response"]
    else:
        llm = get_llm()
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=question)])
        response_text = response.content

    return {
        "final_response": response_text,
        "messages": [AIMessage(content=response_text)],
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify)
    builder.add_node("financial", financial_agent)
    builder.add_node("transaction", transaction_agent)
    builder.add_node("anomaly", anomaly_agent)
    builder.add_node("credit", credit_agent)
    builder.add_node("forecast", forecasting_agent)
    builder.add_node("rag", rag_agent)
    builder.add_node("general", general_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {
            "financial": "financial",
            "transaction": "transaction",
            "anomaly": "anomaly",
            "credit": "credit",
            "forecast": "forecast",
            "rag": "rag",
            "general": "general",
        },
    )
    builder.add_edge("financial", END)
    builder.add_edge("transaction", END)
    builder.add_edge("anomaly", END)
    builder.add_edge("credit", END)
    builder.add_edge("forecast", END)
    builder.add_edge("rag", END)
    builder.add_edge("general", END)
    return builder.compile()


graph = build_graph()


def chat(question: str, history: list[tuple[str, str]] | None = None) -> dict:
    messages: list[SystemMessage | HumanMessage | AIMessage] = []
    if history:
        for role, content in history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=question))
    result = graph.invoke({"messages": messages})
    result["final_response"] = _clean_text(result.get("final_response", ""))
    return result
