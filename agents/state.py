from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the LangGraph supervisor and specialist agents."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_intent: str
    business_id: str | None
    start_date: str | None
    end_date: str | None
    result: dict
    final_response: str
