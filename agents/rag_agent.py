from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from apps.agents.llm import get_llm
from tools.rag import search_documents


def rag_agent(state: AgentState) -> dict:
    question = state["messages"][-1].content

    try:
        retrieved = search_documents(question, limit=5)
    except Exception as exc:
        message = (
            "I am unable to query the document store right now. "
            f"Please ensure the Weaviate service is available. Error: {exc}"
        )
        return {
            "final_response": message,
            "messages": [AIMessage(content=message)],
        }

    sources = [f"{r['source']} (chunk {r['chunk_index']})" for r in retrieved]
    context = "\n\n".join(f"[{r['source']}]\n{r['content']}" for r in retrieved)

    system = (
        "You are a documentation assistant for a Kenyan SME financial platform. "
        "Answer using the provided documents. Always cite the source of each fact. "
        "Distinguish retrieved documentation from business data. "
        "If the documents do not answer the question, say so."
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Provide a concise, cited answer. List the sources used."
    )

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        "final_response": f"{response.content}\n\nSources: {', '.join(sources)}",
        "result": {"retrieved": retrieved, "sources": sources},
        "messages": [AIMessage(content=response.content)],
    }
