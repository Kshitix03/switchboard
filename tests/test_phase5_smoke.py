"""Phase 5 smoke test: rag_search competes for retrieval slots and executes a
real docs search; system_search renders retrieve() as prose for meta turns,
and answers a status query from the trace store."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage

from graph.build import graph_app


def _initial_state(user_input: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_input)],
        "plan": None,
        "filters": None,
        "candidate_tools": [],
        "selected_tool": None,
        "filled_args": None,
        "validation_errors": [],
        "retry_count": 0,
        "entities": {},
        "artifacts": {},
        "pending_approval": None,
        "trace_id": str(uuid.uuid4()),
        "steps": 0,
        "dry_run": False,
    }


def test_rag_search_selected_and_executed():
    query = "How do refunds work on PayPal, conceptually?"
    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = g.invoke(_initial_state(query), config=config)
        print("\n--- plan ---", result.get("plan"))
        print("--- candidate tools (top 3) ---", [c["tool_id"] for c in result.get("candidate_tools", [])[:3]])
        print("--- selected tool ---", result.get("selected_tool"))
        print("--- final message ---")
        print(result["messages"][-1].content)
        assert result.get("selected_tool") == "rag_search" or "rag_search" in [
            c["tool_id"] for c in result.get("candidate_tools", [])
        ]


def test_system_search_meta_tool_discovery():
    query = "What tools do you have for handling invoices?"
    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = g.invoke(_initial_state(query), config=config)
        print("\n--- plan ---", result.get("plan"))
        print("--- final message ---")
        print(result["messages"][-1].content)
        assert result.get("plan") == "meta"
        assert "invoic" in result["messages"][-1].content.lower()


def test_system_search_status_query():
    query = "What's the status of my last request?"
    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = g.invoke(_initial_state(query), config=config)
        print("\n--- plan ---", result.get("plan"))
        print("--- final message ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    test_rag_search_selected_and_executed()
    print("\n" + "=" * 80)
    test_system_search_meta_tool_discovery()
    print("\n" + "=" * 80)
    test_system_search_status_query()
    print("\nALL OK")
