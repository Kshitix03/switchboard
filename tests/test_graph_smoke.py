"""Non-interactive smoke test: invoice creation hits the approval gate and
dry-run refuses to execute (PRD Phase 4 acceptance criterion)."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph.build import graph_app


def _initial_state(user_input: str, dry_run: bool) -> dict:
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
        "dry_run": dry_run,
    }


def test_dry_run_refuses_execute():
    query = "Create a draft invoice for $500 consulting work for jane@acme.com"
    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = g.invoke(_initial_state(query, dry_run=True), config=config)

        assert "__interrupt__" not in result or not result["__interrupt__"], "dry run should never interrupt"
        last_msg = result["messages"][-1].content
        print("\n--- DRY RUN result ---")
        print(last_msg)
        assert "[DRY RUN]" in last_msg
        assert "execute" not in result.get("artifacts", {}).get("last_execution", {}).get("status", "")


def test_interactive_hits_approval_and_executes():
    query = "Create a draft invoice for $500 consulting work for jane@acme.com"
    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = g.invoke(_initial_state(query, dry_run=False), config=config)

        print("\n--- pre-approval state keys ---", list(result.keys()))
        assert result.get("__interrupt__"), "write-class tool should pause for approval"
        payload = result["__interrupt__"][0].value
        print("--- approval payload ---")
        print(payload)
        assert payload["operation"] == "write"

        resumed = g.invoke(Command(resume={"approved": True}), config=config)
        print("\n--- post-approval result ---")
        print(resumed["messages"][-1].content)
        print("execution:", resumed["artifacts"].get("last_execution"))
        assert resumed["artifacts"]["last_execution"]["status"] == "mocked"


if __name__ == "__main__":
    test_dry_run_refuses_execute()
    print("\n" + "=" * 80)
    test_interactive_hits_approval_and_executes()
    print("\nALL OK")
