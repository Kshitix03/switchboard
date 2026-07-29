"""Wire the graph: conditional edges, SQLite checkpointer (PRD section 5.3, 10)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from graph import nodes
from graph.state import AgentState

DB_PATH = Path(__file__).parent.parent / "switchboard.sqlite"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("plan", nodes.plan_node)
    builder.add_node("chat", nodes.chat_node)
    builder.add_node("system_search", nodes.system_search_node)
    builder.add_node("retrieve_tools", nodes.retrieve_tools_node)
    builder.add_node("bind", nodes.bind_node)
    builder.add_node("fill", nodes.fill_node)
    builder.add_node("validate", nodes.validate_node)
    builder.add_node("approve_gate", nodes.approve_gate_node)
    builder.add_node("execute", nodes.execute_node)
    builder.add_node("observe", nodes.observe_node)
    builder.add_node("clarify", nodes.clarify_node)
    builder.add_node("dry_run", nodes.dry_run_node)
    builder.add_node("rejected", nodes.rejected_node)
    builder.add_node("abort", nodes.abort_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", nodes.route_after_plan, {
        "chat": "chat", "system_search": "system_search", "retrieve_tools": "retrieve_tools", "abort": "abort",
    })
    builder.add_conditional_edges("retrieve_tools", nodes.route_after_retrieve, {
        "bind": "bind", "clarify": "clarify", "abort": "abort",
    })
    builder.add_edge("bind", "fill")
    builder.add_edge("fill", "validate")
    builder.add_conditional_edges("validate", nodes.route_after_validate, {
        "fill": "fill", "clarify": "clarify", "approve_gate": "approve_gate", "abort": "abort",
    })
    builder.add_conditional_edges("approve_gate", nodes.route_after_approve, {
        "dry_run": "dry_run", "execute": "execute", "rejected": "rejected", "abort": "abort",
    })
    builder.add_edge("execute", "observe")

    for terminal in ("chat", "system_search", "observe", "clarify", "dry_run", "rejected", "abort"):
        builder.add_edge(terminal, END)

    return builder


@contextmanager
def graph_app(db_path: Path = DB_PATH):
    """Yields a compiled graph with a SQLite checkpointer bound to db_path."""
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        builder = build_graph()
        yield builder.compile(checkpointer=checkpointer)
