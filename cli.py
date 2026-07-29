"""CLI conversation loop over the Switchboard graph (PRD section 7, 10 Phase 4).

Two modes:
  --dry-run     prints the exact call that would be made and stops, never executes
  --interactive (default) asks for y/n confirmation at the approval gate
"""

from __future__ import annotations

import os
import sys
import uuid

import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph.build import graph_app
from observability.otel import setup_tracing

# Windows console defaults to cp1252, which can't encode the emoji Phoenix
# prints on launch -- reconfigure stdout/stderr rather than let that crash
# the whole CLI over a banner message.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Phoenix's default OTLP gRPC port (4317) collides with other local tracing
# tools (e.g. Jaeger) some dev machines already run. Only set a fallback if
# the user hasn't already configured one themselves.
os.environ.setdefault("PHOENIX_GRPC_PORT", "4327")

app = typer.Typer(add_completion=False)


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


def _handle_result(g, result: dict, config: dict) -> None:
    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        typer.echo(
            f"\n[APPROVAL REQUIRED] {payload['operation'].upper()} risk={payload['risk']} "
            f"tool={payload['tool_id']}"
        )
        typer.echo(f"args: {payload['args']}")
        approved = typer.confirm("Approve this call?")
        resumed = g.invoke(Command(resume={"approved": approved}), config=config)
        _handle_result(g, resumed, config)
        return

    messages = result.get("messages", [])
    if messages:
        typer.echo(f"\n{messages[-1].content}\n")


@app.command()
def chat(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the exact call and stop; never execute."),
    thread: str = typer.Option(None, "--thread", help="Reuse an existing conversation thread id."),
    no_trace: bool = typer.Option(False, "--no-trace", help="Skip launching the local Phoenix trace viewer."),
) -> None:
    thread_id = thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if not no_trace:
        import phoenix as px
        session = px.launch_app()
        setup_tracing(project_name="switchboard")
        if session is not None:
            typer.echo(f"Phoenix trace viewer: {session.url}")

    typer.echo(f"Switchboard CLI -- thread={thread_id} dry_run={dry_run}. Ctrl+C to exit.\n")

    with graph_app() as g:
        while True:
            try:
                user_input = typer.prompt(">")
            except (EOFError, KeyboardInterrupt):
                typer.echo("\nbye")
                break
            result = g.invoke(_initial_state(user_input, dry_run), config=config)
            _handle_result(g, result, config)


if __name__ == "__main__":
    app()
