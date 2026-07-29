"""Phase 6 smoke test: launch Phoenix, run one turn, verify spans were
captured with the funnel/token/retry attributes PRD Phase 6 asks for.

Attaches a ConsoleSpanExporter alongside Phoenix's own exporter purely so
this script can print + assert on span content without needing Phoenix's
query API -- the actual UI is at the printed session URL.
"""

from __future__ import annotations

import uuid

import phoenix as px
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from graph.build import graph_app
from observability.otel import setup_tracing


def _initial_state(user_input: str, dry_run: bool = False) -> dict:
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


class _CapturingExporter(ConsoleSpanExporter):
    def __init__(self):
        super().__init__()
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return super().export(spans)


def main() -> None:
    session = px.launch_app()
    print(f"Phoenix trace viewer: {session.url}")

    tracer_provider = setup_tracing(project_name="switchboard-test")
    capturer = _CapturingExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(capturer))

    with graph_app() as g:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        query = "Create a draft invoice for $500 consulting work for jane@acme.com"
        g.invoke(_initial_state(query, dry_run=True), config=config)

    print("\n--- captured span names ---")
    names = [s.name for s in capturer.spans]
    print(names)
    assert "plan_node" in names
    assert "retrieve_tools_node" in names
    assert "fill_node" in names
    assert "dry_run_node" in names

    retrieve_span = next(s for s in capturer.spans if s.name == "retrieve_tools_node")
    attrs = dict(retrieve_span.attributes)
    print("\n--- retrieve_tools_node attributes ---")
    for k, v in attrs.items():
        print(f"  {k}: {str(v)[:150]}")
    assert "funnel.dense_ids" in attrs
    assert "funnel.reranked_ids" in attrs
    assert "funnel.final_candidate_scores" in attrs

    plan_span = next(s for s in capturer.spans if s.name == "plan_node")
    plan_attrs = dict(plan_span.attributes)
    print("\n--- plan_node attributes ---")
    for k, v in plan_attrs.items():
        print(f"  {k}: {str(v)[:150]}")
    assert "llm.token_count.total" in plan_attrs

    fill_span = next(s for s in capturer.spans if s.name == "fill_node")
    fill_attrs = dict(fill_span.attributes)
    print("\n--- fill_node attributes ---")
    for k, v in fill_attrs.items():
        print(f"  {k}: {str(v)[:150]}")
    assert "tool.selected" in fill_attrs

    print("\nALL OK -- spans captured with funnel/token/tool attributes")


if __name__ == "__main__":
    main()
