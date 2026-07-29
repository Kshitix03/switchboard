"""OpenTelemetry instrumentation, exported to a locally running Phoenix
instance (PRD section 12: "Phoenix over LangSmith -- open source, OTel
native, runs locally"). Every graph node is wrapped in a span via
@traced_node(); span attributes capture exactly what PRD Phase 6 asks for:
candidate ids at each funnel stage, rerank scores, tokens, retries, final
tool. Latency is automatic (span duration).
"""

from __future__ import annotations

import functools
import json

from langgraph.errors import GraphInterrupt
from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode

_tracer = None
_tracer_provider = None


def setup_tracing(project_name: str = "switchboard", endpoint: str | None = None):
    """Registers a TracerProvider exporting to Phoenix. Safe to call more
    than once (idempotent). Returns the TracerProvider (use get_tracer() for
    the Tracer used to create spans)."""
    global _tracer, _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider
    from phoenix.otel import register

    _tracer_provider = register(project_name=project_name, endpoint=endpoint, auto_instrument=False)
    _tracer = _tracer_provider.get_tracer("switchboard.graph")
    return _tracer_provider


def get_tracer():
    if _tracer is None:
        setup_tracing()
    return _tracer


def record_tokens(usage_metadata) -> None:
    """Call right after a Gemini generate_content call to attach token counts
    to whichever node span is currently active."""
    if usage_metadata is None:
        return
    span = trace_api.get_current_span()
    span.set_attribute("llm.token_count.prompt", getattr(usage_metadata, "prompt_token_count", 0) or 0)
    span.set_attribute("llm.token_count.completion", getattr(usage_metadata, "candidates_token_count", 0) or 0)
    span.set_attribute("llm.token_count.total", getattr(usage_metadata, "total_token_count", 0) or 0)


def record_funnel(funnel: dict[str, list[str]]) -> None:
    """Call from retrieve_tools_node with router.RetrievalResult.funnel to
    record dense/sparse/fused/reranked ids at each stage on the active span."""
    span = trace_api.get_current_span()
    for stage, ids in funnel.items():
        span.set_attribute(f"funnel.{stage}_ids", json.dumps(ids))


def traced_node(name: str | None = None):
    """Wraps a LangGraph node function (state -> partial state dict) in a span."""

    def decorator(fn):
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(state):
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("graph.trace_id", state.get("trace_id", "") or "")
                span.set_attribute("graph.step", state.get("steps", 0))
                try:
                    result = fn(state)
                except GraphInterrupt:
                    # not a failure -- this is the approval-gate pausing for a human
                    span.set_attribute("graph.interrupted", True)
                    raise
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                if result.get("candidate_tools"):
                    span.set_attribute(
                        "funnel.final_candidate_ids",
                        json.dumps([c["tool_id"] for c in result["candidate_tools"]]),
                    )
                    span.set_attribute(
                        "funnel.final_candidate_scores",
                        json.dumps([c["score"] for c in result["candidate_tools"]]),
                    )
                if result.get("selected_tool"):
                    span.set_attribute("tool.selected", result["selected_tool"])
                if "validation_errors" in result:
                    span.set_attribute("validate.errors", json.dumps(result["validation_errors"]))
                if "retry_count" in result:
                    span.set_attribute("validate.retry_count", result["retry_count"])
                if result.get("plan"):
                    span.set_attribute("plan.turn_type", result["plan"])
                return result

        return wrapper

    return decorator
