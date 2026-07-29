"""AgentState per PRD section 5.3."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


class CandidateTool(TypedDict):
    tool_id: str
    score: float


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    plan: Literal["chat", "retrieve_and_act", "meta"] | None
    filters: dict[str, Any] | None
    candidate_tools: list[CandidateTool]
    selected_tool: str | None
    filled_args: dict[str, Any] | None
    validation_errors: list[str]
    retry_count: int

    entities: dict[str, Any]
    artifacts: dict[str, Any]

    pending_approval: dict[str, Any] | None
    trace_id: str
    steps: int
    dry_run: bool
