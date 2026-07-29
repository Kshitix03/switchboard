"""Graph nodes: plan, retrieve_tools, bind, fill, validate, approve_gate,
execute, observe, clarify, plus dry_run/rejected/abort/chat terminals.

Edge conditions (prose, per PRD Phase 4 instruction):

- plan -> chat: the turn has no PayPal connection at all (small talk). ->
  system_search if plan classified the turn as "meta" (asking about the
  system itself). -> retrieve_tools otherwise.
- retrieve_tools -> clarify: the top fused-and-reranked candidate's score is
  below a confidence floor. We never guess a tool past this point -- a wrong
  guess that reaches fill/execute is worse than one extra turn asking the
  user to rephrase.
- retrieve_tools -> bind: otherwise, proceed with the top candidates.
- validate -> fill: schema validation failed and retry_count is still under
  the cap (2). The validator's error message is fed back into the next fill
  prompt so the repair loop costs zero *additional* API calls in the sense
  that it reuses the same turn's budget, not a fresh user round-trip.
- validate -> clarify: validation failed and the retry cap is exhausted --
  escalate to the user rather than looping forever or guessing.
- validate -> approve_gate: validation passed.
- approve_gate -> dry_run: `--dry-run` mode short-circuits the entire
  approval/execute path and only prints the call that *would* be made. This
  check happens before any interrupt() call so a dry run never pauses
  waiting on a human.
- approve_gate -> execute: the tool is read-only/low-risk (auto-approved) or
  a human approved it via the LangGraph interrupt.
- approve_gate -> rejected: a human declined the interrupt.
- any stage -> abort: a hard step cap (15) is hit, guarding against any
  cycle that isn't making forward progress.
"""

from __future__ import annotations

import json

import jsonschema
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from graph.llm import MODEL, get_client
from graph.schema_utils import flatten_for_prompt
from graph.state import AgentState
from observability import tracing
from observability.otel import record_funnel, record_tokens, traced_node
from registry.models import SchemaStore
from routing import router, store
from tools import executor, rag_search, system_search

MAX_RETRIES = 2
MAX_STEPS = 15
LOW_SCORE_THRESHOLD = 0.0  # cross-encoder logit; below this we don't trust the top hit

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "turn_type": {"type": "string", "enum": ["chat", "retrieve_and_act", "meta"]},
        "operation_filter": {"type": "string", "enum": ["read", "write"]},
    },
    "required": ["turn_type"],
}

TOOL_SELECT_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "tool_id": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["tool_id"],
}


def _last_human_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "")
    return ""


def _step_cap_exceeded(state: AgentState) -> bool:
    return state.get("steps", 0) >= MAX_STEPS


# ---- nodes -----------------------------------------------------------------

@traced_node()
def plan_node(state: AgentState) -> dict:
    client = get_client()
    query = _last_human_text(state["messages"])
    prompt = f"""Classify this user turn for a tool-routing agent over PayPal APIs.

- "chat": small talk or a question with NO connection to PayPal at all
  (greetings, unrelated general knowledge). Only use this when a PayPal tool
  or PayPal documentation genuinely could not help.
- "retrieve_and_act": the user wants to perform an action, look up specific
  data, OR is asking a conceptual/"how does X work" question about PayPal's
  own products (refunds, invoicing, disputes, payouts, etc.) that PayPal's own
  documentation should answer -- even if no API mutation is needed. When in
  doubt for any PayPal-related question, prefer this over "chat" so the
  answer is grounded in PayPal's actual docs/tools rather than general
  training data.
- "meta": the user is asking about the system itself (what tools exist, status
  of a prior request).

If you're confident the turn is read-only (a lookup, not a mutation), set
operation_filter to "read". If clearly a write/mutation, set it to "write".
Omit operation_filter if unsure.

User message: {query}
"""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=PLAN_SCHEMA),
    )
    record_tokens(resp.usage_metadata)
    data = json.loads(resp.text)
    filters = {"operation": data["operation_filter"]} if "operation_filter" in data else None
    return {
        "plan": data.get("turn_type", "retrieve_and_act"),
        "filters": filters,
        "steps": state.get("steps", 0) + 1,
    }


@traced_node()
def chat_node(state: AgentState) -> dict:
    client = get_client()
    query = _last_human_text(state["messages"])
    resp = client.models.generate_content(model=MODEL, contents=query)
    record_tokens(resp.usage_metadata)
    return {"messages": [AIMessage(content=resp.text)], "steps": state.get("steps", 0) + 1}


@traced_node()
def system_search_node(state: AgentState) -> dict:
    query = _last_human_text(state["messages"])
    answer = system_search.system_search(query, trace_id=state.get("trace_id"))
    return {"messages": [AIMessage(content=answer)], "steps": state.get("steps", 0) + 1}


@traced_node()
def retrieve_tools_node(state: AgentState) -> dict:
    query = _last_human_text(state["messages"])
    result = router.retrieve(query, filters=state.get("filters"))
    record_funnel(result.funnel)
    candidates = [{"tool_id": tid, "score": result.scores[tid]} for tid in result.scores]
    return {"candidate_tools": candidates, "steps": state.get("steps", 0) + 1}


@traced_node()
def bind_node(state: AgentState) -> dict:
    schema_store = SchemaStore()
    bound = {c["tool_id"]: schema_store.get(c["tool_id"]) for c in state["candidate_tools"]}
    artifacts = dict(state.get("artifacts", {}))
    artifacts["bound_schemas"] = bound
    return {"artifacts": artifacts, "steps": state.get("steps", 0) + 1}


def _select_tool(client, query: str, state: AgentState) -> str:
    """Step 1: pick exactly one tool id from the candidates (cheap, short prompt)."""
    records_by_id = store.records_by_id()
    candidate_blocks = [
        f"- {c['tool_id']}: {records_by_id[c['tool_id']].summary}"
        for c in state["candidate_tools"]
    ]
    schema = dict(TOOL_SELECT_SCHEMA_TEMPLATE)
    schema["properties"] = dict(schema["properties"])
    schema["properties"]["tool_id"] = {
        "type": "string",
        "enum": [c["tool_id"] for c in state["candidate_tools"]],
    }
    prompt = f"""Pick exactly one tool id that best satisfies the user's request.

User request: {query}

Candidates:
{chr(10).join(candidate_blocks)}
"""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema),
    )
    record_tokens(resp.usage_metadata)
    return json.loads(resp.text)["tool_id"]


def _fill_args(client, query: str, tool_id: str, flat_schema: dict, validation_errors: list[str]) -> dict:
    """Step 2: fill args using the tool's OWN flattened schema as the Gemini
    response_schema, so structured decoding is pressured by the real required
    fields instead of a generic empty `{"type": "object"}`."""
    error_block = ""
    if validation_errors:
        error_block = (
            "\n\nYour previous attempt failed validation with these errors -- fix them:\n"
            + "\n".join(validation_errors)
        )
    prompt = f"""Fill the arguments for calling {tool_id} to satisfy this user request.

User request: {query}
{error_block}
"""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=flat_schema),
    )
    record_tokens(resp.usage_metadata)
    return json.loads(resp.text)


@traced_node()
def fill_node(state: AgentState) -> dict:
    client = get_client()
    query = _last_human_text(state["messages"])
    bound = state["artifacts"]["bound_schemas"]

    # On a repair retry, keep the same tool -- the repair loop fixes args, it
    # doesn't re-litigate tool choice.
    if state.get("selected_tool") and state.get("validation_errors"):
        tool_id = state["selected_tool"]
    else:
        tool_id = _select_tool(client, query, state)

    flat_schema = flatten_for_prompt(bound[tool_id]["request_body"])
    args = _fill_args(client, query, tool_id, flat_schema, state.get("validation_errors") or [])

    return {
        "selected_tool": tool_id,
        "filled_args": args,
        "steps": state.get("steps", 0) + 1,
    }


@traced_node()
def validate_node(state: AgentState) -> dict:
    tool_id = state["selected_tool"]
    bound = state["artifacts"]["bound_schemas"]
    schema_entry = bound.get(tool_id)

    if schema_entry is None:
        return {
            "validation_errors": [f"Unknown tool_id '{tool_id}' -- not among the retrieved candidates."],
            "retry_count": state.get("retry_count", 0) + 1,
            "steps": state.get("steps", 0) + 1,
        }

    try:
        jsonschema.validate(instance=state.get("filled_args") or {}, schema=schema_entry["request_body"])
        return {"validation_errors": [], "steps": state.get("steps", 0) + 1}
    except jsonschema.exceptions.ValidationError as e:
        return {
            "validation_errors": [e.message],
            "retry_count": state.get("retry_count", 0) + 1,
            "steps": state.get("steps", 0) + 1,
        }


@traced_node()
def approve_gate_node(state: AgentState) -> dict:
    tool_id = state["selected_tool"]
    record = store.records_by_id()[tool_id]

    if state.get("dry_run"):
        return {"pending_approval": None, "steps": state.get("steps", 0) + 1}

    if record.requires_approval:
        decision = interrupt({
            "type": "approval_request",
            "tool_id": tool_id,
            "args": state.get("filled_args"),
            "risk": record.risk,
            "operation": record.operation,
        })
        approved = decision.get("approved", False) if isinstance(decision, dict) else bool(decision)
        return {
            "pending_approval": {"tool_id": tool_id, "approved": approved},
            "steps": state.get("steps", 0) + 1,
        }

    return {
        "pending_approval": {"tool_id": tool_id, "approved": True},
        "steps": state.get("steps", 0) + 1,
    }


@traced_node()
def execute_node(state: AgentState) -> dict:
    tool_id = state["selected_tool"]
    args = state.get("filled_args") or {}
    schema_entry = state["artifacts"]["bound_schemas"][tool_id]

    if tool_id == "rag_search":
        hits = rag_search.search(get_client(), args.get("query", ""))
        result = {"status": "executed", "tool_id": tool_id, "idempotency_key": None, "response": {"results": hits}}
    else:
        result = executor.execute(tool_id, args, schema_entry, trace_id=state["trace_id"], mode="mock")

    tracing.log_execution(state["trace_id"], tool_id, args, result)

    artifacts = dict(state.get("artifacts", {}))
    artifacts["last_execution"] = result
    return {"artifacts": artifacts, "steps": state.get("steps", 0) + 1}


@traced_node()
def observe_node(state: AgentState) -> dict:
    client = get_client()
    result = state["artifacts"].get("last_execution", {})
    tool_id = state.get("selected_tool")
    execution_note = "a real docs search" if result.get("status") == "executed" else "a mock execution (no real API call was made)"
    prompt = (
        f"The tool {tool_id} was called ({execution_note}) with args "
        f"{json.dumps(state.get('filled_args'))}. Result: {json.dumps(result.get('response', {}))}. "
        "Write a short, natural 1-2 sentence confirmation to the user."
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    record_tokens(resp.usage_metadata)
    return {"messages": [AIMessage(content=resp.text)], "steps": state.get("steps", 0) + 1}


@traced_node()
def clarify_node(state: AgentState) -> dict:
    candidates = state.get("candidate_tools", [])
    if state.get("validation_errors") and state.get("retry_count", 0) >= MAX_RETRIES:
        msg = (
            f"I couldn't fill in valid arguments for this request after {MAX_RETRIES} attempts "
            f"(last error: {state['validation_errors'][-1]}). Could you clarify the details?"
        )
    elif candidates:
        top = candidates[0]
        msg = (
            f"I'm not confident which action you want (best guess: {top['tool_id']}, "
            f"score {top['score']:.2f}). Could you rephrase or give more detail?"
        )
    else:
        msg = "I couldn't find a matching action for that request. Could you rephrase?"
    return {"messages": [AIMessage(content=msg)], "steps": state.get("steps", 0) + 1}


@traced_node()
def dry_run_node(state: AgentState) -> dict:
    tool_id = state["selected_tool"]
    schema_entry = state["artifacts"]["bound_schemas"][tool_id]
    call_desc = (
        f"[DRY RUN] Would call {schema_entry['method']} {schema_entry['path']} "
        f"with args:\n{json.dumps(state.get('filled_args'), indent=2)}"
    )
    return {"messages": [AIMessage(content=call_desc)]}


@traced_node()
def rejected_node(state: AgentState) -> dict:
    tool_id = state.get("selected_tool")
    return {"messages": [AIMessage(content=f"Okay, I will not execute {tool_id}.")]}


@traced_node()
def abort_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(
        content="This turn hit the internal step cap, so I'm stopping to avoid a runaway loop. "
                "Please try rephrasing your request."
    )]}


# ---- routing (conditional edges) -------------------------------------------

def route_after_plan(state: AgentState) -> str:
    if _step_cap_exceeded(state):
        return "abort"
    plan = state.get("plan")
    if plan == "chat":
        return "chat"
    if plan == "meta":
        return "system_search"
    return "retrieve_tools"


def route_after_retrieve(state: AgentState) -> str:
    if _step_cap_exceeded(state):
        return "abort"
    candidates = state.get("candidate_tools", [])
    if not candidates or candidates[0]["score"] < LOW_SCORE_THRESHOLD:
        return "clarify"
    return "bind"


def route_after_validate(state: AgentState) -> str:
    if _step_cap_exceeded(state):
        return "abort"
    if not state.get("validation_errors"):
        return "approve_gate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "clarify"
    return "fill"


def route_after_approve(state: AgentState) -> str:
    if _step_cap_exceeded(state):
        return "abort"
    if state.get("dry_run"):
        return "dry_run"
    pending = state.get("pending_approval") or {}
    return "execute" if pending.get("approved") else "rejected"
