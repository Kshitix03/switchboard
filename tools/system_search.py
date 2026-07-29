"""system_search: renders router.retrieve() results as prose instead of
binding them as callables, and answers "status of my last request" from the
trace store (PRD section 4.2).

Same retrieval path as retrieve_tools_node -- routing.router.retrieve -- just
a different renderer. Not a competing registry entry: it's invoked directly
when plan_node classifies a turn as "meta".
"""

from __future__ import annotations

from observability import tracing
from routing import router

STATUS_KEYWORDS = ("status", "last request", "did it work", "what happened", "my request")


def _is_status_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in STATUS_KEYWORDS)


def render_status(trace_id: str | None = None) -> str:
    entries = tracing.last_n(5)
    if trace_id:
        entries = [e for e in entries if e["trace_id"] == trace_id] or entries
    if not entries:
        return "I don't have any recorded requests yet in this session."

    last = entries[-1]
    return (
        f"Your last request called `{last['tool_id']}` -- status: {last['status']}"
        + (f" (idempotency key: {last['idempotency_key']})" if last.get("idempotency_key") else "")
        + f". Args: {last['args']}"
    )


def render_tool_search(query: str) -> str:
    result = router.retrieve(query, top_n=6)
    if not result.records:
        return "I couldn't find any tools matching that."

    lines = ["Here's what I found in the tool registry:"]
    for record in result.records:
        lines.append(f"- **{record.id}** ({record.domain}, {record.operation}): {record.summary}")
    return "\n".join(lines)


def system_search(query: str, trace_id: str | None = None) -> str:
    if _is_status_query(query):
        return render_status(trace_id)
    return render_tool_search(query)
