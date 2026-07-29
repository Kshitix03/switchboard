"""Mock and live execution (PRD section 7, 10 Phase 4).

Mock mode (default) returns a schema-valid fake response derived from the
tool's response schema -- no network call. Live mode is out of scope for this
build (see DECISIONS.md / PRD cut list item 1): it raises rather than
silently no-op'ing, so a caller can't mistake a stub for a real call.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Literal, TypedDict


class ExecutionResult(TypedDict):
    status: Literal["mocked", "error"]
    tool_id: str
    idempotency_key: str | None
    response: dict


def make_idempotency_key(trace_id: str, tool_id: str, args: dict) -> str:
    """Deterministic per logical-call key: same trace + tool + args always
    yields the same key, so a retried write doesn't get a fresh key and
    therefore can't double-send under a real gateway that dedupes on it."""
    payload = f"{trace_id}:{tool_id}:{sorted(args.items())}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _fake_string(fmt: str | None, prop_name: str | None) -> str:
    if fmt == "date-time":
        return "2026-07-28T12:00:00Z"
    if fmt == "date":
        return "2026-07-28"
    if fmt == "email" or (prop_name and "email" in prop_name.lower()):
        return "user@example.com"
    if fmt in ("uri", "url"):
        return "https://api.paypal.com/v2/mock/resource"
    return f"mock_{uuid.uuid4().hex[:8]}"


def _generate_mock_value(schema: dict, defs: dict, depth: int = 0, prop_name: str | None = None) -> Any:
    if depth > 6:
        return None
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        target = defs.get(name, {})
        return _generate_mock_value(target, defs, depth + 1, prop_name)

    if schema.get("enum"):
        return schema["enum"][0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0] if schema_type else None)

    if schema_type == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        out: dict[str, Any] = {}
        for key in required:
            if key in props:
                out[key] = _generate_mock_value(props[key], defs, depth + 1, prop_name=key)
        for key in list(props.keys())[:3]:
            if key not in out:
                out[key] = _generate_mock_value(props[key], defs, depth + 1, prop_name=key)
        return out

    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_generate_mock_value(item_schema, defs, depth + 1, prop_name)]

    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    if schema_type == "string":
        return _fake_string(schema.get("format"), prop_name)
    return None


def execute_mock(tool_id: str, args: dict, schema_entry: dict, trace_id: str) -> ExecutionResult:
    method = schema_entry["method"]
    idempotency_key = make_idempotency_key(trace_id, tool_id, args) if method != "GET" else None

    response_schema = schema_entry.get("response_body", {"type": "object", "properties": {}})
    defs = response_schema.get("$defs", {})
    mock_response = _generate_mock_value(response_schema, defs)

    return ExecutionResult(
        status="mocked",
        tool_id=tool_id,
        idempotency_key=idempotency_key,
        response=mock_response if isinstance(mock_response, dict) else {"result": mock_response},
    )


def execute_live(tool_id: str, args: dict, schema_entry: dict) -> ExecutionResult:
    raise NotImplementedError(
        "Live execution is out of scope for this build (PRD cut list item 1). "
        "Switchboard runs fully mocked; wire an httpx call here against PayPal "
        "sandbox credentials if live execution is ever needed."
    )


def execute(tool_id: str, args: dict, schema_entry: dict, trace_id: str, mode: str = "mock") -> ExecutionResult:
    if mode == "mock":
        return execute_mock(tool_id, args, schema_entry, trace_id)
    return execute_live(tool_id, args, schema_entry)
