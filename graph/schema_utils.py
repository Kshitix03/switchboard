"""Flatten a bundled ($ref/$defs/allOf) request schema into a fully inlined,
LLM-friendly shape for the fill prompt only. Validation always uses the
original schema unmodified (jsonschema resolves $ref/allOf natively, so
correctness doesn't depend on this flattening)."""

from __future__ import annotations

MAX_DEPTH = 14
MAX_ENUM_ITEMS = 15
MAX_DESCRIPTION_LEN = 120
MAX_OPTIONAL_PROPS = 10  # per object level, in addition to all required properties


def _trim(text: str | None) -> str | None:
    if not text:
        return None
    return text if len(text) <= MAX_DESCRIPTION_LEN else text[:MAX_DESCRIPTION_LEN] + "..."


def flatten_for_prompt(schema: dict, defs: dict | None = None, depth: int = 0, seen: frozenset = frozenset()) -> dict:
    defs = defs if defs is not None else schema.get("$defs", {})

    if depth > MAX_DEPTH:
        return {"type": "object"}

    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        if name in seen:
            return {"type": "object", "description": f"(recursive ref to {name}, not expanded)"}
        target = defs.get(name, {})
        return flatten_for_prompt(target, defs, depth + 1, seen | {name})

    if "allOf" in schema:
        merged_props: dict = {}
        merged_required: set[str] = set()
        for sub in schema["allOf"]:
            resolved = flatten_for_prompt(sub, defs, depth + 1, seen)
            merged_props.update(resolved.get("properties", {}))
            merged_required |= set(resolved.get("required", []))
        out = {"type": "object", "properties": merged_props}
        if merged_required:
            out["required"] = sorted(merged_required)
        desc = _trim(schema.get("description"))
        if desc:
            out["description"] = desc
        return out

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        all_props = schema.get("properties", {})
        required = schema.get("required", [])
        optional_keys = [k for k in all_props if k not in required]
        kept = set(required) | set(optional_keys[:MAX_OPTIONAL_PROPS])
        props = {
            k: flatten_for_prompt(v, defs, depth + 1, seen)
            for k, v in all_props.items() if k in kept
        }
        out = {"type": "object", "properties": props}
        if required:
            out["required"] = required
        desc = _trim(schema.get("description"))
        if desc:
            out["description"] = desc
        return out

    if schema_type == "array":
        return {"type": "array", "items": flatten_for_prompt(schema.get("items", {}), defs, depth + 1, seen)}

    out: dict = {"type": schema_type or "string"}
    if schema.get("enum"):
        out["enum"] = schema["enum"][:MAX_ENUM_ITEMS]
    if schema.get("format"):
        out["format"] = schema["format"]
    desc = _trim(schema.get("description"))
    if desc:
        out["description"] = desc
    return out
