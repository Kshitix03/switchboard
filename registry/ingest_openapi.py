"""Parse PayPal (or any) OpenAPI 3.x specs into ToolRecords + a schema store.

One operation (method + path) = one tool. Full request/response schemas are
bundled with local $refs resolved into a self-contained "$defs" block and
written to the schema store keyed by schema_ref -- they never enter the
ToolRecord itself, so the vector index only ever sees the short card text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from registry.models import SchemaStore, ToolRecord

RAW_SPECS_DIR = Path(__file__).parent / "data" / "raw_specs"
TOOLS_OUT_PATH = Path(__file__).parent / "data" / "tools.json"

# Domain + risk are inferred from the spec filename since PayPal's own
# tag/path conventions aren't consistent enough across specs to derive
# a clean domain automatically.
SPEC_META = {
    "invoicing_v2": {"domain": "invoicing", "risk_write": "medium"},
    "checkout_orders_v2": {"domain": "payments", "risk_write": "high"},
    "payments_payment_v2": {"domain": "payments", "risk_write": "high"},
    "customer_disputes_v1": {"domain": "disputes", "risk_write": "high"},
    "reporting_transactions_v1": {"domain": "reporting", "risk_write": "medium"},
    "payments_payouts_batch_v1": {"domain": "payments", "risk_write": "high"},
    "catalogs_products_v1": {"domain": "catalog", "risk_write": "low"},
    "vault_payment_tokens_v3": {"domain": "vault", "risk_write": "high"},
    "billing_subscriptions_v1": {"domain": "subscriptions", "risk_write": "high"},
    "notifications_webhooks_v1": {"domain": "webhooks", "risk_write": "medium"},
    "shipping_shipment_tracking_v1": {"domain": "tracking", "risk_write": "medium"},
}

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _resolve_refs(node: Any, components: dict, defs: dict, seen: set[str]) -> Any:
    """Recursively inline local $ref pointers into a shared $defs block."""
    if isinstance(node, dict):
        if "$ref" in node and node["$ref"].startswith("#/components/"):
            ref_path = node["$ref"].removeprefix("#/components/")
            def_name = ref_path.replace("/", ".")
            if def_name not in seen:
                seen.add(def_name)
                target = components
                for part in ref_path.split("/"):
                    target = target[part]
                defs[def_name] = _resolve_refs(target, components, defs, seen)
            return {"$ref": f"#/$defs/{def_name}"}
        return {k: _resolve_refs(v, components, defs, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, components, defs, seen) for item in node]
    return node


def _bundle_schema(node: dict, components: dict) -> dict:
    defs: dict = {}
    resolved = _resolve_refs(node, components, defs, set())
    if defs:
        resolved["$defs"] = defs
    return resolved


def _extract_request_schema(op: dict, components: dict) -> dict:
    body = op.get("requestBody", {})
    content = body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema")
    if not schema:
        return {"type": "object", "properties": {}}
    return _bundle_schema(schema, components)


def _extract_response_schema(op: dict, components: dict) -> dict:
    responses = op.get("responses", {})
    for code in ("200", "201", "202", "204"):
        resp = responses.get(code)
        if not resp:
            continue
        schema = resp.get("content", {}).get("application/json", {}).get("schema")
        if schema:
            return _bundle_schema(schema, components)
    return {"type": "object", "properties": {}}


def _keywords_from_path(path: str, op: dict) -> list[str]:
    tokens = [seg.strip("{}") for seg in path.split("/") if seg]
    tokens += op.get("tags", [])
    return sorted(set(t.lower() for t in tokens if t))


def ingest_spec(spec_path: Path, schema_store: SchemaStore) -> list[ToolRecord]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    components = spec.get("components", {})
    meta = SPEC_META.get(spec_path.stem, {"domain": "unknown", "risk_write": "medium"})
    domain = meta["domain"]
    records: list[ToolRecord] = []

    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in HTTP_METHODS:
                continue
            operation_id = op.get("operationId")
            if not operation_id:
                continue

            operation: Literal["read", "write"] = "read" if method.lower() == "get" else "write"
            risk: Literal["low", "medium", "high"] = "low" if operation == "read" else meta["risk_write"]

            tool_id = f"paypal.{domain}.{operation_id}"
            schema_ref = tool_id

            request_schema = _extract_request_schema(op, components)
            response_schema = _extract_response_schema(op, components)
            schema_store.put(schema_ref, {
                "method": method.upper(),
                "path": path,
                "parameters": op.get("parameters", []),
                "request_body": request_schema,
                "response_body": response_schema,
            })

            summary = op.get("summary", operation_id)
            description = op.get("description", summary) or summary
            # descriptions in these specs can be long HTML; keep first ~400 chars as plain-ish text
            description = " ".join(description.split())[:400]

            records.append(ToolRecord(
                id=tool_id,
                service="paypal",
                domain=domain,
                name=summary,
                summary=summary,
                description=description,
                keywords=_keywords_from_path(path, op),
                utterances=[],  # filled in by enrich.py
                operation=operation,
                risk=risk,
                method=method.upper(),
                path=path,
                schema_ref=schema_ref,
                requires_approval=(operation == "write" or risk == "high"),
                source="paypal",
            ))

    return records


def main() -> None:
    schema_store = SchemaStore()
    all_records: list[ToolRecord] = []

    for spec_name in SPEC_META:
        spec_path = RAW_SPECS_DIR / f"{spec_name}.json"
        if not spec_path.exists():
            print(f"skip (missing): {spec_path}")
            continue
        records = ingest_spec(spec_path, schema_store)
        print(f"{spec_name}: {len(records)} tools")
        all_records.extend(records)

    schema_store.save()
    TOOLS_OUT_PATH.write_text(
        json.dumps([r.model_dump() for r in all_records], indent=2),
        encoding="utf-8",
    )
    print(f"\ntotal tools: {len(all_records)}")
    print(f"wrote: {TOOLS_OUT_PATH}")
    print(f"wrote: {schema_store.path}")


if __name__ == "__main__":
    main()
