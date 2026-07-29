"""Pad the registry to 250 and 500 tools using GitHub's public REST API
OpenAPI spec (PRD section 8.2/10 Phase 7).

Padding is real tools from a real spec, tagged source="padding:github" so the
report can separate real (PayPal) vs padding tools. Deliberately NO LLM
enrichment for padding tools -- utterances stay empty, keywords come only
from path tokens/tags. Production tools get hand-tuned utterances because
they're the surface we actually built; incidental padding doesn't need the
same investment, and this asymmetry is disclosed rather than hidden.

Registry size tiers are 108/250/500, not the PRD's suggested 50/150/500,
because our real (unpadded) registry already grew to 108 tools in Phase 3 to
support golden-set near-miss pairs -- see DECISIONS.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from registry.ingest_openapi import (
    HTTP_METHODS,
    _extract_request_schema,
    _extract_response_schema,
    _keywords_from_path,
)
from registry.models import SchemaStore, ToolRecord

GITHUB_SPEC_PATH = Path(__file__).parent.parent / "registry" / "data" / "padding_specs" / "github_api.json"
CORE_TOOLS_PATH = Path(__file__).parent.parent / "registry" / "data" / "tools_enriched.json"
BENCH_DATA_DIR = Path(__file__).parent / "data"

SEED = 42
TIERS = [108, 250, 500]


def ingest_github_tools(schema_store: SchemaStore) -> list[ToolRecord]:
    spec = json.loads(GITHUB_SPEC_PATH.read_text(encoding="utf-8"))
    components = spec.get("components", {})
    records: list[ToolRecord] = []

    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in HTTP_METHODS:
                continue
            operation_id = op.get("operationId")
            if not operation_id:
                continue

            domain = operation_id.split("/")[0] if "/" in operation_id else "general"
            tool_id = f"github.{operation_id.replace('/', '.')}"
            operation = "read" if method.lower() == "get" else "write"
            risk = "low" if operation == "read" else "medium"

            request_schema = _extract_request_schema(op, components)
            response_schema = _extract_response_schema(op, components)
            schema_store.put(tool_id, {
                "method": method.upper(),
                "path": path,
                "parameters": op.get("parameters", []),
                "request_body": request_schema,
                "response_body": response_schema,
            })

            summary = op.get("summary", operation_id)
            description = op.get("description", summary) or summary
            description = " ".join(description.split())[:400]

            records.append(ToolRecord(
                id=tool_id,
                service="github",
                domain=domain,
                name=summary,
                summary=summary,
                description=description,
                keywords=_keywords_from_path(path, op),
                utterances=[],
                operation=operation,
                risk=risk,
                method=method.upper(),
                path=path,
                schema_ref=tool_id,
                requires_approval=(operation == "write"),
                source="padding:github",
            ))

    return records


def build_registry_tiers() -> None:
    core = json.loads(CORE_TOOLS_PATH.read_text(encoding="utf-8"))
    core_ids = {t["id"] for t in core}
    print(f"core (real, unpadded) registry: {len(core)} tools")

    schema_store = SchemaStore()
    github_records = ingest_github_tools(schema_store)
    schema_store.save()
    print(f"github padding pool: {len(github_records)} tools ingested")

    github_dicts = [r.model_dump() for r in github_records if r.id not in core_ids]
    rng = random.Random(SEED)
    rng.shuffle(github_dicts)

    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for size in TIERS:
        padding_needed = max(0, size - len(core))
        tools = core + github_dicts[:padding_needed]
        out_path = BENCH_DATA_DIR / f"registry_{size}.json"
        out_path.write_text(json.dumps(tools, indent=2), encoding="utf-8")
        print(f"tier {size}: {len(tools)} tools ({len(core)} real + {min(padding_needed, len(github_dicts))} padding) -> {out_path}")


if __name__ == "__main__":
    build_registry_tiers()
