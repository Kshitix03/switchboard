"""Append internal (non-PayPal-API) tools -- currently just rag_search -- to
the registry. Idempotent: re-running replaces the existing entry rather than
duplicating it. Run after ingest_openapi.py/enrich.py, before index.py."""

from __future__ import annotations

import json
from pathlib import Path

from registry.models import SchemaStore, ToolRecord

TOOLS_ENRICHED_PATH = Path(__file__).parent / "data" / "tools_enriched.json"

RAG_SEARCH_TOOL = ToolRecord(
    id="rag_search",
    service="internal",
    domain="knowledge",
    name="Search PayPal developer docs",
    summary="Searches PayPal API conceptual documentation (not the API reference itself).",
    description=(
        "Use this when the user asks a conceptual 'how does X work' question -- e.g. how "
        "invoicing templates work, what dispute statuses mean, how payout eligibility is "
        "determined -- rather than asking to perform an action. Returns relevant documentation "
        "excerpts, not an API call result."
    ),
    keywords=["docs", "documentation", "how does", "explain", "help", "guide", "what is", "concept"],
    utterances=[
        "How do refunds work on PayPal?",
        "What do the different invoice statuses mean?",
        "Explain how payout eligibility works",
        "What is idempotency in the payments API?",
        "How does the disputes process work?",
    ],
    operation="read",
    risk="low",
    method="INTERNAL",
    path="rag_search",
    schema_ref="rag_search",
    requires_approval=False,
    source="internal",
)

RAG_SEARCH_SCHEMA = {
    "method": "INTERNAL",
    "path": "rag_search",
    "parameters": [],
    "request_body": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The user's question to search docs for."}},
        "required": ["query"],
    },
    "response_body": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "text": {"type": "string"}},
                },
            },
        },
    },
}


def main() -> None:
    tools = json.loads(TOOLS_ENRICHED_PATH.read_text(encoding="utf-8"))
    tools = [t for t in tools if t["id"] != RAG_SEARCH_TOOL.id]
    tools.append(RAG_SEARCH_TOOL.model_dump())
    TOOLS_ENRICHED_PATH.write_text(json.dumps(tools, indent=2), encoding="utf-8")

    schema_store = SchemaStore()
    schema_store.put(RAG_SEARCH_TOOL.schema_ref, RAG_SEARCH_SCHEMA)
    schema_store.save()

    print(f"rag_search tool registered. total tools: {len(tools)}")


if __name__ == "__main__":
    main()
