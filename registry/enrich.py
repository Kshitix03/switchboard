"""Generate utterances + keyword synonyms per tool via Gemini 2.5 Flash.

Batches multiple tools into one call (RPD is tight: 20/day on this key) and
caches results to disk keyed by tool id, so reruns after the first cost zero
API calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from registry.models import ToolRecord

load_dotenv()

TOOLS_PATH = Path(__file__).parent / "data" / "tools.json"
CACHE_PATH = Path(__file__).parent / "data" / "enrichment_cache.json"
ENRICHED_OUT_PATH = Path(__file__).parent / "data" / "tools_enriched.json"

MODEL = "gemini-flash-lite-latest"
BATCH_SIZE = 8
# stay under 5 RPM even though batching already keeps us under 20 RPD
SECONDS_BETWEEN_CALLS = 13

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "utterances": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "utterances", "keywords"],
    },
}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _build_prompt(batch: list[ToolRecord]) -> str:
    tool_lines = []
    for t in batch:
        tool_lines.append(
            f"- id: {t.id}\n"
            f"  method/path: {t.method} {t.path}\n"
            f"  summary: {t.summary}\n"
            f"  description: {t.description}"
        )
    tools_block = "\n".join(tool_lines)
    return f"""You are helping build a retrieval index for an API tool registry.

For each tool below, generate:
1. "utterances": 3 to 5 short, natural phrasings a real user might type in a
   chat interface that should trigger this exact tool. Vary phrasing and
   vocabulary. Do not just restate the summary. Include realistic details
   (amounts, emails, names) where relevant.
2. "keywords": short list of synonyms and related terms a user might use
   instead of the tool's own field names (do not repeat obvious path tokens).

Pay special attention to disambiguating tools that sound similar (e.g. "create"
vs "send", "refund" vs "dispute") -- the utterances should reflect what makes
THIS tool distinct from its near neighbors.

Tools:
{tools_block}

Return a JSON array, one object per tool, matching the id field exactly.
"""


def enrich_batch(client: genai.Client, batch: list[ToolRecord]) -> dict[str, dict]:
    prompt = _build_prompt(batch)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    items = json.loads(response.text)
    return {item["id"]: item for item in items}


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Create switchboard/.env with GEMINI_API_KEY=...")

    tools = [ToolRecord(**t) for t in json.loads(TOOLS_PATH.read_text(encoding="utf-8"))]
    cache = _load_cache()

    pending = [t for t in tools if t.id not in cache]
    print(f"{len(tools)} tools total, {len(pending)} need enrichment, {len(tools) - len(pending)} cached")

    client = genai.Client(api_key=api_key)

    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        print(f"batch {i + 1}/{len(batches)} ({len(batch)} tools)...")
        result = enrich_batch(client, batch)
        for t in batch:
            if t.id in result:
                cache[t.id] = result[t.id]
            else:
                print(f"  WARNING: model did not return an entry for {t.id}")
        _save_cache(cache)  # save after every batch so partial progress survives
        if i < len(batches) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    enriched = []
    for t in tools:
        entry = cache.get(t.id, {"utterances": [], "keywords": []})
        t.utterances = entry.get("utterances", [])
        t.keywords = sorted(set(t.keywords) | set(entry.get("keywords", [])))
        enriched.append(t)

    ENRICHED_OUT_PATH.write_text(
        json.dumps([t.model_dump() for t in enriched], indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote: {ENRICHED_OUT_PATH}")


if __name__ == "__main__":
    main()
