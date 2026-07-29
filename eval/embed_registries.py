"""Embed all tools across the padded registry tiers, cached separately from
production (registry/data/embedding_cache.json) so benchmark runs never touch
the live index. Reuses the production cache directly for core tools (same
content hash), so no core tool gets re-embedded."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from registry.models import ToolRecord

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
CORE_EMBEDDING_CACHE = Path(__file__).parent.parent / "registry" / "data" / "embedding_cache.json"
PADDING_EMBEDDING_CACHE = DATA_DIR / "padding_embedding_cache.json"

EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 10


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    # the 500 tier is a superset of 250 and 108, so embedding it covers all tiers
    tools = [ToolRecord(**t) for t in json.loads((DATA_DIR / "registry_500.json").read_text(encoding="utf-8"))]

    core_cache = json.loads(CORE_EMBEDDING_CACHE.read_text(encoding="utf-8")) if CORE_EMBEDDING_CACHE.exists() else {}
    cache = json.loads(PADDING_EMBEDDING_CACHE.read_text(encoding="utf-8")) if PADDING_EMBEDDING_CACHE.exists() else {}

    to_embed: list[ToolRecord] = []
    reused = 0
    for t in tools:
        text = t.card_text()
        h = _hash(text)
        if t.id in cache and cache[t.id]["hash"] == h:
            continue
        if t.id in core_cache and core_cache[t.id]["hash"] == h:
            cache[t.id] = core_cache[t.id]
            reused += 1
            continue
        to_embed.append(t)

    print(f"{len(tools)} tools total, {len(to_embed)} need embedding, {reused} reused from core cache, "
          f"{len(tools) - len(to_embed) - reused} already in padding cache")

    client = genai.Client(api_key=api_key)
    for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
        batch = to_embed[i:i + EMBED_BATCH_SIZE]
        texts = [t.card_text() for t in batch]
        for attempt in range(8):
            try:
                result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < 7:
                    time.sleep(35)
                    continue
                raise
        for t, emb in zip(batch, result.embeddings):
            cache[t.id] = {"hash": _hash(t.card_text()), "vector": emb.values}
        PADDING_EMBEDDING_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        if i % (EMBED_BATCH_SIZE * 10) == 0:
            print(f"  ...{i + len(batch)}/{len(to_embed)}")
        time.sleep(1.5)

    PADDING_EMBEDDING_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"wrote: {PADDING_EMBEDDING_CACHE}")


if __name__ == "__main__":
    main()
