"""Embed the docs corpus for rag_search, cached to disk by content hash
(same pattern as registry/index.py for tools)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

DOCS_PATH = Path(__file__).parent / "data" / "docs.json"
DOCS_EMBEDDING_CACHE_PATH = Path(__file__).parent / "data" / "docs_embedding_cache.json"

EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 10


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    cache = json.loads(DOCS_EMBEDDING_CACHE_PATH.read_text(encoding="utf-8")) if DOCS_EMBEDDING_CACHE_PATH.exists() else {}

    to_embed = [d for d in docs if not (d["id"] in cache and cache[d["id"]]["hash"] == _hash(d["text"]))]
    print(f"{len(docs)} docs, {len(to_embed)} need (re)embedding, {len(docs) - len(to_embed)} cached")

    client = genai.Client(api_key=api_key)
    for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
        batch = to_embed[i:i + EMBED_BATCH_SIZE]
        texts = [f"{d['title']}. {d['text']}" for d in batch]
        result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
        for d, emb in zip(batch, result.embeddings):
            cache[d["id"]] = {"hash": _hash(d["text"]), "vector": emb.values}

    DOCS_EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    print(f"wrote: {DOCS_EMBEDDING_CACHE_PATH}")


if __name__ == "__main__":
    main()
