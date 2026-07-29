"""Build card text, embed it, upsert to an in-memory Qdrant collection, and
build a persisted BM25 index over the same card text (PRD section 5.1, 10).

Embeddings are cached to disk by tool id + a hash of the card text, separate
from the (ephemeral) Qdrant collection itself -- so re-running this script
after the first time costs zero embedding calls even though Qdrant's
in-memory index rebuilds from the cache every process start.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from registry.models import ToolRecord

load_dotenv()

TOOLS_PATH = Path(__file__).parent / "data" / "tools_enriched.json"
EMBEDDING_CACHE_PATH = Path(__file__).parent / "data" / "embedding_cache.json"
BM25_INDEX_PATH = Path(__file__).parent / "data" / "bm25_index.pkl"

EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 10
COLLECTION_NAME = "tools"
VECTOR_DIM = 3072  # gemini-embedding-001 default output dim


def _card_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_embedding_cache() -> dict:
    if EMBEDDING_CACHE_PATH.exists():
        return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_embedding_cache(cache: dict) -> None:
    EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def embed_tools(client: genai.Client, records: list[ToolRecord]) -> dict[str, list[float]]:
    """Return tool_id -> embedding vector, using the disk cache where the
    card text hasn't changed since it was last embedded."""
    cache = _load_embedding_cache()
    vectors: dict[str, list[float]] = {}
    to_embed: list[ToolRecord] = []
    to_embed_texts: list[str] = []

    for r in records:
        text = r.card_text()
        h = _card_hash(text)
        entry = cache.get(r.id)
        if entry and entry["hash"] == h:
            vectors[r.id] = entry["vector"]
        else:
            to_embed.append(r)
            to_embed_texts.append(text)

    print(f"{len(records)} tools, {len(to_embed)} need (re)embedding, {len(records) - len(to_embed)} cached")

    for i in range(0, len(to_embed), EMBED_BATCH_SIZE):
        batch = to_embed[i:i + EMBED_BATCH_SIZE]
        batch_texts = to_embed_texts[i:i + EMBED_BATCH_SIZE]
        result = client.models.embed_content(model=EMBED_MODEL, contents=batch_texts)
        for r, emb in zip(batch, result.embeddings):
            vectors[r.id] = emb.values
            cache[r.id] = {"hash": _card_hash(r.card_text()), "vector": emb.values}

    _save_embedding_cache(cache)
    return vectors


def build_qdrant(records: list[ToolRecord], vectors: dict[str, list[float]]) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[r.id],
            payload={
                "tool_id": r.id,
                "service": r.service,
                "domain": r.domain,
                "operation": r.operation,
                "risk": r.risk,
                "source": r.source,
            },
        )
        for i, r in enumerate(records)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def build_bm25(records: list[ToolRecord]) -> None:
    tokenized = [r.card_text().lower().split() for r in records]
    bm25 = BM25Okapi(tokenized)
    ids = [r.id for r in records]
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"wrote: {BM25_INDEX_PATH}")


def main() -> None:
    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Create switchboard/.env with GEMINI_API_KEY=...")

    records = [ToolRecord(**t) for t in json.loads(TOOLS_PATH.read_text(encoding="utf-8"))]
    client = genai.Client(api_key=api_key)

    vectors = embed_tools(client, records)
    qdrant = build_qdrant(records, vectors)
    build_bm25(records)

    count = qdrant.count(collection_name=COLLECTION_NAME).count
    print(f"\nQdrant collection '{COLLECTION_NAME}': {count} points")


if __name__ == "__main__":
    main()
