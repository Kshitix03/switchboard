"""Shared, lazily-built retrieval store: records, Qdrant client, BM25 index.

One load per process. router.retrieve() and (later) system_search both read
through this module so there is exactly one retrieval path, not two.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from registry.models import ToolRecord

DATA_DIR = Path(__file__).parent.parent / "registry" / "data"
TOOLS_PATH = DATA_DIR / "tools_enriched.json"
EMBEDDING_CACHE_PATH = DATA_DIR / "embedding_cache.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
DOCS_PATH = DATA_DIR / "docs.json"
DOCS_EMBEDDING_CACHE_PATH = DATA_DIR / "docs_embedding_cache.json"

COLLECTION_NAME = "tools"
DOCS_COLLECTION_NAME = "docs"
VECTOR_DIM = 3072

_records: list[ToolRecord] | None = None
_records_by_id: dict[str, ToolRecord] | None = None
_qdrant: QdrantClient | None = None
_bm25_data: dict | None = None
_docs: list[dict] | None = None
_docs_indexed: bool = False


def _load_records() -> list[ToolRecord]:
    global _records, _records_by_id
    if _records is None:
        raw = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
        _records = [ToolRecord(**r) for r in raw]
        _records_by_id = {r.id: r for r in _records}
    return _records


def records() -> list[ToolRecord]:
    return _load_records()


def records_by_id() -> dict[str, ToolRecord]:
    _load_records()
    return _records_by_id  # type: ignore[return-value]


def qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        recs = _load_records()
        cache = json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=cache[r.id]["vector"],
                payload={
                    "tool_id": r.id,
                    "service": r.service,
                    "domain": r.domain,
                    "operation": r.operation,
                    "risk": r.risk,
                    "source": r.source,
                },
            )
            for i, r in enumerate(recs)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        _qdrant = client
    return _qdrant


def bm25_data() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def docs() -> list[dict]:
    global _docs
    if _docs is None:
        _docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    return _docs


def ensure_docs_collection() -> QdrantClient:
    """Indexes the docs corpus into a 'docs' collection on the SAME Qdrant
    client instance used for tools -- one process, one client, two collections."""
    global _docs_indexed
    client = qdrant()
    if not _docs_indexed:
        cache = json.loads(DOCS_EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
        doc_list = docs()
        client.create_collection(
            collection_name=DOCS_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=i,
                vector=cache[d["id"]]["vector"],
                payload={"doc_id": d["id"], "title": d["title"], "text": d["text"], "source": d["source"]},
            )
            for i, d in enumerate(doc_list)
        ]
        client.upsert(collection_name=DOCS_COLLECTION_NAME, points=points)
        _docs_indexed = True
    return client


def reset() -> None:
    """For tests / reindex-in-process scenarios."""
    global _records, _records_by_id, _qdrant, _bm25_data, _docs, _docs_indexed
    _records = None
    _records_by_id = None
    _qdrant = None
    _bm25_data = None
    _docs = None
    _docs_indexed = False
