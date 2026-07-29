"""Isolated (benchmark-only) Qdrant collection + BM25 index for a given
registry tier. Never touches the production 'tools' collection or its
embedding cache -- benchmark runs are fully separate from the live agent."""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from registry.models import ToolRecord

DATA_DIR = Path(__file__).parent / "data"
PADDING_EMBEDDING_CACHE = DATA_DIR / "padding_embedding_cache.json"
VECTOR_DIM = 3072
COLLECTION_NAME = "bench_tools"


class BenchStore:
    def __init__(self, size: int):
        self.size = size
        raw = json.loads((DATA_DIR / f"registry_{size}.json").read_text(encoding="utf-8"))
        self.records: list[ToolRecord] = [ToolRecord(**t) for t in raw]
        self.records_by_id: dict[str, ToolRecord] = {r.id: r for r in self.records}
        self._qdrant: QdrantClient | None = None
        self._bm25 = None
        self._bm25_ids: list[str] | None = None
        self._embeddings: dict[str, list[float]] | None = None

    def embeddings(self) -> dict[str, list[float]]:
        if self._embeddings is None:
            cache = json.loads(PADDING_EMBEDDING_CACHE.read_text(encoding="utf-8"))
            self._embeddings = {r.id: cache[r.id]["vector"] for r in self.records}
        return self._embeddings

    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            embeddings = self.embeddings()
            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            points = [
                PointStruct(id=i, vector=embeddings[r.id], payload={"tool_id": r.id})
                for i, r in enumerate(self.records)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            self._qdrant = client
        return self._qdrant

    def bm25(self) -> tuple[BM25Okapi, list[str]]:
        if self._bm25 is None:
            tokenized = [r.card_text().lower().split() for r in self.records]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_ids = [r.id for r in self.records]
        return self._bm25, self._bm25_ids
