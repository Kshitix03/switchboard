"""Dense retrieval: embed the query, search Qdrant with optional payload filters."""

from __future__ import annotations

from google import genai
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from routing import store

EMBED_MODEL = "gemini-embedding-001"


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions)


def embed_query(client: genai.Client, query: str) -> list[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    return result.embeddings[0].values


def search(
    client: genai.Client,
    query: str,
    top_k: int = 30,
    filters: dict | None = None,
) -> list[tuple[str, float]]:
    """Return [(tool_id, score), ...] ranked by cosine similarity."""
    qvec = embed_query(client, query)
    qdrant_client = store.qdrant()
    qdrant_filter = _build_filter(filters)
    hits = qdrant_client.query_points(
        store.COLLECTION_NAME,
        query=qvec,
        limit=top_k,
        query_filter=qdrant_filter,
    ).points
    return [(h.payload["tool_id"], h.score) for h in hits]
