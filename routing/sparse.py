"""Sparse retrieval: BM25 over the persisted card-text index.

BM25 has no native filter concept, so filters (service/domain/operation) are
applied post hoc against tool metadata, then the ranked list is truncated to
top_k -- same filter semantics as the dense side, applied after scoring.
"""

from __future__ import annotations

from routing import store


def _matches_filters(tool_id: str, filters: dict | None) -> bool:
    if not filters:
        return True
    record = store.records_by_id()[tool_id]
    for key, value in filters.items():
        actual = getattr(record, key, None)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


def search(query: str, top_k: int = 30, filters: dict | None = None) -> list[tuple[str, float]]:
    """Return [(tool_id, score), ...] ranked by BM25 score, filters applied post hoc."""
    data = store.bm25_data()
    bm25 = data["bm25"]
    ids = data["ids"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    filtered = [(tool_id, score) for tool_id, score in ranked if _matches_filters(tool_id, filters)]
    return filtered[:top_k]
