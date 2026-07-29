"""Cross-encoder reranking. Off-the-shelf, not fine-tuned (PRD section 6, 14)."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from registry.models import ToolRecord
from routing import store

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(
    query: str,
    candidate_ids: list[str],
    top_n: int = 6,
    records_by_id: dict[str, ToolRecord] | None = None,
) -> list[tuple[str, float]]:
    """Score (query, card_text) pairs for each candidate, return top_n [(tool_id, score), ...].

    records_by_id defaults to the production registry; the benchmark (Phase 7)
    passes its own per-tier records so reranking never touches production state.
    """
    if not candidate_ids:
        return []
    records_by_id = records_by_id if records_by_id is not None else store.records_by_id()
    pairs = [(query, records_by_id[tool_id].card_text()) for tool_id in candidate_ids]
    scores = [float(s) for s in _get_model().predict(pairs)]
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
