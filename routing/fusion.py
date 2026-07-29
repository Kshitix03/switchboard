"""Reciprocal rank fusion. Pure function over two (or more) ranked id lists."""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """rankings: list of ranked [(tool_id, score), ...] lists (score unused, rank order matters).
    Returns [(tool_id, fused_score), ...] sorted descending by fused score."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (tool_id, _score) in enumerate(ranking, start=1):
            fused[tool_id] = fused.get(tool_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
