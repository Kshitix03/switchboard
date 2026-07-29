"""The funnel, one entry point (PRD section 4.1, 6).

dense(30) + sparse(30) -> RRF -> rerank -> top 6. Logs candidate ids at every
stage so the pipeline is inspectable.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google import genai

from routing import dense, fusion, rerank, sparse, store
from registry.models import ToolRecord

load_dotenv()

logger = logging.getLogger("routing.router")

DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_K = 60
FINAL_TOP_N = 6

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


class RetrievalResult:
    def __init__(self, records: list[ToolRecord], scores: dict[str, float], funnel: dict[str, list[str]]):
        self.records = records
        self.scores = scores
        self.funnel = funnel  # stage name -> ordered list of tool ids


def retrieve(query: str, filters: dict | None = None, top_n: int = FINAL_TOP_N) -> RetrievalResult:
    client = _get_genai_client()

    dense_hits = dense.search(client, query, top_k=DENSE_TOP_K, filters=filters)
    sparse_hits = sparse.search(query, top_k=SPARSE_TOP_K, filters=filters)
    logger.info("dense top ids: %s", [t for t, _ in dense_hits])
    logger.info("sparse top ids: %s", [t for t, _ in sparse_hits])

    fused = fusion.reciprocal_rank_fusion([dense_hits, sparse_hits], k=RRF_K)
    fused_ids = [t for t, _ in fused]
    logger.info("fused (RRF) ids: %s", fused_ids)

    reranked = rerank.rerank(query, fused_ids, top_n=top_n)
    reranked_ids = [t for t, _ in reranked]
    logger.info("reranked top-%d ids: %s", top_n, reranked_ids)

    records_by_id = store.records_by_id()
    records = [records_by_id[tool_id] for tool_id, _ in reranked]
    scores = {tool_id: score for tool_id, score in reranked}

    funnel = {
        "dense": [t for t, _ in dense_hits],
        "sparse": [t for t, _ in sparse_hits],
        "fused": fused_ids,
        "reranked": reranked_ids,
    }
    return RetrievalResult(records=records, scores=scores, funnel=funnel)
