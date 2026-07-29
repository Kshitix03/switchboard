"""Phase 7 benchmark (PRD section 8, 10).

Two tiers of metric, both disclosed plainly rather than blended together:

1. Zero-LLM-call retrieval metrics: recall@6 and a top-1 accuracy PROXY,
   computed via embeddings + a local cross-encoder only. Run at full scale
   -- all scoreable golden queries x all 3 registry sizes x all 3 arms.
   No quota risk, so no coverage gaps.
2. A small real end-to-end sample (actual Gemini calls through the real
   two-step tool-selection logic) at the smallest (core, unpadded) tier
   only -- see eval/validate_proxy.py -- checking the zero-cost proxy
   isn't badly miscalibrated against real model behavior.

Arms:
  A. full_binding  -- no retrieval at all. Accuracy proxy = is the expected
     tool the single nearest neighbor by raw cosine similarity across the
     ENTIRE registry (no top-k cutoff)? Prompt size = every tool's full
     flattened schema bound at once -- computed once per registry size
     (query-independent), not per query.
  B. dense_only    -- dense embedding search, top 6, no BM25, no rerank.
  C. hybrid_rerank -- the production funnel: dense(30)+sparse(30)->RRF->
     cross-encoder rerank->top 6. Same code path as routing/router.py,
     pointed at the benchmark's isolated per-tier store.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from qdrant_client.models import Distance, PointStruct, VectorParams

from eval.bench_store import COLLECTION_NAME, BenchStore
from graph.schema_utils import flatten_for_prompt
from registry.models import SchemaStore
from routing import fusion
from routing import rerank as rerank_module

load_dotenv()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
DATA_DIR = Path(__file__).parent / "data"
QUERY_EMBEDDING_CACHE_PATH = DATA_DIR / "query_embedding_cache.json"
DOCS_DIR = Path(__file__).parent.parent / "docs"

SIZES = [108, 250, 500]
ARMS = ["full_binding", "dense_only", "hybrid_rerank"]
DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_K = 60
FINAL_TOP_N = 6
EMBED_MODEL = "gemini-embedding-001"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_golden_set() -> list[dict]:
    lines = GOLDEN_SET_PATH.read_text(encoding="utf-8").strip().splitlines()
    all_queries = [json.loads(line) for line in lines]
    # only queries with a real registry tool id as the target are scoreable
    # for retrieval metrics -- excludes needs_clarification (null) and meta
    # (system_search, not a registry entry)
    return [q for q in all_queries if q["expected_tool"] and q["expected_tool"] != "system_search"]


def embed_queries(queries: list[dict]) -> dict[str, list[float]]:
    """query id -> embedding vector, disk-cached by query text hash."""
    cache = json.loads(QUERY_EMBEDDING_CACHE_PATH.read_text(encoding="utf-8")) if QUERY_EMBEDDING_CACHE_PATH.exists() else {}
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    vectors: dict[str, list[float]] = {}
    to_embed = []
    for q in queries:
        h = _hash(q["query"])
        entry = cache.get(q["id"])
        if entry and entry["hash"] == h:
            vectors[q["id"]] = entry["vector"]
        else:
            to_embed.append(q)

    print(f"{len(queries)} queries, {len(to_embed)} need embedding, {len(queries) - len(to_embed)} cached")
    for i in range(0, len(to_embed), 10):
        batch = to_embed[i:i + 10]
        texts = [q["query"] for q in batch]
        result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
        for q, emb in zip(batch, result.embeddings):
            vectors[q["id"]] = emb.values
            cache[q["id"]] = {"hash": _hash(q["query"]), "vector": emb.values}
        time.sleep(1.0)

    QUERY_EMBEDDING_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return vectors


def dense_ids(bench: BenchStore, qvec: list[float], top_k: int) -> list[str]:
    hits = bench.qdrant().query_points(COLLECTION_NAME, query=qvec, limit=top_k).points
    return [h.payload["tool_id"] for h in hits]


def sparse_ids(bench: BenchStore, query_text: str, top_k: int) -> list[str]:
    bm25, ids = bench.bm25()
    scores = bm25.get_scores(query_text.lower().split())
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [tid for tid, _ in ranked[:top_k]]


def full_binding_top1(bench: BenchStore, qvec: list[float]) -> str:
    """Nearest neighbor across the ENTIRE registry, no cutoff -- the proxy
    for 'if the model saw everything, would naive similarity find it'."""
    hits = bench.qdrant().query_points(COLLECTION_NAME, query=qvec, limit=1).points
    return hits[0].payload["tool_id"] if hits else None


def full_binding_prompt_chars(bench: BenchStore) -> int:
    """Query-independent: every tool's full flattened request schema, once."""
    schema_store = SchemaStore()
    total = 0
    for r in bench.records:
        try:
            schema_entry = schema_store.get(r.schema_ref)
            flat = flatten_for_prompt(schema_entry["request_body"])
            total += len(r.card_text()) + len(json.dumps(flat))
        except KeyError:
            total += len(r.card_text())
    return total


def hybrid_rerank_ids(bench: BenchStore, query_text: str, qvec: list[float]) -> tuple[list[str], list[str]]:
    """Returns (fused_ids_for_recall, reranked_top6_ids)."""
    dense = dense_ids(bench, qvec, DENSE_TOP_K)
    sparse = sparse_ids(bench, query_text, SPARSE_TOP_K)
    fused = fusion.reciprocal_rank_fusion(
        [[(d, 0.0) for d in dense], [(s, 0.0) for s in sparse]], k=RRF_K,
    )
    fused_ids = [t for t, _ in fused]
    reranked = rerank_module.rerank(query_text, fused_ids, top_n=FINAL_TOP_N, records_by_id=bench.records_by_id)
    return fused_ids, [t for t, _ in reranked]


def avg_bound_schema_chars(bench: BenchStore, candidate_ids: list[str]) -> int:
    schema_store = SchemaStore()
    total = 0
    for tid in candidate_ids:
        try:
            schema_entry = schema_store.get(bench.records_by_id[tid].schema_ref)
            flat = flatten_for_prompt(schema_entry["request_body"])
            total += len(bench.records_by_id[tid].card_text()) + len(json.dumps(flat))
        except KeyError:
            total += len(bench.records_by_id[tid].card_text())
    return total


def run_size(size: int, queries: list[dict], query_vectors: dict[str, list[float]]) -> dict:
    bench = BenchStore(size)
    print(f"\n=== registry size {size} ({len(bench.records)} tools) ===")

    results = {arm: {"hits_top1": 0, "hits_recall6": 0, "latencies_ms": [], "prompt_chars": []} for arm in ARMS}

    full_binding_chars = full_binding_prompt_chars(bench)

    for q in queries:
        expected = q["expected_tool"]
        qvec = query_vectors[q["id"]]

        # Arm A: full binding
        t0 = time.perf_counter()
        top1 = full_binding_top1(bench, qvec)
        results["full_binding"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
        results["full_binding"]["hits_top1"] += int(top1 == expected)
        results["full_binding"]["hits_recall6"] += int(top1 == expected)  # no top-6 concept; top1==recall proxy
        results["full_binding"]["prompt_chars"].append(full_binding_chars)

        # Arm B: dense only, top 6
        t0 = time.perf_counter()
        d_ids = dense_ids(bench, qvec, FINAL_TOP_N)
        results["dense_only"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
        results["dense_only"]["hits_top1"] += int(bool(d_ids) and d_ids[0] == expected)
        results["dense_only"]["hits_recall6"] += int(expected in d_ids)
        results["dense_only"]["prompt_chars"].append(avg_bound_schema_chars(bench, d_ids))

        # Arm C: hybrid + rerank (production funnel)
        t0 = time.perf_counter()
        fused_ids, reranked_ids = hybrid_rerank_ids(bench, q["query"], qvec)
        results["hybrid_rerank"]["latencies_ms"].append((time.perf_counter() - t0) * 1000)
        results["hybrid_rerank"]["hits_top1"] += int(bool(reranked_ids) and reranked_ids[0] == expected)
        results["hybrid_rerank"]["hits_recall6"] += int(expected in reranked_ids)
        results["hybrid_rerank"]["prompt_chars"].append(avg_bound_schema_chars(bench, reranked_ids))

    n = len(queries)
    summary = {}
    for arm in ARMS:
        r = results[arm]
        lat = sorted(r["latencies_ms"])
        summary[arm] = {
            "top1_accuracy": r["hits_top1"] / n,
            "recall_at_6": r["hits_recall6"] / n,
            "avg_prompt_chars": statistics.mean(r["prompt_chars"]),
            "latency_p50_ms": lat[len(lat) // 2],
            "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        }
        print(f"  {arm:15s} top1_acc={summary[arm]['top1_accuracy']:.2f}  "
              f"recall@6={summary[arm]['recall_at_6']:.2f}  "
              f"avg_chars={summary[arm]['avg_prompt_chars']:.0f}  "
              f"p50={summary[arm]['latency_p50_ms']:.1f}ms  p95={summary[arm]['latency_p95_ms']:.1f}ms")
    return summary


def main() -> None:
    queries = load_golden_set()
    print(f"golden set: {len(queries)}/60 queries are scoreable for retrieval metrics "
          f"(excludes needs_clarification + meta categories)")
    query_vectors = embed_queries(queries)

    all_results = {}
    for size in SIZES:
        all_results[size] = run_size(size, queries, query_vectors)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
