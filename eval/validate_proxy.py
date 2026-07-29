"""Validates the zero-LLM-call retrieval proxy against real end-to-end tool
selection, at the core (108, unpadded) registry tier only, on a small
deterministic sample -- see DECISIONS.md Phase 7 for why the full benchmark
can't run real LLM calls at scale (quota).

Uses the actual two-step fill logic (graph.nodes._select_tool) with real
Gemini calls -- not a separate reimplementation -- so this validates the
real production code path, not an approximation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.bench_store import BenchStore
from eval.run_benchmark import embed_queries, hybrid_rerank_ids, load_golden_set
from graph.llm import get_client
from graph.nodes import _select_tool

SAMPLE_SIZE = 15
OUT_PATH = Path(__file__).parent.parent / "docs" / "proxy_validation.json"


def main() -> None:
    queries = load_golden_set()
    # first N in file order after filtering -- deterministic, not cherry-picked
    sample = queries[:SAMPLE_SIZE]
    query_vectors = embed_queries(sample)
    bench = BenchStore(108)
    client = get_client()

    results = []
    for q in sample:
        qvec = query_vectors[q["id"]]
        _, reranked = hybrid_rerank_ids(bench, q["query"], qvec)
        proxy_pick = reranked[0] if reranked else None
        proxy_correct = proxy_pick == q["expected_tool"]

        fake_state = {"candidate_tools": [{"tool_id": tid, "score": 0.0} for tid in reranked]}
        try:
            real_pick = _select_tool(client, q["query"], fake_state)
        except Exception as e:
            real_pick = f"ERROR: {e}"
        real_correct = real_pick == q["expected_tool"]

        result = {
            "id": q["id"], "query": q["query"], "expected": q["expected_tool"],
            "proxy_pick": proxy_pick, "proxy_correct": proxy_correct,
            "real_pick": real_pick, "real_correct": real_correct,
            "agree": proxy_pick == real_pick,
        }
        results.append(result)
        print(f"{q['id']}: expected={q['expected_tool']} | "
              f"proxy={proxy_pick} ({'OK' if proxy_correct else 'X'}) | "
              f"real={real_pick} ({'OK' if real_correct else 'X'})")

    n = len(results)
    proxy_acc = sum(r["proxy_correct"] for r in results) / n
    real_acc = sum(r["real_correct"] for r in results) / n
    agreement = sum(r["agree"] for r in results) / n
    print(f"\nn={n}  proxy_accuracy={proxy_acc:.2f}  real_accuracy={real_acc:.2f}  proxy/real agreement={agreement:.2f}")

    OUT_PATH.write_text(json.dumps({
        "n": n, "proxy_accuracy": proxy_acc, "real_accuracy": real_acc,
        "agreement": agreement, "results": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
