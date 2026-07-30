# Switchboard

A retrieval-gated agentic system for 500+ tools.

**Thesis:** an LLM given 500 tool definitions performs worse than one given 6 — tool descriptions
consume context, similar tools blur together, and the model picks wrong or hallucinates parameters.
Prompt engineering does not fix this; it is a scaling wall. Tool selection at scale is a **retrieval
problem, not a prompting problem**. If the model never sees more than 8 tools per turn, the prompt
it receives stops growing with the registry, and accuracy stops being a direct function of registry
size.

![Prompt size vs registry size](docs/scaling.png)

Full binding — showing the model every tool at once — grows from ~413K to ~642K characters of
prompt as the registry scales 108→500 real + padded tools. The retrieval-gated arms (dense-only,
hybrid+rerank) stay flat at ~17–29K characters regardless of registry size. That's the thesis,
measured directly, with zero LLM calls needed to produce this chart.

| Registry size | Arm | Recall@6 | Avg. prompt chars | Latency p50 (ms) |
|---|---|---|---|---|
| 108 | A: Full binding | 1.00 (trivial, unfiltered) | 412,865 | 9.5 |
| 108 | B: Dense-only, top 6 | 0.83 | 17,922 | 6.7 |
| 108 | C: Hybrid + rerank, top 6 | 0.72 | 28,809 | 2465.8 |
| 500 | A: Full binding | 1.00 (trivial, unfiltered) | 642,246 | 17.9 |
| 500 | B: Dense-only, top 6 | 0.80 | 17,042 | 17.4 |
| 500 | C: Hybrid + rerank, top 6 | 0.70 | 27,217 | 2429.7 |

Full table, chart, methodology, and named limitations (including a real finding that the untuned
reranker sometimes hurts recall vs. dense-only alone) are in `docs/RESULTS.md`. The full report —
architecture, state management, error handling, framework trade-offs, and limitations — is in
`REPORT.md`, with a condensed 5-page version at
[`docs/Switchboard_Report.pdf`](docs/Switchboard_Report.pdf). The complete build-by-build decision
log is in `DECISIONS.md`.

## Run it

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .    # Windows; always use .venv, never install globally
cp .env.example .env                            # fill in GEMINI_API_KEY

.venv\Scripts\python.exe cli.py chat            # interactive, asks for approval on writes
.venv\Scripts\python.exe cli.py chat --dry-run   # prints the exact call, never executes
```

`cli.py chat` also launches a local Phoenix trace viewer automatically (http://localhost:6006) and
traces every graph node — funnel candidate ids at each stage, rerank scores, LLM token counts,
retries, and the final selected tool. Pass `--no-trace` to skip it. If port 4317 is already in use
on your machine (e.g. by Jaeger), set `PHOENIX_GRPC_PORT`; `cli.py` defaults it to `4327` if unset.

## Honesty flags

- **The golden set (`eval/golden_set.jsonl`, 60 queries) is entirely AI-generated, not
  hand-written.** A direct deviation from the PRD's explicit instruction to write it yourself, made
  under real time constraints, disclosed rather than misrepresented — see `DECISIONS.md`'s Phase 3
  section for the full context and the reasoning risk this creates.
- Registry above the ~22-tool invoicing core is real tools from real OpenAPI specs (PayPal + GitHub
  for benchmark padding), not synthetic noise — but PayPal domains were added incrementally to
  support specific golden-set near-miss pairs, not for coverage breadth alone, and padding is from
  an unrelated domain (GitHub's API), which likely understates real-world retrieval degradation.
- Execution is fully mocked; no live PayPal sandbox calls are made anywhere in this build.
- Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and embedding model (`gemini-embedding-001`) are
  off-the-shelf, not tuned — and Phase 7 shows this measurably costing recall in some cases, not
  just as an abstract caveat.
- The Phase 7 benchmark's tool-selection accuracy numbers are a small real sample (n=15, one
  registry size) plus a zero-LLM-call proxy shown to underestimate true accuracy by ~26 points —
  not a full accuracy-vs-scale curve at every size. See `docs/RESULTS.md`.
- Entity resolution (the `RESOLVE:`-marked references in the golden set) and large-result artifact
  handles are in the state schema but not implemented — see `REPORT.md` section 4 and 8.
- Benchmark ran once per configuration, not averaged over seeds.

## Models

- LLM: `gemini-flash-lite-latest` (graph nodes: plan/fill/observe/chat). Utterance generation
  (Phase 1, one-time) used `gemini-flash-latest` — both are free-tier aliases with tight daily
  quotas on this account; see `DECISIONS.md` if you hit a 429 and need to swap models.
- Embeddings: `gemini-embedding-001` (3072-dim)

## Registry pipeline

```
.venv\Scripts\python.exe -m registry.ingest_openapi     # OpenAPI specs -> registry/data/tools.json + schemas.json
.venv\Scripts\python.exe -m registry.enrich             # + utterances/keywords -> tools_enriched.json (LLM, disk-cached)
.venv\Scripts\python.exe -m registry.add_internal_tools # + rag_search (internal tool, not from an OpenAPI spec)
.venv\Scripts\python.exe -m registry.index              # embed + Qdrant (in-memory) + BM25 (persisted)
.venv\Scripts\python.exe -m registry.ingest_docs        # docs corpus for rag_search (from spec info/tag descriptions)
.venv\Scripts\python.exe -m registry.index_docs         # embed the docs corpus
```

## Tests

```
.venv\Scripts\python.exe tests/test_graph_smoke.py    # Phase 4: dry-run + approval gate
.venv\Scripts\python.exe tests/test_phase5_smoke.py   # Phase 5: rag_search + system_search
.venv\Scripts\python.exe tests/test_phase6_smoke.py   # Phase 6: Phoenix tracing + span attributes
```

## Benchmark (Phase 7)

```
.venv\Scripts\python.exe -m eval.pad_registry       # builds eval/data/registry_{108,250,500}.json
.venv\Scripts\python.exe -m eval.embed_registries   # embeds padding tools (reuses core cache)
.venv\Scripts\python.exe -m eval.run_benchmark      # zero-LLM-call metrics, all sizes/arms
.venv\Scripts\python.exe -m eval.validate_proxy     # small real-LLM validation sample (n=15)
.venv\Scripts\python.exe -m eval.build_report       # docs/scaling.png, recall.png, RESULTS.md
```
