"""Minimal trace log: one JSON line per executed tool call.

This is the seed of PRD Phase 6's full instrumentation (every node, every
funnel stage). For now it's just enough for system_search's "status of my
last request" to read real history instead of nothing -- a genuine trace
store, not a stub, just a narrow one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

TRACE_LOG_PATH = Path(__file__).parent.parent / "traces.jsonl"


def log_execution(trace_id: str, tool_id: str, args: dict, result: dict) -> None:
    entry = {
        "trace_id": trace_id,
        "timestamp": time.time(),
        "tool_id": tool_id,
        "args": args,
        "status": result.get("status"),
        "idempotency_key": result.get("idempotency_key"),
    }
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def last_n(n: int = 5) -> list[dict]:
    if not TRACE_LOG_PATH.exists():
        return []
    lines = TRACE_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-n:]]
