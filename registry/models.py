"""ToolRecord and the on-disk schema store (PRD section 5.1, 5.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_STORE_PATH = Path(__file__).parent / "data" / "schemas.json"


class ToolRecord(BaseModel):
    id: str
    service: str
    domain: str
    name: str
    summary: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    utterances: list[str] = Field(default_factory=list)
    operation: Literal["read", "write"]
    risk: Literal["low", "medium", "high"]
    method: str
    path: str
    schema_ref: str
    requires_approval: bool
    source: str = "paypal"  # tags padded/non-paypal records for the benchmark report

    def card_text(self) -> str:
        """Text that gets embedded and BM25 indexed. Never includes the full schema."""
        parts = [
            f"{self.name}. {self.summary} {self.description}",
        ]
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        if self.utterances:
            parts.append(f"Examples: {' | '.join(self.utterances)}")
        return "\n".join(parts)


class SchemaStore:
    """Plain JSON file on disk, keyed by schema_ref. Loaded only at bind time."""

    def __init__(self, path: Path = SCHEMA_STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, schema_ref: str) -> dict:
        return self._data[schema_ref]

    def put(self, schema_ref: str, schema: dict) -> None:
        self._data[schema_ref] = schema

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def __contains__(self, schema_ref: str) -> bool:
        return schema_ref in self._data
