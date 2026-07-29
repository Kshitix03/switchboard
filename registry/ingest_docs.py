"""Build a small docs corpus for rag_search from real PayPal-authored prose
already embedded in the OpenAPI specs (info.description + tags[].description).

developer.paypal.com itself is a JS-rendered Redocly SPA and returns an empty
shell to non-browser fetches, so it isn't scrapable in the time available for
this build -- see DECISIONS.md. This corpus is real PayPal text, just sourced
from the spec files rather than the rendered docs site.
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_SPECS_DIR = Path(__file__).parent / "data" / "raw_specs"
DOCS_OUT_PATH = Path(__file__).parent / "data" / "docs.json"

MIN_DESC_LEN = 120  # skip near-empty tag descriptions; keeps the corpus to a curated ~25-30 chunks


def build_docs() -> list[dict]:
    docs: list[dict] = []
    for spec_path in sorted(RAW_SPECS_DIR.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        info = spec.get("info", {})
        title = info.get("title", spec_path.stem)
        source = spec_path.stem

        overview = info.get("description", "")
        if len(overview) >= MIN_DESC_LEN:
            docs.append({
                "id": f"docs.{source}.overview",
                "title": f"{title} API Overview",
                "text": " ".join(overview.split()),
                "source": source,
            })

        for tag in spec.get("tags", []):
            desc = tag.get("description", "")
            if len(desc) < MIN_DESC_LEN:
                continue
            name = tag.get("name", "unknown")
            docs.append({
                "id": f"docs.{source}.tag.{name}",
                "title": f"{title}: {name}",
                "text": " ".join(desc.split()),
                "source": source,
            })
    return docs


def main() -> None:
    docs = build_docs()
    DOCS_OUT_PATH.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"{len(docs)} doc chunks -> {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
