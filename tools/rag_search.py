"""rag_search: dense search over the docs collection (PRD section 4.2).

Registered as a normal ToolRecord in the main registry -- it competes for
retrieval slots on equal footing with every PayPal API tool. When actually
selected and executed, this module runs a real search instead of the mock
response every other (PayPal API) tool gets.
"""

from __future__ import annotations

from google import genai

from routing import store

EMBED_MODEL = "gemini-embedding-001"


def search(client: genai.Client, query: str, top_k: int = 3) -> list[dict]:
    qdrant_client = store.ensure_docs_collection()
    qvec = client.models.embed_content(model=EMBED_MODEL, contents=[query]).embeddings[0].values
    hits = qdrant_client.query_points(store.DOCS_COLLECTION_NAME, query=qvec, limit=top_k).points
    return [
        {"title": h.payload["title"], "text": h.payload["text"], "score": h.score}
        for h in hits
    ]
