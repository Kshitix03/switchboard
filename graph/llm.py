"""Shared Gemini client for graph nodes."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-flash-lite-latest"

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client
