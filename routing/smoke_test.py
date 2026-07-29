"""Run 10 sample queries through the funnel and print each stage so it's
eyeball-inspectable (PRD Phase 2 acceptance criterion)."""

from __future__ import annotations

import logging

from routing.router import retrieve

QUERIES = [
    "Send an invoice for $50 to jane@acme.com",
    "Create a draft invoice for consulting work",
    "Refund a customer for a captured payment",
    "Record a manual refund on an invoice",
    "Cancel invoice INV2-1234",
    "List all my recent transactions",
    "What's the status of dispute D-9982?",
    "Send a payout to multiple recipients",
    "Create a product in the catalog",
    "Delete a saved payment token from the vault",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for i, query in enumerate(QUERIES, start=1):
        print(f"\n{'=' * 80}\n[{i}] QUERY: {query}\n{'=' * 80}")
        result = retrieve(query)
        print("\nFinal top 6:")
        for tool_id, score in result.scores.items():
            print(f"  {score:.4f}  {tool_id}")


if __name__ == "__main__":
    main()
