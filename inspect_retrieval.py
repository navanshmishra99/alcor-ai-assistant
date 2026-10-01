from __future__ import annotations

import sys

from backend.app.knowledge.retriever import search_knowledge


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Question: ").strip()

    if not query:
        print("No question provided.")
        return

    results = search_knowledge(
        query=query,
        limit=10,
    )

    print("\n========== RETRIEVAL RESULTS ==========")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Title: {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"Vector distance: {result.get('distance')}")
        print(f"Text score: {result.get('text_score')}")
        print(f"OR text score: {result.get('or_text_score')}")
        print(f"Heading score: {result.get('heading_score')}")
        print(
            "Base relevance: "
            f"{result.get('base_relevance_score')}"
        )
        print(
            "Final relevance: "
            f"{result.get('relevance_score')}"
        )

    print("\n========================================")


if __name__ == "__main__":
    main()