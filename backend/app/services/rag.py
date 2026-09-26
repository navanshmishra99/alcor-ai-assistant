from __future__ import annotations

import time

from backend.app.knowledge.retriever import search_knowledge


def build_context(
    query: str,
    limit: int = 5,
) -> dict:

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()

    results = search_knowledge(
        query=query,
        limit=limit,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not results:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            "\n========== RAG TIMING =========="
        )
        print(
            "Results          : 0"
        )
        print(
            f"Retrieval        : {retrieval_time:.2f}s"
        )
        print(
            f"RAG Total        : {total_time:.2f}s"
        )
        print(
            "===============================\n"
        )

        return {
            "has_context": False,
            "context": "",
            "sources": [],
        }

    context_parts: list[str] = []

    sources_by_url: dict[str, dict] = {}

    for result in results:

        context_parts.append(
            f"Source: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content:\n{result['content']}"
        )

        url = result["url"]

        if url not in sources_by_url:

            sources_by_url[url] = {
                "title": result["title"],
                "url": url,
            }

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n========== RAG TIMING =========="
    )
    print(
        f"Results          : {len(results)}"
    )
    print(
        f"Retrieval        : {retrieval_time:.2f}s"
    )
    print(
        f"RAG Total        : {total_time:.2f}s"
    )
    print(
        "===============================\n"
    )

    return {
        "has_context": True,
        "context": "\n\n---\n\n".join(
            context_parts
        ),
        "sources": list(
            sources_by_url.values()
        ),
    }