from __future__ import annotations

import time

from backend.app.knowledge.ingest import get_chunk_count
from backend.app.knowledge.retriever import search_knowledge


DEFAULT_LIMIT = 2


def build_context(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> dict:

    total_start = time.perf_counter()

    if not query or not query.strip():

        print(
            "\n========== RAG TIMING =========="
        )
        print(
            "Results          : 0"
        )
        print(
            "Retrieval        : 0.00s"
        )
        print(
            "RAG Total        : 0.00s"
        )
        print(
            "===============================\n"
        )

        return {
            "has_context": False,
            "context": "",
            "sources": [],
        }

    if get_chunk_count() == 0:
        return {
            "has_context": False,
            "context": "",
            "sources": [],
        }

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

        title = result.get(
            "title",
            "Untitled",
        )

        url = result.get(
            "url",
            "",
        )

        content = (
            result.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        context_parts.append(
            f"Source: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

        if url and url not in sources_by_url:

            sources_by_url[url] = {
                "title": title,
                "url": url,
            }

    total_time = (
        time.perf_counter()
        - total_start
    )

    context = "\n\n---\n\n".join(
        context_parts
    )

    print(
        "\n========== RAG TIMING =========="
    )
    print(
        f"Results          : {len(results)}"
    )
    print(
        f"Context Chars    : {len(context)}"
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

    if not context_parts:

        return {
            "has_context": False,
            "context": "",
            "sources": [],
        }

    return {
        "has_context": True,
        "context": context,
        "sources": list(
            sources_by_url.values()
        ),
    }