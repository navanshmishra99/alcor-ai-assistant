from __future__ import annotations

import logging
import os
import time

from backend.app.knowledge.ingest import get_chunk_count
from backend.app.knowledge.retriever import search_knowledge


logger = logging.getLogger("ask_alcor.rag")


# ============================================================
# Configuration
# ============================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Number of results requested from the retrieval pipeline.
DEFAULT_LIMIT = _env_int("RAG_LIMIT", 4)

# Maximum context sent to the AI layer.
MAX_CONTEXT_CHARS = _env_int(
    "RAG_MAX_CONTEXT_CHARS",
    6000,
)

# Relative filtering for the FINAL reranker scores.
#
# Example:
#
#   top rerank score = 2.0
#   ratio = 0.60
#
#   results >= 1.20 are retained.
#
# This is only used when rerank_score is available.
RERANK_RELATIVE_SCORE_RATIO = _env_float(
    "RAG_RERANK_RELATIVE_SCORE_RATIO",
    0.60,
)

# Existing hybrid-score filtering is retained only as a fallback
# for retrieval results that do not contain rerank_score.
MIN_RELEVANCE_SCORE = _env_float(
    "RAG_MIN_RELEVANCE_SCORE",
    0.35,
)

RELATIVE_SCORE_RATIO = _env_float(
    "RAG_RELATIVE_SCORE_RATIO",
    0.60,
)


SEPARATOR = "\n\n---\n\n"


def _empty() -> dict:
    return {
        "has_context": False,
        "context": "",
        "sources": [],
    }


def _filter_relevant_results(
    results: list[dict],
) -> list[dict]:
    """
    Filter retrieved results using the best available scoring layer.

    Priority:

    1. CrossEncoder `rerank_score`
    2. Hybrid retrieval `relevance_score`

    The CrossEncoder score is used whenever it is available because
    reranking happens after the initial hybrid retrieval and therefore
    represents the final semantic ranking of the candidates.

    No domain-specific terms or rules are used.
    """

    if not results:
        return []

    # --------------------------------------------------------
    # 1. Prefer CrossEncoder reranking
    # --------------------------------------------------------

    reranked_results = [
        result
        for result in results
        if isinstance(
            result.get("rerank_score"),
            (int, float),
        )
    ]

    if reranked_results:
        # Always work in descending CrossEncoder order.
        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        best_score = reranked_results[0]["rerank_score"]

        logger.info(
            "rag_rerank best_score=%.4f ratio=%.4f candidates=%d",
            best_score,
            RERANK_RELATIVE_SCORE_RATIO,
            len(reranked_results),
        )

        # If relative filtering is disabled, keep the final
        # reranked results as-is.
        if RERANK_RELATIVE_SCORE_RATIO <= 0:
            return reranked_results

        # CrossEncoder scores are not guaranteed to be positive.
        # A multiplicative ratio is therefore not meaningful when
        # the best score is zero or negative.
        #
        # In that case, preserve the reranked ordering rather than
        # making an unsafe assumption about the score scale.
        if best_score <= 0:
            logger.info(
                "rag_rerank_non_positive_best_score "
                "best_score=%.4f; preserving ranked results",
                best_score,
            )
            return reranked_results

        relative_threshold = (
            best_score
            * RERANK_RELATIVE_SCORE_RATIO
        )

        kept = [
            result
            for result in reranked_results
            if result["rerank_score"]
            >= relative_threshold
        ]

        # Always keep the strongest result.
        if not kept:
            kept = [reranked_results[0]]

        logger.info(
            "rag_rerank_filtered before=%d after=%d "
            "threshold=%.4f",
            len(reranked_results),
            len(kept),
            relative_threshold,
        )

        return kept

    # --------------------------------------------------------
    # 2. Fallback to hybrid retrieval score
    # --------------------------------------------------------

    scored_results = [
        result
        for result in results
        if isinstance(
            result.get("relevance_score"),
            (int, float),
        )
    ]

    if not scored_results:
        logger.warning(
            "rag_missing_relevance_scores results=%d",
            len(results),
        )
        return results

    best_score = max(
        result["relevance_score"]
        for result in scored_results
    )

    logger.info(
        "rag_hybrid best_score=%.4f min_score=%.4f "
        "ratio=%.4f",
        best_score,
        MIN_RELEVANCE_SCORE,
        RELATIVE_SCORE_RATIO,
    )

    # No sufficiently relevant hybrid result.
    if best_score < MIN_RELEVANCE_SCORE:
        logger.info(
            "rag_hybrid_rejected best_score=%.4f "
            "min_score=%.4f",
            best_score,
            MIN_RELEVANCE_SCORE,
        )
        return []

    if RELATIVE_SCORE_RATIO <= 0:
        return results

    relative_threshold = (
        best_score
        * RELATIVE_SCORE_RATIO
    )

    kept = [
        result
        for result in results
        if isinstance(
            result.get("relevance_score"),
            (int, float),
        )
        and result["relevance_score"]
        >= relative_threshold
    ]

    return kept or [max(
        scored_results,
        key=lambda result: result["relevance_score"],
    )]


def build_context(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """
    Retrieve knowledge for a query and format it for the AI layer.

    The retrieval pipeline may return CrossEncoder-reranked results.
    When available, rerank_score is treated as the final ranking signal.

    The context format remains:

        Source:
        URL:
        Content:

    so existing AI and grounding components remain compatible.
    """

    total_start = time.perf_counter()

    if not query or not query.strip():
        return _empty()

    if get_chunk_count() == 0:
        logger.warning(
            "rag_empty_index query=%r",
            query[:120],
        )
        return _empty()

    retrieval_start = time.perf_counter()

    results = search_knowledge(
        query=query,
        limit=limit,
    ) or []

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not results:
        logger.info(
            "rag_no_results retrieval=%.3fs query=%r",
            retrieval_time,
            query[:120],
        )
        return _empty()

    # --------------------------------------------------------
    # Final relevance filtering
    # --------------------------------------------------------

    original_result_count = len(results)

    results = _filter_relevant_results(results)

    if not results:
        logger.info(
            "rag_no_relevant_results original=%d "
            "retrieval=%.3fs query=%r",
            original_result_count,
            retrieval_time,
            query[:120],
        )
        return _empty()

    # --------------------------------------------------------
    # Context construction
    # --------------------------------------------------------

    context_parts: list[str] = []
    sources_by_url: dict[str, dict] = {}
    seen_content: set[str] = set()
    used_chars = 0

    for result in results:

        title = result.get("title") or "Untitled"
        url = result.get("url") or ""
        content = (
            result.get("content") or ""
        ).strip()

        if not content:
            continue

        # Prevent duplicate chunks.
        fingerprint = " ".join(
            content.lower().split()
        )[:300]

        if fingerprint in seen_content:
            continue

        block = (
            f"Source: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

        # Results are already ordered by the final scoring layer,
        # so stronger results are processed first.
        if (
            context_parts
            and used_chars + len(block)
            > MAX_CONTEXT_CHARS
        ):
            break

        seen_content.add(fingerprint)

        context_parts.append(block)

        used_chars += (
            len(block)
            + len(SEPARATOR)
        )

        if url and url not in sources_by_url:
            sources_by_url[url] = {
                "title": title,
                "url": url,
            }

    if not context_parts:
        return _empty()

    context = SEPARATOR.join(context_parts)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    best_rerank_score = max(
        (
            result.get("rerank_score")
            for result in results
            if isinstance(
                result.get("rerank_score"),
                (int, float),
            )
        ),
        default=None,
    )

    best_relevance_score = max(
        (
            result.get("relevance_score")
            for result in results
            if isinstance(
                result.get("relevance_score"),
                (int, float),
            )
        ),
        default=None,
    )

    logger.info(
        "rag_result results=%d used=%d "
        "context_chars=%d rerank_score=%s "
        "relevance_score=%s retrieval=%.3fs "
        "total=%.3fs query=%r",
        len(results),
        len(context_parts),
        len(context),
        (
            f"{best_rerank_score:.4f}"
            if isinstance(
                best_rerank_score,
                (int, float),
            )
            else "n/a"
        ),
        (
            f"{best_relevance_score:.4f}"
            if isinstance(
                best_relevance_score,
                (int, float),
            )
            else "n/a"
        ),
        retrieval_time,
        time.perf_counter() - total_start,
        query[:120],
    )

    return {
        "has_context": True,
        "context": context,
        "sources": list(
            sources_by_url.values()
        ),
    }