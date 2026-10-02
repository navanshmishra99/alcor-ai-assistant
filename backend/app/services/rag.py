from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

from ..knowledge.ingest import get_chunk_count
from ..knowledge.retriever import search_knowledge


logger = logging.getLogger("ask_alcor.rag")


# ============================================================
# Configuration helpers
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


# ============================================================
# Retrieval / RAG configuration
# ============================================================

# Final number of chunks normally allowed into the context.
DEFAULT_LIMIT = _env_int("RAG_LIMIT", 4)

# Maximum total characters sent to the AI layer.
MAX_CONTEXT_CHARS = _env_int(
    "RAG_MAX_CONTEXT_CHARS",
    6000,
)

# Number of reranked candidates that RAG should inspect.
#
# This should be >= the number of candidates returned by the
# retriever after reranking.
#
# IMPORTANT:
# The retriever must also be configured to rerank this many
# candidates. See RERANKER_MAX_CANDIDATES in .env.
RAG_CANDIDATE_POOL = _env_int(
    "RAG_CANDIDATE_POOL",
    30,
)

# Maximum number of chunks that may come from one document.
#
# This prevents one document from flooding the context while
# still allowing multiple chunks from the same document when
# they are independently relevant.
MAX_CHUNKS_PER_DOCUMENT = _env_int(
    "RAG_MAX_CHUNKS_PER_DOCUMENT",
    3,
)

# Maximum number of distinct documents represented in the
# final context.
MAX_DOCUMENTS = _env_int(
    "RAG_MAX_DOCUMENTS",
    4,
)

# ============================================================
# CrossEncoder filtering
# ============================================================

# CrossEncoder scores are raw model scores.
#
# A multiplicative threshold such as:
#
#     score >= best_score * 0.60
#
# is not a reliable interpretation of those scores.
#
# Set to 0 to disable this filter.
RERANK_RELATIVE_SCORE_RATIO = _env_float(
    "RAG_RERANK_RELATIVE_SCORE_RATIO",
    0.0,
)

# ============================================================
# Hybrid-score filtering
# ============================================================

MIN_RELEVANCE_SCORE = _env_float(
    "RAG_MIN_RELEVANCE_SCORE",
    0.35,
)

RELATIVE_SCORE_RATIO = _env_float(
    "RAG_RELATIVE_SCORE_RATIO",
    0.60,
)


SEPARATOR = "\n\n---\n\n"


# ============================================================
# Empty result
# ============================================================


def _empty() -> dict:
    return {
        "has_context": False,
        "context": "",
        "sources": [],
    }


# ============================================================
# Score helpers
# ============================================================


def _rerank_score(result: dict) -> float | None:
    value = result.get("rerank_score")

    if isinstance(value, (int, float)):
        return float(value)

    return None


def _relevance_score(result: dict) -> float | None:
    value = result.get("relevance_score")

    if isinstance(value, (int, float)):
        return float(value)

    return None


# ============================================================
# Document identity
# ============================================================


def _document_key(result: dict) -> str:
    """
    Return a stable generic document identifier.

    Prefer document_id because multiple chunks from the same
    document should be grouped together.

    Fall back to URL when document_id is unavailable.
    """

    document_id = result.get("document_id")

    if document_id is not None:
        value = str(document_id).strip()

        if value:
            return f"id:{value}"

    url = str(result.get("url") or "").strip()

    if url:
        return f"url:{url}"

    title = str(result.get("title") or "").strip()

    if title:
        return f"title:{title.lower()}"

    return f"anonymous:{id(result)}"


# ============================================================
# Relevance filtering
# ============================================================


def _filter_relevant_results(
    results: list[dict],
) -> list[dict]:
    """
    Filter retrieved results using the best available scoring layer.

    Priority:

    1. CrossEncoder rerank_score
    2. Hybrid relevance_score

    CrossEncoder scores are treated primarily as ranking signals.

    By default, the CrossEncoder relative score filter is disabled
    because raw CrossEncoder scores are not normalized percentages.
    """

    if not results:
        return []

    # --------------------------------------------------------
    # 1. CrossEncoder results
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
        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        best_score = float(
            reranked_results[0]["rerank_score"]
        )

        logger.info(
            "rag_rerank best_score=%.4f ratio=%.4f "
            "candidates=%d",
            best_score,
            RERANK_RELATIVE_SCORE_RATIO,
            len(reranked_results),
        )

        # ----------------------------------------------------
        # Disabled by configuration.
        # This is the preferred behavior for raw CrossEncoder
        # scores.
        # ----------------------------------------------------

        if RERANK_RELATIVE_SCORE_RATIO <= 0:
            return reranked_results

        # ----------------------------------------------------
        # If the best score is zero or negative, do not apply
        # a multiplicative threshold.
        # ----------------------------------------------------

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
    # 2. Hybrid retrieval fallback
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

    return kept or [
        max(
            scored_results,
            key=lambda result: result["relevance_score"],
        )
    ]


# ============================================================
# Document-aware selection
# ============================================================


def _select_document_aware_results(
    results: list[dict],
    limit: int,
) -> list[dict]:
    """
    Select final context chunks while preserving document diversity.

    The function does not contain any domain-specific knowledge.

    Strategy:

    1. Results are already ordered by their strongest available
       relevance signal.
    2. Group chunks by document identity.
    3. Rank documents by their strongest chunk.
    4. Consider documents in that order.
    5. Allow multiple chunks from the same document.
    6. Do not allow one document to consume the entire context.
    7. Stop at the configured final chunk limit.

    This allows several useful chunks from one document to survive
    while still protecting the context from a single-document flood.
    """

    if not results:
        return []

    if limit <= 0:
        return []

    # --------------------------------------------------------
    # Preserve existing reranker ordering.
    # --------------------------------------------------------

    ordered_results = list(results)

    # --------------------------------------------------------
    # Group by document.
    # --------------------------------------------------------

    groups: dict[str, list[dict]] = defaultdict(list)

    for result in ordered_results:
        key = _document_key(result)
        groups[key].append(result)

    # --------------------------------------------------------
    # Sort each document's chunks by the same final ranking.
    # --------------------------------------------------------

    for key in groups:
        groups[key].sort(
            key=lambda result: (
                _rerank_score(result)
                if _rerank_score(result) is not None
                else (
                    _relevance_score(result)
                    if _relevance_score(result) is not None
                    else float("-inf")
                )
            ),
            reverse=True,
        )

    # --------------------------------------------------------
    # Rank documents by their strongest chunk.
    # --------------------------------------------------------

    ranked_documents = sorted(
        groups.items(),
        key=lambda item: (
            _rerank_score(item[1][0])
            if _rerank_score(item[1][0]) is not None
            else (
                _relevance_score(item[1][0])
                if _relevance_score(item[1][0]) is not None
                else float("-inf")
            )
        ),
        reverse=True,
    )

    selected: list[dict] = []

    # --------------------------------------------------------
    # First pass:
    #
    # Take the strongest chunk from each of the strongest
    # documents.
    #
    # This preserves document diversity.
    # --------------------------------------------------------

    for document_key, chunks in ranked_documents:
        if len(selected) >= limit:
            break

        if len(selected) >= MAX_DOCUMENTS:
            break

        if not chunks:
            continue

        selected.append(chunks[0])

    # --------------------------------------------------------
    # Second pass:
    #
    # Add additional chunks from already-selected documents.
    #
    # This is the important part for pages/documents that contain
    # multiple independently useful chunks.
    # --------------------------------------------------------

    selected_document_keys = [
        _document_key(result)
        for result in selected
    ]

    selected_document_keys = list(
        dict.fromkeys(selected_document_keys)
    )

    for document_key in selected_document_keys:
        if len(selected) >= limit:
            break

        chunks = groups.get(document_key, [])

        if not chunks:
            continue

        selected_from_document = sum(
            1
            for result in selected
            if _document_key(result) == document_key
        )

        additional_chunks = chunks[
            selected_from_document:
            MAX_CHUNKS_PER_DOCUMENT
        ]

        for chunk in additional_chunks:
            if len(selected) >= limit:
                break

            selected.append(chunk)

    # --------------------------------------------------------
    # If the first/second passes did not fill the requested
    # number, use remaining globally ranked candidates.
    # --------------------------------------------------------

    selected_ids = {
        id(result)
        for result in selected
    }

    for result in ordered_results:
        if len(selected) >= limit:
            break

        if id(result) in selected_ids:
            continue

        document_key = _document_key(result)

        count_for_document = sum(
            1
            for item in selected
            if _document_key(item) == document_key
        )

        if count_for_document >= MAX_CHUNKS_PER_DOCUMENT:
            continue

        selected.append(result)
        selected_ids.add(id(result))

    # --------------------------------------------------------
    # Final ordering:
    #
    # Restore score order so the strongest evidence is presented
    # first to the model.
    # --------------------------------------------------------

    selected.sort(
        key=lambda result: (
            _rerank_score(result)
            if _rerank_score(result) is not None
            else (
                _relevance_score(result)
                if _relevance_score(result) is not None
                else float("-inf")
            )
        ),
        reverse=True,
    )

    logger.info(
        "rag_document_selection input=%d output=%d "
        "documents=%d max_per_document=%d",
        len(results),
        len(selected),
        len(
            {
                _document_key(result)
                for result in selected
            }
        ),
        MAX_CHUNKS_PER_DOCUMENT,
    )

    return selected


# ============================================================
# Context construction
# ============================================================


def build_context(
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """
    Retrieve knowledge and construct the final context for the
    AI layer.

    The retrieval pipeline is responsible for semantic retrieval
    and reranking.

    This layer is responsible for:

    - relevance filtering
    - document-aware chunk selection
    - context size control
    - duplicate removal
    - source deduplication
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

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    retrieval_limit = max(
        int(limit),
        RAG_CANDIDATE_POOL,
    )

    results = search_knowledge(
        query=query,
        limit=retrieval_limit,
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
    # Relevance filtering
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

    # retriever.py already performs reranking, document expansion, and
    # document-aware final selection. Preserve that evidence order here;
    # a second selection pass could discard expanded chunks.
    selected_results = results[:limit]

    if not selected_results:
        logger.info(
            "rag_no_document_aware_results "
            "filtered=%d query=%r",
            len(results),
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

    for result in selected_results:

        title = (
            result.get("title")
            or "Untitled"
        )

        url = (
            result.get("url")
            or ""
        )

        content = (
            result.get("content")
            or ""
        ).strip()

        if not content:
            continue

        # ----------------------------------------------------
        # Duplicate content protection.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Context size protection.
        # ----------------------------------------------------

        separator_size = (
            len(SEPARATOR)
            if context_parts
            else 0
        )

        remaining_chars = (
            MAX_CONTEXT_CHARS
            - used_chars
            - separator_size
        )

        if remaining_chars <= 0:
            logger.info(
                "rag_context_limit_reached "
                "used_chars=%d next_block=%d max=%d",
                used_chars,
                len(block),
                MAX_CONTEXT_CHARS,
            )
            break

        if len(block) > remaining_chars:
            block = block[:remaining_chars].rstrip()

            logger.info(
                "rag_context_block_truncated "
                "used_chars=%d final_block=%d max=%d",
                used_chars,
                len(block),
                MAX_CONTEXT_CHARS,
            )

        seen_content.add(fingerprint)

        context_parts.append(block)

        used_chars += (
            len(block)
            + separator_size
        )

        # ----------------------------------------------------
        # Sources are derived ONLY from chunks actually used
        # in the final context.
        # ----------------------------------------------------

        if (
            url
            and url not in sources_by_url
        ):
            sources_by_url[url] = {
                "title": title,
                "url": url,
            }

    if not context_parts:
        return _empty()

    context = SEPARATOR.join(
        context_parts
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    best_rerank_score = max(
        (
            result.get("rerank_score")
            for result in selected_results
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
            for result in selected_results
            if isinstance(
                result.get("relevance_score"),
                (int, float),
            )
        ),
        default=None,
    )

    selected_document_keys = {
        _document_key(result)
        for result in selected_results
    }

    logger.info(
        "rag_result "
        "retrieved=%d "
        "filtered=%d "
        "selected=%d "
        "documents=%d "
        "context_chars=%d "
        "rerank_score=%s "
        "relevance_score=%s "
        "retrieval=%.3fs "
        "total=%.3fs "
        "query=%r",
        original_result_count,
        len(results),
        len(context_parts),
        len(selected_document_keys),
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
        time.perf_counter()
        - total_start,
        query[:120],
    )

    return {
        "has_context": True,
        "context": context,
        "sources": list(
            sources_by_url.values()
        ),
    }