"""
Hybrid knowledge retriever (pgvector + PostgreSQL full-text + heading boost).

Design goals
------------
* Same scoring formula as before (vector / AND-text / OR-text weights,
  heading fallback boost), so existing retrieval regression tests keep
  their meaning.
* One database round trip.
* Uses stored content_tsv + GIN index and HNSW vector index when available.
* Falls back gracefully when embedding/reranking/index features fail.
* CrossEncoder can evaluate a larger candidate pool without forcing all
  candidates into the final context.
* Final result selection is document-aware and generic.
* Everything tunable is an environment variable.
* No company-specific logic.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv

from .db import get_database_connection
from .local_embeddings import create_local_embedding
from .reranker import rerank_results


load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _float_setting(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def _int_setting(
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        value = int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError, OverflowError):
        return default

    if minimum is not None:
        value = max(minimum, value)

    if maximum is not None:
        value = min(maximum, value)

    return value


def _bool_setting(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class _Settings:
    # Requested number of final results. The retriever never returns more
    # than the caller requested, even if reranking/selection has a larger pool.
    result_limit: int

    max_query_chars: int

    max_distance: float

    candidate_limit: int

    vector_weight: float
    text_weight: float
    or_text_weight: float

    heading_weight: float
    heading_fallback_threshold: float

    rerank_enabled: bool
    rerank_top_k: int
    rerank_max_candidates: int

    rerank_skip_min_score: float
    rerank_skip_margin: float

    # Generic document-aware selection.
    max_chunks_per_document: int
    max_documents: int

    # Recover additional chunks from documents that already have strong
    # evidence, even when those chunks fall outside the reranker window.
    document_expansion_enabled: bool
    document_expansion_candidates: int


def _load_settings(limit: int) -> _Settings:
    rerank_top_k = _int_setting(
        "RERANKER_TOP_K",
        max(1, limit),
        minimum=1,
        maximum=200,
    )

    return _Settings(
        result_limit=max(1, limit),

        max_query_chars=_int_setting(
            "RETRIEVAL_MAX_QUERY_CHARS",
            1000,
            minimum=20,
        ),

        max_distance=_float_setting(
            "RETRIEVAL_MAX_DISTANCE",
            0.80,
            minimum=0.0,
        ),

        candidate_limit=_int_setting(
            "RERANKER_CANDIDATE_LIMIT",
            max(limit * 5, 10),
            minimum=1,
            maximum=200,
        ),

        vector_weight=_float_setting(
            "RETRIEVAL_VECTOR_WEIGHT",
            0.70,
            minimum=0.0,
        ),

        text_weight=_float_setting(
            "RETRIEVAL_TEXT_WEIGHT",
            0.30,
            minimum=0.0,
        ),

        or_text_weight=_float_setting(
            "RETRIEVAL_OR_TEXT_WEIGHT",
            0.15,
            minimum=0.0,
        ),

        heading_weight=_float_setting(
            "RETRIEVAL_HEADING_WEIGHT",
            0.10,
            minimum=0.0,
        ),

        heading_fallback_threshold=_float_setting(
            "RETRIEVAL_HEADING_FALLBACK_THRESHOLD",
            0.35,
            minimum=0.0,
        ),

        rerank_enabled=_bool_setting(
            "RERANKER_ENABLED",
            True,
        ),

        rerank_top_k=rerank_top_k,

        rerank_max_candidates=_int_setting(
            "RERANKER_MAX_CANDIDATES",
            30,
            minimum=1,
            maximum=200,
        ),

        rerank_skip_min_score=_float_setting(
            "RERANKER_SKIP_MIN_SCORE",
            0.0,
            minimum=0.0,
        ),

        rerank_skip_margin=_float_setting(
            "RERANKER_SKIP_MARGIN",
            0.0,
            minimum=0.0,
        ),

        max_chunks_per_document=_int_setting(
            "RAG_MAX_CHUNKS_PER_DOCUMENT",
            3,
            minimum=1,
            maximum=20,
        ),

        max_documents=_int_setting(
            "RAG_MAX_DOCUMENTS",
            4,
            minimum=1,
            maximum=50,
        ),

        document_expansion_enabled=_bool_setting(
            "RAG_DOCUMENT_EXPANSION_ENABLED",
            True,
        ),

        document_expansion_candidates=_int_setting(
            "RAG_DOCUMENT_EXPANSION_CANDIDATES",
            50,
            minimum=1,
            maximum=200,
        ),
    )


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")
_HEADING_RE = re.compile(r"^#{1,6}\s+")

_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were
    be been being am do does did done have has had having it its this that
    these those i you he she we they me my your our their him her them us
    what who whom whose which when where why how can could should would will
    may might must shall about into over under than then there here so not no
    yes tell please give show
    """.split()
)


def _normalize_query(
    query: str,
    max_chars: int,
) -> str:
    return " ".join(query.split())[:max_chars].strip()


def _meaningful_tokens(
    text: str,
) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())

    kept = [
        token
        for token in tokens
        if len(token) >= 2
        and token not in _STOPWORDS
    ]

    return kept or tokens


def _build_or_query(
    query: str,
) -> str:
    seen: dict[str, None] = {}

    for token in _meaningful_tokens(query)[:32]:
        seen.setdefault(token, None)

    return (
        " or ".join(seen)
        if seen
        else query
    )


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


_TSV_STORED = "c.content_tsv"
_TSV_ON_THE_FLY = (
    "to_tsvector('english', COALESCE(c.content, ''))"
)

def _build_sql(
    use_vector: bool,
    tsv: str,
) -> str:
    parts: list[str] = []

    if use_vector:
        parts.append(
            """
            SELECT
                v.id,
                v.document_id,
                v.content,
                v.vector_distance,
                0.0::float8 AS text_score,
                0.0::float8 AS or_text_score
            FROM (
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.embedding <=> %(embedding)s::vector
                        AS vector_distance
                FROM chunks c
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %(embedding)s::vector
                LIMIT %(candidate_limit)s
            ) v
            """
        )

    parts.append(
        f"""
        SELECT
            t.id,
            t.document_id,
            t.content,
            NULL::float8 AS vector_distance,
            t.score AS text_score,
            0.0::float8 AS or_text_score
        FROM (
            SELECT
                c.id,
                c.document_id,
                c.content,
                ts_rank_cd(
                    {tsv},
                    q.and_q
                ) AS score
            FROM chunks c
            CROSS JOIN q
            WHERE {tsv} @@ q.and_q
            ORDER BY score DESC
            LIMIT %(candidate_limit)s
        ) t
        """
    )

    parts.append(
        f"""
        SELECT
            o.id,
            o.document_id,
            o.content,
            NULL::float8 AS vector_distance,
            0.0::float8 AS text_score,
            o.score AS or_text_score
        FROM (
            SELECT
                c.id,
                c.document_id,
                c.content,
                ts_rank_cd(
                    {tsv},
                    q.or_q
                ) AS score
            FROM chunks c
            CROSS JOIN q
            WHERE {tsv} @@ q.or_q
            ORDER BY score DESC
            LIMIT %(candidate_limit)s
        ) o
        """
    )

    union = "\nUNION ALL\n".join(parts)

    return f"""
        WITH q AS (
            SELECT
                plainto_tsquery(
                    'english',
                    %(query)s
                ) AS and_q,

                websearch_to_tsquery(
                    'english',
                    %(or_query)s
                ) AS or_q
        )

        SELECT
            p.id,
            p.document_id,
            p.content,
            d.title,
            d.url,
            p.vector_distance,
            p.text_score,
            p.or_text_score

        FROM (
            {union}
        ) p

        JOIN documents d
            ON d.id = p.document_id
    """


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _merge_rows(rows) -> list[dict]:
    merged: dict = {}

    for (
        chunk_id,
        document_id,
        content,
        title,
        url,
        distance,
        text_score,
        or_score,
    ) in rows:

        distance = (
            float(distance)
            if distance is not None
            else None
        )

        text_score = float(
            text_score or 0.0
        )

        or_score = float(
            or_score or 0.0
        )

        entry = merged.get(chunk_id)

        if entry is None:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": content,
                "title": title,
                "url": url,
                "distance": distance,
                "text_score": text_score,
                "or_text_score": or_score,
            }

            continue

        if (
            distance is not None
            and (
                entry["distance"] is None
                or distance < entry["distance"]
            )
        ):
            entry["distance"] = distance

        entry["text_score"] = max(
            entry["text_score"],
            text_score,
        )

        entry["or_text_score"] = max(
            entry["or_text_score"],
            or_score,
        )

    return list(merged.values())


def _heading_overlap(
    content: str | None,
    title: str | None,
    query_tokens: set[str],
) -> float:

    if not query_tokens:
        return 0.0

    # --------------------------------------------------------
    # Document title gets priority.
    # --------------------------------------------------------

    if title:
        title_tokens = set(
            _TOKEN_RE.findall(
                title.lower()
            )
        )

        if query_tokens & title_tokens:
            return 1.0

    # --------------------------------------------------------
    # Then inspect the first markdown heading.
    # --------------------------------------------------------

    if not content:
        return 0.0

    first_line = content.split(
        "\n",
        1,
    )[0]

    if not _HEADING_RE.match(first_line):
        return 0.0

    heading = _HEADING_RE.sub(
        "",
        first_line,
    ).lower()

    heading_tokens = set(
        _TOKEN_RE.findall(heading)
    )

    return (
        1.0
        if query_tokens & heading_tokens
        else 0.0
    )


def _score_candidates(
    candidates: list[dict],
    query: str,
    s: _Settings,
) -> list[dict]:

    query_tokens = set(
        _meaningful_tokens(query)
    )

    scored: list[dict] = []

    for item in candidates:
        distance = item["distance"]
        text_score = item["text_score"]
        or_score = item["or_text_score"]

        if not (
            distance is None
            or distance <= s.max_distance
            or text_score > 0
            or or_score > 0
        ):
            continue

        if (
            distance is None
            or s.max_distance <= 0
        ):
            vector_score = 0.0
        else:
            vector_score = max(
                0.0,
                1.0
                - distance
                / s.max_distance,
            )

        base = (
            vector_score
            * s.vector_weight
            + min(1.0, text_score)
            * s.text_weight
            + min(1.0, or_score)
            * s.or_text_weight
        )

        heading_score = _heading_overlap(
            item["content"],
            item.get("title"),
            query_tokens,
        )

        relevance = base

        if (
            base
            < s.heading_fallback_threshold
        ):
            relevance += (
                heading_score
                * s.heading_weight
            )

        item["heading_score"] = (
            heading_score
        )

        item["base_relevance_score"] = (
            base
        )

        item["relevance_score"] = (
            relevance
        )

        scored.append(item)

    scored.sort(
        key=lambda result: (
            -result["relevance_score"],
            -result["heading_score"],
            -result["or_text_score"],
            -result["text_score"],
            (
                result["distance"]
                if result["distance"]
                is not None
                else math.inf
            ),
            len(result["url"] or ""),
            result["url"] or "",
        )
    )

    return scored


def _dedupe_by_content(
    results: list[dict],
) -> list[dict]:

    seen: set[str] = set()
    unique: list[dict] = []

    for item in results:
        normalized = " ".join(
            (
                item["content"]
                or ""
            ).lower().split()
        )

        key = hashlib.sha1(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def _rerank(
    query: str,
    results: list[dict],
    s: _Settings,
) -> tuple[list[dict], bool]:
    """
    Rerank the strongest candidates and recover additional evidence from
    documents that are already strongly represented in the result set.

    The important distinction is between *chunk relevance* and *document
    relevance*. A single useful document can contain several complementary
    chunks. A hard reranker window must not permanently discard those chunks
    merely because their individual hybrid rank is slightly lower.

    The expansion is completely generic. It uses only document_id/url/title,
    existing hybrid relevance_score, and configurable limits. No domain
    names, page names, entities, or query-specific rules are used.
    """
    if not results:
        return [], False

    # ------------------------------------------------------------
    # 1. Normal reranker candidate pool
    # ------------------------------------------------------------
    pool = results[: s.rerank_max_candidates]

    if not pool:
        return [], False

    reranked = False
    working = list(pool)

    if s.rerank_enabled:
        if (
            s.rerank_skip_margin > 0
            and len(pool) > 1
        ):
            top = pool[0].get("relevance_score", 0.0)
            second = pool[1].get("relevance_score", 0.0)

            if (
                top >= s.rerank_skip_min_score
                and (top - second) >= s.rerank_skip_margin
            ):
                working = pool
            else:
                try:
                    candidate_results = rerank_results(
                        query=query,
                        results=pool,
                        limit=len(pool),
                    )
                    if candidate_results:
                        working = list(candidate_results)
                        reranked = True
                except Exception:
                    logger.exception(
                        "Reranker failed; falling back to hybrid ranking"
                    )
        else:
            try:
                candidate_results = rerank_results(
                    query=query,
                    results=pool,
                    limit=len(pool),
                )
                if candidate_results:
                    working = list(candidate_results)
                    reranked = True
            except Exception:
                logger.exception(
                    "Reranker failed; falling back to hybrid ranking"
                )

    # ------------------------------------------------------------
    # 2. Document expansion
    # ------------------------------------------------------------
    #
    # Find the strongest documents from the reranked pool. Then recover
    # additional chunks belonging to those documents from the complete
    # hybrid-ranked result set. This specifically prevents useful chunks
    # just outside RERANKER_MAX_CANDIDATES from being lost.
    # ------------------------------------------------------------
    if (
        s.document_expansion_enabled
        and working
        and s.max_chunks_per_document > 1
    ):
        strong_document_keys: list[str] = []

        for result in working:
            key = _document_key(result)
            if key not in strong_document_keys:
                strong_document_keys.append(key)

            if len(strong_document_keys) >= s.max_documents:
                break

        all_by_document: dict[str, list[dict]] = defaultdict(list)

        # Only inspect a configurable prefix of the already-ranked hybrid
        # results. This bounds work while allowing expansion beyond the
        # CrossEncoder window.
        expansion_source = results[: s.document_expansion_candidates]

        for result in expansion_source:
            all_by_document[_document_key(result)].append(result)

        for document_results in all_by_document.values():
            document_results.sort(
                key=lambda result: (
                    _relevance_score_for_expansion(result),
                    result.get("heading_score", 0.0),
                ),
                reverse=True,
            )

        existing_ids = {
            result.get("chunk_id")
            for result in working
            if result.get("chunk_id") is not None
        }

        expanded: list[dict] = []

        for document_key in strong_document_keys:
            document_results = all_by_document.get(
                document_key,
                [],
            )

            # Keep the existing maximum per document as the hard upper
            # bound. The reranked result already occupies one slot, so
            # recover only the remaining available slots.
            current_count = sum(
                1
                for result in working
                if _document_key(result) == document_key
            )

            remaining_slots = max(
                0,
                s.max_chunks_per_document - current_count,
            )

            if remaining_slots <= 0:
                continue

            for chunk in document_results:
                chunk_id = chunk.get("chunk_id")

                if (
                    chunk_id is not None
                    and chunk_id in existing_ids
                ):
                    continue

                expanded.append(chunk)

                if chunk_id is not None:
                    existing_ids.add(chunk_id)

                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

        if expanded:
            logger.info(
                "retriever document expansion | base=%d expanded=%d "
                "documents=%d source=%d",
                len(working),
                len(expanded),
                len(strong_document_keys),
                len(expansion_source),
            )
            working.extend(expanded)

    # ------------------------------------------------------------
    # 3. Final generic document-aware selection
    # ------------------------------------------------------------
    selected = _select_document_aware_results(
        working,
        s,
    )

    return selected, reranked


def _relevance_score_for_expansion(
    result: dict,
) -> float:
    """Return the existing hybrid relevance used for document expansion."""
    value = result.get("relevance_score")

    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)

    return float("-inf")


# ---------------------------------------------------------------------------
# Document-aware result selection
# ---------------------------------------------------------------------------


def _result_score(
    result: dict,
) -> float:

    rerank_score = result.get(
        "rerank_score"
    )

    if isinstance(
        rerank_score,
        (int, float),
    ):
        return float(
            rerank_score
        )

    relevance_score = result.get(
        "relevance_score"
    )

    if isinstance(
        relevance_score,
        (int, float),
    ):
        return float(
            relevance_score
        )

    return float("-inf")


def _document_key(
    result: dict,
) -> str:

    document_id = result.get(
        "document_id"
    )

    if document_id is not None:
        return f"id:{document_id}"

    url = str(
        result.get("url")
        or ""
    ).strip()

    if url:
        return f"url:{url}"

    title = str(
        result.get("title")
        or ""
    ).strip()

    if title:
        return (
            f"title:{title.lower()}"
        )

    return (
        "anonymous:"
        + str(id(result))
    )


def _select_document_aware_results(
    results: list[dict],
    s: _Settings,
) -> list[dict]:
    """
    Select final evidence using generic document-level aggregation.

    A document is ranked from the strength of its best available chunks.
    Selection then gives the strongest document access to its strongest
    evidence before moving to weaker documents. This prevents unrelated
    documents from consuming context slots simply because of round-robin
    ordering, while max_chunks_per_document and max_documents still protect
    context diversity.
    """
    if not results:
        return []

    grouped: dict[str, list[dict]] = defaultdict(list)

    for result in results:
        grouped[_document_key(result)].append(result)

    for document_results in grouped.values():
        document_results.sort(
            key=_result_score,
            reverse=True,
        )

    document_scores: dict[str, float] = {}
    weights = (1.0, 0.50, 0.25)

    for document_key, document_results in grouped.items():
        scores = [
            _result_score(result)
            for result in document_results[: s.max_chunks_per_document]
        ]
        finite_scores = [
            score
            for score in scores
            if math.isfinite(score)
        ]

        document_scores[document_key] = (
            sum(
                score * weights[index]
                for index, score in enumerate(finite_scores)
            )
            if finite_scores
            else float("-inf")
        )

    ranked_documents = sorted(
        (
            document_key
            for document_key in grouped
            if math.isfinite(document_scores[document_key])
        ),
        key=lambda document_key: (
            document_scores[document_key],
            _result_score(grouped[document_key][0]),
        ),
        reverse=True,
    )[: s.max_documents]

    selected: list[dict] = []
    selected_counts: defaultdict[str, int] = defaultdict(int)

    target_count = min(
        s.result_limit,
        s.rerank_top_k,
    )

    # First select the strongest available evidence globally. This avoids
    # arbitrary round-robin allocation between documents. A document can
    # contribute multiple chunks when those chunks are actually among the
    # strongest available evidence, subject to max_chunks_per_document.
    while len(selected) < target_count:
        best_candidate = None
        best_document_key = None
        best_score = float("-inf")

        for document_key in ranked_documents:
            document_results = grouped[document_key]
            current_count = selected_counts[document_key]

            if current_count >= min(
                s.max_chunks_per_document,
                len(document_results),
            ):
                continue

            candidate = document_results[current_count]
            candidate_score = _result_score(candidate)

            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate
                best_document_key = document_key

        if best_candidate is None:
            break

        selected.append(best_candidate)
        selected_counts[best_document_key] += 1

    selected.sort(
        key=_result_score,
        reverse=True,
    )

    logger.info(
        "retriever document-aware selection | input=%d output=%d "
        "documents=%d max_per_document=%d | selected=%s",
        len(results),
        len(selected),
        len({ _document_key(item) for item in selected }),
        s.max_chunks_per_document,
        [
            {
                "document_id": grouped[key][0].get("document_id"),
                "title": grouped[key][0].get("title"),
                "document_score": document_scores[key],
                "selected_chunks": selected_counts[key],
            }
            for key in ranked_documents
            if selected_counts[key]
        ],
    )

    return selected



# ---------------------------------------------------------------------------
# Database retrieval with graceful fallbacks
# ---------------------------------------------------------------------------


def _execute_retrieval(
    conn,
    query: str,
    or_query: str,
    params: dict,
) -> tuple[list, float]:
    """
    Execute the hybrid retrieval query with progressively safer fallbacks.

    Normal path:
        vector + stored content_tsv

    Fallbacks:
        vector + generated tsv
        text-only + stored content_tsv
        text-only + generated tsv

    The normal path remains one database execution. Extra attempts happen only
    when an optional database feature is unavailable or fails.
    """
    sql_start = time.perf_counter()
    errors: list[tuple[str, Exception]] = []

    attempts = [
        ("vector+stored_tsv", True, True),
        ("vector+generated_tsv", True, False),
        ("text+stored_tsv", False, True),
        ("text+generated_tsv", False, False),
    ]

    # Do not repeat equivalent text-only attempts after embedding is absent.
    if params.get("embedding") is None:
        attempts = [
            ("text+stored_tsv", False, True),
            ("text+generated_tsv", False, False),
        ]

    for label, use_vector, use_stored_tsv in attempts:
        try:
            tsv = _TSV_STORED if use_stored_tsv else _TSV_ON_THE_FLY

            with conn.cursor() as cursor:
                cursor.execute(
                    _build_sql(
                        use_vector=use_vector,
                        tsv=tsv,
                    ),
                    params,
                )
                rows = cursor.fetchall()

            return rows, time.perf_counter() - sql_start

        except Exception as exc:
            errors.append((label, exc))
            logger.warning(
                "retriever database attempt failed (%s): %s",
                label,
                exc,
            )

            # PostgreSQL marks the transaction as failed after a SQL error.
            # Roll back before trying the next fallback query.
            try:
                conn.rollback()
            except Exception:
                logger.exception(
                    "retriever database rollback failed after %s",
                    label,
                )

    # All supported retrieval paths failed. Do not take down the application;
    # return an empty result set so the caller can handle "no knowledge found".
    logger.error(
        "All database retrieval fallbacks failed. attempts=%s",
        [label for label, _ in errors],
        exc_info=(
            (
                type(errors[-1][1]),
                errors[-1][1],
                errors[-1][1].__traceback__,
            )
            if errors
            else None
        ),
    )

    return [], time.perf_counter() - sql_start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_knowledge(
    query: str,
    limit: int = 5,
) -> list[dict]:

    total_start = time.perf_counter()

    if not query or not query.strip():
        return []

    try:
        limit = max(
            1,
            int(limit),
        )
    except (
        TypeError,
        ValueError,
    ):
        limit = 5

    s = _load_settings(limit)

    query = _normalize_query(
        query,
        s.max_query_chars,
    )

    if not query:
        return []

    # --------------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------------

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = None

    try:
        query_embedding = (
            create_local_embedding(
                query
            )
        )

    except Exception:
        logger.exception(
            "Query embedding failed; "
            "continuing with keyword-only retrieval"
        )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # --------------------------------------------------------
    # 2. PostgreSQL
    # --------------------------------------------------------

    params: dict = {
        "query": query,
        "or_query": _build_or_query(
            query
        ),
        "candidate_limit": s.candidate_limit,
    }

    use_vector = (
        query_embedding is not None
    )

    if use_vector:
        params["embedding"] = (
            query_embedding
        )

    connection_start = (
        time.perf_counter()
    )

    try:
        with get_database_connection() as conn:
            connection_time = (
                time.perf_counter()
                - connection_start
            )

            rows, sql_time = _execute_retrieval(
                conn=conn,
                query=query,
                or_query=params["or_query"],
                params=params,
            )
    except Exception:
        # Connection acquisition itself can fail. Retrieval is an optional
        # knowledge layer and should never crash the request path.
        connection_time = (
            time.perf_counter()
            - connection_start
        )
        sql_time = 0.0
        rows = []
        logger.exception(
            "Knowledge database connection failed; returning no retrieval results"
        )

    # --------------------------------------------------------
    # 3. Hybrid scoring
    # --------------------------------------------------------

    processing_start = (
        time.perf_counter()
    )

    candidates = _merge_rows(
        rows
    )

    ranked = _dedupe_by_content(
        _score_candidates(
            candidates,
            query,
            s,
        )
    )

    processing_time = (
        time.perf_counter()
        - processing_start
    )

    # --------------------------------------------------------
    # 4. CrossEncoder + document-aware selection
    # --------------------------------------------------------

    rerank_start = (
        time.perf_counter()
    )

    results, reranked = _rerank(
        query,
        ranked,
        s,
    )

    rerank_time = (
        time.perf_counter()
        - rerank_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------------

    logger.info(
        "retriever timing | "
        "embed=%.3fs connect=%.3fs sql=%.3fs "
        "python=%.3fs rerank=%.3fs total=%.3fs | "
        "rows=%d unique=%d "
        "rerank_pool=%d final=%d "
        "reranked=%s vector=%s",
        embedding_time,
        connection_time,
        sql_time,
        processing_time,
        rerank_time,
        total_time,
        len(rows),
        len(ranked),
        min(
            len(ranked),
            s.rerank_max_candidates,
        ),
        len(results),
        reranked,
        use_vector,
    )

    return results[: s.result_limit]
