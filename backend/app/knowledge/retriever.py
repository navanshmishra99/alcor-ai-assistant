"""Hybrid knowledge retriever (pgvector + PostgreSQL full-text + heading boost).

Design goals
------------
* Same scoring formula as before (vector / AND-text / OR-text weights, heading
  fallback boost), so existing retrieval regression tests keep their meaning.
* One database round trip. Vector search, AND full-text search and OR full-text
  search run as three small candidate queries in a single statement; scoring,
  merging and de-duplication happen in Python on a tiny candidate pool.
* Uses a stored ``content_tsv`` column + GIN index and an HNSW vector index when
  they exist (see retrieval_indexes.sql); falls back to on-the-fly
  ``to_tsvector`` when they do not, so it works on any database state.
* Degrades instead of failing: embedding failure -> keyword-only retrieval,
  reranker failure -> hybrid order.
* Everything tunable is an environment variable. No company-specific logic.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from .db import get_database_connection
from .local_embeddings import create_local_embedding
from .reranker import rerank_results


load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (read per call, so .env / environment changes apply without code
# changes; values are validated and clamped so a bad value never breaks search)
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


def _bool_setting(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _Settings:
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


def _load_settings(limit: int) -> _Settings:
    rerank_top_k = _int_setting("RERANKER_TOP_K", limit, minimum=1)

    return _Settings(
        max_query_chars=_int_setting(
            "RETRIEVAL_MAX_QUERY_CHARS", 1000, minimum=20
        ),
        max_distance=_float_setting(
            "RETRIEVAL_MAX_DISTANCE", 0.80, minimum=0.0
        ),
        candidate_limit=_int_setting(
            "RERANKER_CANDIDATE_LIMIT",
            max(limit * 5, 10),
            minimum=1,
            maximum=200,
        ),
        vector_weight=_float_setting(
            "RETRIEVAL_VECTOR_WEIGHT", 0.70, minimum=0.0
        ),
        text_weight=_float_setting(
            "RETRIEVAL_TEXT_WEIGHT", 0.30, minimum=0.0
        ),
        or_text_weight=_float_setting(
            "RETRIEVAL_OR_TEXT_WEIGHT", 0.15, minimum=0.0
        ),
        heading_weight=_float_setting(
            "RETRIEVAL_HEADING_WEIGHT", 0.10, minimum=0.0
        ),
        heading_fallback_threshold=_float_setting(
            "RETRIEVAL_HEADING_FALLBACK_THRESHOLD", 0.35, minimum=0.0
        ),
        rerank_enabled=_bool_setting("RERANKER_ENABLED", True),
        rerank_top_k=rerank_top_k,
        # Cross-encoder cost is linear in the number of candidates, so only the
        # best hybrid-ranked candidates are reranked.
        rerank_max_candidates=_int_setting(
            "RERANKER_MAX_CANDIDATES",
            max(limit * 2, 8),
            minimum=1,
            maximum=200,
        ),
        # Optional confidence gate: skip the reranker when the best hybrid hit
        # is clearly ahead. Disabled while RERANKER_SKIP_MARGIN is 0. Tune it
        # against your retrieval regression set before enabling.
        rerank_skip_min_score=_float_setting(
            "RERANKER_SKIP_MIN_SCORE", 0.0, minimum=0.0
        ),
        rerank_skip_margin=_float_setting(
            "RERANKER_SKIP_MARGIN", 0.0, minimum=0.0
        ),
    )


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")
_HEADING_RE = re.compile(r"^#{1,6}\s+")

# Small generic English stop list, used only for heading overlap and the OR
# query. Not domain specific.
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


def _normalize_query(query: str, max_chars: int) -> str:
    return " ".join(query.split())[:max_chars].strip()


def _meaningful_tokens(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    kept = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
    return kept or tokens


def _build_or_query(query: str) -> str:
    seen: dict[str, None] = {}
    for token in _meaningful_tokens(query)[:32]:
        seen.setdefault(token, None)
    # websearch_to_tsquery treats the word "or" as the OR operator.
    return " or ".join(seen) if seen else query


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_TSV_STORED = "c.content_tsv"
_TSV_ON_THE_FLY = "to_tsvector('english', COALESCE(c.content, ''))"

_TSV_COLUMN_AVAILABLE: bool | None = None
_TSV_CHECKED_AT = 0.0
_TSV_RECHECK_SECONDS = 60.0


def _tsv_expression(cursor) -> str:
    """Use the stored tsvector column if present, else compute on the fly.

    A positive result is cached for the process lifetime; a negative result is
    re-checked periodically so running the index migration needs no restart.
    """
    global _TSV_COLUMN_AVAILABLE, _TSV_CHECKED_AT

    now = time.monotonic()
    stale_negative = (
        _TSV_COLUMN_AVAILABLE is False
        and now - _TSV_CHECKED_AT > _TSV_RECHECK_SECONDS
    )

    if _TSV_COLUMN_AVAILABLE is None or stale_negative:
        cursor.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'chunks'
              AND column_name = 'content_tsv'
            LIMIT 1
            """
        )
        _TSV_COLUMN_AVAILABLE = cursor.fetchone() is not None
        _TSV_CHECKED_AT = now

    return _TSV_STORED if _TSV_COLUMN_AVAILABLE else _TSV_ON_THE_FLY


def _build_sql(use_vector: bool, tsv: str) -> str:
    """Build the single candidate-retrieval statement.

    ``tsv`` is one of two fixed internal expressions (never user input). All
    user-controlled values are bound parameters.
    """
    parts: list[str] = []

    if use_vector:
        # ORDER BY ... LIMIT directly on chunks.embedding so an HNSW/IVFFlat
        # index can be used; documents are joined afterwards.
        parts.append(
            """
            SELECT v.id, v.document_id, v.content,
                   v.vector_distance,
                   0.0::float8 AS text_score,
                   0.0::float8 AS or_text_score
            FROM (
                SELECT c.id, c.document_id, c.content,
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
        SELECT t.id, t.document_id, t.content,
               NULL::float8 AS vector_distance,
               t.score AS text_score,
               0.0::float8 AS or_text_score
        FROM (
            SELECT c.id, c.document_id, c.content,
                   ts_rank_cd({tsv}, q.and_q) AS score
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
        SELECT o.id, o.document_id, o.content,
               NULL::float8 AS vector_distance,
               0.0::float8 AS text_score,
               o.score AS or_text_score
        FROM (
            SELECT c.id, c.document_id, c.content,
                   ts_rank_cd({tsv}, q.or_q) AS score
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
                plainto_tsquery('english', %(query)s) AS and_q,
                websearch_to_tsquery('english', %(or_query)s) AS or_q
        )
        SELECT p.id, p.content, d.title, d.url,
               p.vector_distance, p.text_score, p.or_text_score
        FROM (
            {union}
        ) p
        JOIN documents d ON d.id = p.document_id
    """


# ---------------------------------------------------------------------------
# Scoring, merging, de-duplication (pure Python, small candidate pool)
# ---------------------------------------------------------------------------


def _merge_rows(rows) -> list[dict]:
    """A chunk can be found by several candidate queries; merge them."""
    merged: dict = {}

    for chunk_id, content, title, url, distance, text_score, or_score in rows:
        distance = float(distance) if distance is not None else None
        text_score = float(text_score or 0.0)
        or_score = float(or_score or 0.0)

        entry = merged.get(chunk_id)
        if entry is None:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "content": content,
                "title": title,
                "url": url,
                "distance": distance,
                "text_score": text_score,
                "or_text_score": or_score,
            }
            continue

        if distance is not None and (
            entry["distance"] is None or distance < entry["distance"]
        ):
            entry["distance"] = distance
        entry["text_score"] = max(entry["text_score"], text_score)
        entry["or_text_score"] = max(entry["or_text_score"], or_score)

    return list(merged.values())


def _heading_overlap(content: str | None, query_tokens: set[str]) -> float:
    if not content or not query_tokens:
        return 0.0

    first_line = content.split("\n", 1)[0]
    if not _HEADING_RE.match(first_line):
        return 0.0

    heading = _HEADING_RE.sub("", first_line).lower()
    heading_tokens = set(_TOKEN_RE.findall(heading))
    return 1.0 if query_tokens & heading_tokens else 0.0


def _score_candidates(
    candidates: list[dict],
    query: str,
    s: _Settings,
) -> list[dict]:
    query_tokens = set(_meaningful_tokens(query))
    scored: list[dict] = []

    for item in candidates:
        distance = item["distance"]
        text_score = item["text_score"]
        or_score = item["or_text_score"]

        # Same pre-filter as the original SQL.
        if not (
            distance is None
            or distance <= s.max_distance
            or text_score > 0
            or or_score > 0
        ):
            continue

        if distance is None or s.max_distance <= 0:
            vector_score = 0.0
        else:
            vector_score = max(0.0, 1.0 - distance / s.max_distance)

        base = (
            vector_score * s.vector_weight
            + min(1.0, text_score) * s.text_weight
            + min(1.0, or_score) * s.or_text_weight
        )

        heading_score = _heading_overlap(item["content"], query_tokens)

        relevance = base
        if base < s.heading_fallback_threshold:
            relevance += heading_score * s.heading_weight

        item["heading_score"] = heading_score
        item["base_relevance_score"] = base
        item["relevance_score"] = relevance
        scored.append(item)

    scored.sort(
        key=lambda r: (
            -r["relevance_score"],
            -r["heading_score"],
            -r["or_text_score"],
            -r["text_score"],
            r["distance"] if r["distance"] is not None else math.inf,
            # Deterministic tie-break so identical scores never flip between
            # runs: prefer the shorter (usually more canonical) URL.
            len(r["url"] or ""),
            r["url"] or "",
        )
    )
    return scored


def _dedupe_by_content(results: list[dict]) -> list[dict]:
    """Drop exact repeated chunk text (e.g. footers/boilerplate repeated on many
    pages), keeping the best-ranked copy. Near-duplicates are not touched."""
    seen: set[str] = set()
    unique: list[dict] = []

    for item in results:
        normalized = " ".join((item["content"] or "").lower().split())
        key = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Reranking (bounded, optional, failure-tolerant)
# ---------------------------------------------------------------------------


def _rerank(
    query: str,
    results: list[dict],
    s: _Settings,
) -> tuple[list[dict], bool]:
    pool = results[: s.rerank_max_candidates]
    fallback = pool[: s.rerank_top_k]

    if not pool or not s.rerank_enabled:
        return fallback, False

    if s.rerank_skip_margin > 0:
        top = pool[0]["relevance_score"]
        second = pool[1]["relevance_score"] if len(pool) > 1 else 0.0
        if top >= s.rerank_skip_min_score and (top - second) >= s.rerank_skip_margin:
            return fallback, False

    try:
        reranked = rerank_results(
            query=query,
            results=pool,
            limit=s.rerank_top_k,
        )
    except Exception:
        logger.exception("Reranker failed; falling back to hybrid ranking")
        return fallback, False

    return list(reranked)[: s.rerank_top_k], True


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
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    s = _load_settings(limit)

    query = _normalize_query(query, s.max_query_chars)
    if not query:
        return []

    # 1. Query embedding. If it fails, keep going with keyword search only.
    embedding_start = time.perf_counter()
    query_embedding = None
    try:
        query_embedding = create_local_embedding(query)
    except Exception:
        logger.exception(
            "Query embedding failed; continuing with keyword-only retrieval"
        )
    embedding_time = time.perf_counter() - embedding_start

    # 2. One round trip to PostgreSQL.
    params: dict = {
        "query": query,
        "or_query": _build_or_query(query),
        "candidate_limit": s.candidate_limit,
    }
    use_vector = query_embedding is not None
    if use_vector:
        params["embedding"] = query_embedding

    connection_start = time.perf_counter()
    with get_database_connection() as conn:
        connection_time = time.perf_counter() - connection_start

        with conn.cursor() as cursor:
            sql_start = time.perf_counter()

            tsv = _tsv_expression(cursor)
            cursor.execute(_build_sql(use_vector, tsv), params)
            rows = cursor.fetchall()

            sql_time = time.perf_counter() - sql_start

    # 3. Score, de-duplicate, rerank.
    processing_start = time.perf_counter()
    candidates = _merge_rows(rows)
    ranked = _dedupe_by_content(_score_candidates(candidates, query, s))
    processing_time = time.perf_counter() - processing_start

    rerank_start = time.perf_counter()
    results, reranked = _rerank(query, ranked, s)
    rerank_time = time.perf_counter() - rerank_start

    total_time = time.perf_counter() - total_start

    logger.info(
        "retriever timing | embed=%.3fs connect=%.3fs sql=%.3fs "
        "python=%.3fs rerank=%.3fs total=%.3fs | rows=%d unique=%d "
        "final=%d reranked=%s vector=%s",
        embedding_time,
        connection_time,
        sql_time,
        processing_time,
        rerank_time,
        total_time,
        len(rows),
        len(ranked),
        len(results),
        reranked,
        use_vector,
    )

    return results