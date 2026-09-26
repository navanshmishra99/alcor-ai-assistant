from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from .db import get_database_connection
from .local_embeddings import create_local_embedding


load_dotenv()


def _get_float_setting(
    name: str,
    default: float,
) -> float:
    return float(
        os.getenv(
            name,
            str(default),
        )
    )


def _get_int_setting(
    name: str,
    default: int,
) -> int:
    return int(
        os.getenv(
            name,
            str(default),
        )
    )


def search_knowledge(
    query: str,
    limit: int = 5,
) -> list[dict]:

    total_start = time.perf_counter()

    if not query or not query.strip():
        return []

    query = query.strip()

    # ---------------------------------------------------------
    # 1. Query embedding
    # ---------------------------------------------------------
    embedding_start = time.perf_counter()

    query_embedding = create_local_embedding(query)

    embedding_time = time.perf_counter() - embedding_start

    # ---------------------------------------------------------
    # Retrieval settings
    # ---------------------------------------------------------
    max_distance = _get_float_setting(
        "RETRIEVAL_MAX_DISTANCE",
        0.80,
    )

    candidate_limit = _get_int_setting(
        "RETRIEVAL_CANDIDATE_LIMIT",
        max(limit * 5, 10),
    )

    vector_weight = _get_float_setting(
        "RETRIEVAL_VECTOR_WEIGHT",
        0.70,
    )

    text_weight = _get_float_setting(
        "RETRIEVAL_TEXT_WEIGHT",
        0.30,
    )

    heading_weight = _get_float_setting(
        "RETRIEVAL_HEADING_WEIGHT",
        0.10,
    )

    heading_fallback_threshold = _get_float_setting(
        "RETRIEVAL_HEADING_FALLBACK_THRESHOLD",
        0.35,
    )

    # ---------------------------------------------------------
    # 2. Database connection
    # ---------------------------------------------------------
    connection_start = time.perf_counter()

    with get_database_connection() as conn:

        connection_time = time.perf_counter() - connection_start

        with conn.cursor() as cursor:

            # -------------------------------------------------
            # 3. PostgreSQL retrieval query
            # -------------------------------------------------
            query_start = time.perf_counter()

            cursor.execute(
                """
                WITH vector_candidates AS (

                    SELECT
                        chunks.id,
                        chunks.content,
                        documents.title,
                        documents.url,

                        chunks.embedding <=> %s::vector
                            AS vector_distance

                    FROM chunks

                    JOIN documents
                        ON documents.id = chunks.document_id

                    WHERE chunks.embedding IS NOT NULL

                    ORDER BY
                        chunks.embedding <=> %s::vector

                    LIMIT %s
                ),

                text_candidates AS (

                    SELECT
                        chunks.id,
                        chunks.content,
                        documents.title,
                        documents.url,

                        ts_rank_cd(
                            to_tsvector(
                                'english',
                                COALESCE(
                                    chunks.content,
                                    ''
                                )
                            ),
                            plainto_tsquery(
                                'english',
                                %s
                            )
                        ) AS text_score

                    FROM chunks

                    JOIN documents
                        ON documents.id = chunks.document_id

                    WHERE
                        to_tsvector(
                            'english',
                            COALESCE(
                                chunks.content,
                                ''
                            )
                        )
                        @@ plainto_tsquery(
                            'english',
                            %s
                        )

                    ORDER BY
                        ts_rank_cd(
                            to_tsvector(
                                'english',
                                COALESCE(
                                    chunks.content,
                                    ''
                                )
                            ),
                            plainto_tsquery(
                                'english',
                                %s
                            )
                        ) DESC

                    LIMIT %s
                ),

                or_text_candidates AS (

                    SELECT
                        chunks.id,
                        chunks.content,
                        documents.title,
                        documents.url,

                        ts_rank_cd(
                            to_tsvector(
                                'english',
                                COALESCE(
                                    chunks.content,
                                    ''
                                )
                            ),
                            replace(
                                plainto_tsquery(
                                    'english',
                                    %s
                                )::text,
                                ' & ',
                                ' | '
                            )::tsquery
                        ) AS or_text_score

                    FROM chunks

                    JOIN documents
                        ON documents.id = chunks.document_id

                    WHERE
                        to_tsvector(
                            'english',
                            COALESCE(
                                chunks.content,
                                ''
                            )
                        )
                        @@ replace(
                            plainto_tsquery(
                                'english',
                                %s
                            )::text,
                            ' & ',
                            ' | '
                        )::tsquery

                    ORDER BY
                        ts_rank_cd(
                            to_tsvector(
                                'english',
                                COALESCE(
                                    chunks.content,
                                    ''
                                )
                            ),
                            replace(
                                plainto_tsquery(
                                    'english',
                                    %s
                                )::text,
                                ' & ',
                                ' | '
                            )::tsquery
                        ) DESC

                    LIMIT %s
                ),

                candidate_pool AS (

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        vector_distance,
                        0.0::double precision
                            AS text_score,
                        0.0::double precision
                            AS or_text_score

                    FROM vector_candidates

                    UNION

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        NULL::double precision
                            AS vector_distance,
                        text_score,
                        0.0::double precision
                            AS or_text_score

                    FROM text_candidates

                    UNION

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        NULL::double precision
                            AS vector_distance,
                        0.0::double precision
                            AS text_score,
                        or_text_score

                    FROM or_text_candidates
                ),

                merged_candidates AS (

                    SELECT
                        id,
                        MAX(content) AS content,
                        MAX(title) AS title,
                        MAX(url) AS url,

                        MIN(vector_distance)
                            AS vector_distance,

                        MAX(text_score)
                            AS text_score,

                        MAX(or_text_score)
                            AS or_text_score

                    FROM candidate_pool

                    GROUP BY id
                ),

                normalized_candidates AS (

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        vector_distance,
                        text_score,
                        or_text_score,

                        CASE
                            WHEN vector_distance IS NULL
                                THEN 0.0

                            WHEN %s <= 0
                                THEN 0.0

                            ELSE GREATEST(
                                0.0,
                                1.0 - (
                                    vector_distance
                                    / %s
                                )
                            )
                        END AS vector_score,

                        CASE
                            WHEN text_score <= 0
                                THEN 0.0

                            ELSE LEAST(
                                1.0,
                                text_score
                            )
                        END AS normalized_text_score,

                        CASE
                            WHEN or_text_score <= 0
                                THEN 0.0

                            ELSE LEAST(
                                1.0,
                                or_text_score
                            )
                        END AS normalized_or_text_score,

                        CASE
                            WHEN split_part(
                                COALESCE(
                                    content,
                                    ''
                                ),
                                E'\n',
                                1
                            ) !~ '^#{1,6}[[:space:]]+'
                            THEN 0.0

                            ELSE
                                CASE
                                    WHEN (
                                        SELECT COUNT(*)
                                        FROM regexp_split_to_table(
                                            lower(query_text),
                                            '[^[:alnum:]]+'
                                        ) AS query_token
                                        WHERE
                                            query_token <> ''
                                            AND EXISTS (
                                                SELECT 1
                                                FROM regexp_split_to_table(
                                                    lower(heading_text),
                                                    '[^[:alnum:]]+'
                                                ) AS heading_token
                                                WHERE
                                                    heading_token <> ''
                                                    AND heading_token
                                                        = query_token
                                            )
                                    ) > 0
                                    THEN 1.0
                                    ELSE 0.0
                                END
                        END AS heading_overlap_score

                    FROM (
                        SELECT
                            merged_candidates.*,
                            %s AS query_text,
                            regexp_replace(
                                split_part(
                                    COALESCE(
                                        merged_candidates.content,
                                        ''
                                    ),
                                    E'\n',
                                    1
                                ),
                                '^#{1,6}[[:space:]]+',
                                ''
                            ) AS heading_text
                        FROM merged_candidates
                    ) AS candidate_data
                ),

                base_scored_candidates AS (

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        vector_distance,
                        text_score,
                        or_text_score,
                        heading_overlap_score,

                        (
                            vector_score
                            * %s
                        )
                        +
                        (
                            normalized_text_score
                            * %s
                        )
                        +
                        (
                            normalized_or_text_score
                            * %s
                        ) AS base_relevance_score

                    FROM normalized_candidates
                ),

                scored_candidates AS (

                    SELECT
                        id,
                        content,
                        title,
                        url,
                        vector_distance,
                        text_score,
                        or_text_score,
                        heading_overlap_score,
                        base_relevance_score,

                        base_relevance_score
                        +
                        CASE
                            WHEN base_relevance_score
                                < %s
                            THEN
                                heading_overlap_score
                                * %s
                            ELSE 0.0
                        END AS relevance_score

                    FROM base_scored_candidates
                )

                SELECT
                    id,
                    content,
                    title,
                    url,
                    vector_distance,
                    text_score,
                    or_text_score,
                    heading_overlap_score,
                    base_relevance_score,
                    relevance_score

                FROM scored_candidates

                WHERE
                    (
                        vector_distance IS NULL
                        OR vector_distance <= %s
                        OR text_score > 0
                        OR or_text_score > 0
                    )

                ORDER BY
                    relevance_score DESC,
                    heading_overlap_score DESC,
                    or_text_score DESC,
                    text_score DESC,
                    vector_distance ASC NULLS LAST

                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    candidate_limit,

                    query,
                    query,
                    query,
                    candidate_limit,

                    query,
                    query,
                    query,
                    candidate_limit,

                    max_distance,
                    max_distance,

                    query,

                    vector_weight,
                    text_weight,
                    0.15,

                    heading_fallback_threshold,
                    heading_weight,

                    max_distance,

                    limit,
                ),
            )

            query_time = time.perf_counter() - query_start

            # ---------------------------------------------
            # 4. Fetch results
            # ---------------------------------------------
            fetch_start = time.perf_counter()

            rows = cursor.fetchall()

            fetch_time = time.perf_counter() - fetch_start

    # ---------------------------------------------------------
    # 5. Python result processing
    # ---------------------------------------------------------
    processing_start = time.perf_counter()

    results = []

    for row in rows:
        results.append(
            {
                "chunk_id": row[0],
                "content": row[1],
                "title": row[2],
                "url": row[3],
                "distance": (
                    float(row[4])
                    if row[4] is not None
                    else None
                ),
                "text_score": float(row[5]),
                "or_text_score": float(row[6]),
                "heading_score": float(row[7]),
                "base_relevance_score": float(row[8]),
                "relevance_score": float(row[9]),
            }
        )

    processing_time = time.perf_counter() - processing_start
    total_time = time.perf_counter() - total_start

    # ---------------------------------------------------------
    # Diagnostic timing
    # ---------------------------------------------------------
    print("\n========== RETRIEVER TIMING ==========")
    print(f"Embedding       : {embedding_time:.3f}s")
    print(f"DB Connection   : {connection_time:.3f}s")
    print(f"SQL Query       : {query_time:.3f}s")
    print(f"Fetch Results   : {fetch_time:.3f}s")
    print(f"Python Process  : {processing_time:.3f}s")
    print(f"RAG Total       : {total_time:.3f}s")
    print("======================================\n")

    return results