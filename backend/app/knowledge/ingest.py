from pathlib import Path
import logging
import json

from .chunker import chunk_text, store_document_chunks
from .db import get_database_connection
from .local_embeddings import create_local_embedding
from .vector_store import store_embedding


DEFAULT_KB_DIRECTORIES = (
    Path(__file__).resolve().parents[2] / "data" / "raw",
    Path(__file__).resolve().parents[2] / "data" / "about_poc",
)

logger = logging.getLogger("ask_alcor.knowledge")


def describe_kb_state(document_count: int, chunk_count: int) -> str:
    if document_count > 0 and chunk_count > 0:
        return "Knowledge base is ready."
    if document_count > 0 and chunk_count == 0:
        return "Knowledge base documents exist but chunks are not indexed yet."
    if document_count == 0 and chunk_count == 0:
        return "The knowledge base has not been populated yet."
    return "Knowledge base is partially populated."


def get_document_count() -> int:
    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM documents")
            row = cursor.fetchone()
            return int(row[0]) if row else 0


def get_chunk_count() -> int:
    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            row = cursor.fetchone()
            return int(row[0]) if row else 0


def ensure_knowledge_base_loaded() -> dict[str, int | str]:
    try:
        document_count = get_document_count()
        chunk_count = get_chunk_count()

        if document_count > 0:
            with get_database_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT documents.id, documents.content
                        FROM documents
                        LEFT JOIN chunks
                            ON chunks.document_id = documents.id
                        GROUP BY documents.id
                        HAVING COUNT(chunks.id) = 0
                        ORDER BY documents.id
                        """
                    )
                    documents_without_chunks = cursor.fetchall()

            for document_id, content in documents_without_chunks:
                chunks = chunk_text(content)
                store_document_chunks(document_id, chunks)

            with get_database_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id"
                    )
                    missing_embeddings = cursor.fetchall()

            for chunk_id, content in missing_embeddings:
                store_embedding(chunk_id, create_local_embedding(content))

        document_count = get_document_count()
        chunk_count = get_chunk_count()
        if document_count > 0 and chunk_count > 0:
            return {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "status": describe_kb_state(document_count, chunk_count),
            }

        if document_count == 0:
            for directory in DEFAULT_KB_DIRECTORIES:
                if directory.exists() and ingest_directory(directory):
                    return ensure_knowledge_base_loaded()

        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "status": describe_kb_state(document_count, chunk_count),
        }
    except Exception:
        logger.exception("Knowledge base bootstrap failed.")
        raise


def ingest_json_file(path: Path) -> None:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    url,
                    title,
                    content,
                    content_hash,
                    last_modified,
                    crawled_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (url)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    last_modified = EXCLUDED.last_modified,
                    crawled_at = EXCLUDED.crawled_at
                """,
                (
                    data["url"],
                    data.get("title"),
                    data["content"],
                    data["content_hash"],
                    data.get("last_modified"),
                    data["crawled_at"],
                ),
            )


def ingest_directory(directory: Path) -> int:
    files = sorted(directory.glob("*.json"))

    for path in files:
        ingest_json_file(path)

    return len(files)