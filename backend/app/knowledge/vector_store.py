from .db import get_database_connection


def store_embedding(
    chunk_id: int,
    embedding: list[float],
) -> None:
    with get_database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chunks
                SET embedding = %s
                WHERE id = %s
                """,
                (embedding, chunk_id),
            )