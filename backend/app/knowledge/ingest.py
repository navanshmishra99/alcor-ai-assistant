from pathlib import Path
import json

from .db import get_database_connection


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