from backend.app.knowledge.db import get_database_connection


query = "What major milestone did Alcor reach in 2018?"

sql = """
SELECT
    chunk_index,
    ts_rank_cd(
        to_tsvector('english', content),
        replace(
            plainto_tsquery('english', %s)::text,
            ' & ',
            ' | '
        )::tsquery
    ) AS score,
    content
FROM chunks
WHERE
    to_tsvector('english', content)
    @@ replace(
        plainto_tsquery('english', %s)::text,
        ' & ',
        ' | '
    )::tsquery
ORDER BY score DESC
LIMIT 8
"""


with get_database_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute(sql, (query, query))
        rows = cursor.fetchall()


for row in rows:
    print()
    print("CHUNK:", row[0])
    print("OR FTS SCORE:", round(row[1], 4))
    print(row[2])