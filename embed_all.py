from backend.app.knowledge.db import get_database_connection
from backend.app.knowledge.local_embeddings import create_local_embedding
from backend.app.knowledge.vector_store import store_embedding


connection = get_database_connection()

try:
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, content
        FROM chunks
        ORDER BY id
        """
    )

    chunks = cursor.fetchall()

finally:
    connection.close()


for chunk_id, content in chunks:
    embedding = create_local_embedding(content)

    store_embedding(
        chunk_id,
        embedding,
    )

    print(
        "EMBEDDED CHUNK:",
        chunk_id,
        "| DIMENSIONS:",
        len(embedding),
    )

print("EMBEDDING COMPLETE")