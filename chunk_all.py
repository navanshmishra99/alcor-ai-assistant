from backend.app.knowledge.db import get_database_connection
from backend.app.knowledge.chunker import chunk_text, store_document_chunks


connection = get_database_connection()

try:
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, title, content
        FROM documents
        ORDER BY id
        """
    )

    documents = cursor.fetchall()

finally:
    connection.close()


for document_id, title, content in documents:
    chunks = chunk_text(content)

    print(
        "DOCUMENT:",
        title,
        "| CHUNKS:",
        len(chunks),
    )

    store_document_chunks(
        document_id,
        chunks,
    )

print("CHUNKING COMPLETE")