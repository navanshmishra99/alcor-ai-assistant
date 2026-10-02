import os

from dotenv import load_dotenv

from ..services.ai import get_openai_client

load_dotenv()


def create_embedding(text: str) -> list[float]:
    client = get_openai_client()

    response = client.embeddings.create(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
        input=text,
    )

    return response.data[0].embedding