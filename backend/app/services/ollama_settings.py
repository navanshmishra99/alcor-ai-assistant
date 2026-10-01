from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)
OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE",
    "10m",
)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))


def build_ollama_options(
    **overrides: int | float,
) -> dict[str, int | float]:
    options: dict[str, int | float] = {
        "num_ctx": OLLAMA_NUM_CTX,
    }
    options.update(overrides)
    return options