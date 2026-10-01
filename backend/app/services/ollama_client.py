from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger("ask_alcor.ollama")

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

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client

    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=30.0,
        )

    return _http_client


async def warm_ollama_model() -> None:
    try:
        response = await _get_http_client().post(
            "/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Reply with OK.",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "num_predict": 1,
                },
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Ollama warm-up skipped: service unavailable or model not loaded; "
                "model=%s error=%s",
                OLLAMA_MODEL,
                exc,
            )


async def close_http_client() -> None:
    global _http_client

    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None