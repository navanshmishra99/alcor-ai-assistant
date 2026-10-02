from __future__ import annotations

import logging
import os
import re

import httpx

from .ollama_settings import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    build_ollama_options,
)


logger = logging.getLogger("ask_alcor.query_understanding")


# =========================================================
# Configuration
# =========================================================

REQUEST_TIMEOUT = float(
    os.getenv("QUERY_TIMEOUT_SECONDS", "10")
)

MAX_HISTORY_MESSAGES = int(
    os.getenv("QUERY_HISTORY_MESSAGES", "4")
)

MAX_ASSISTANT_CHARS = int(
    os.getenv("QUERY_ASSISTANT_MESSAGE_CHARS", "300")
)

MAX_QUESTION_CHARS = int(
    os.getenv("QUERY_MAX_QUESTION_CHARS", "1000")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("QUERY_MAX_TOKENS", "80")
)


# =========================================================
# Small talk
# =========================================================

_SMALL_TALK_PATTERNS = (
    r"(?:hi|hello|hey)(?: there)?(?: alcor)?",
    r"good (?:morning|afternoon|evening)(?: alcor)?",
    r"(?:how are you(?: doing)?|how's it going|how are things|what's up)",
    r"(?:thanks|thank you)(?: so much)?",
    r"(?:you're welcome|no problem|okay|ok|got it|understood|i see)",
)


def is_small_talk(question: str) -> bool:
    """Return True when the complete message is simple small talk."""

    normalized = re.sub(
        r"\s+",
        " ",
        (question or "").casefold(),
    ).strip()

    normalized = normalized.strip(" .,!?;:")

    return any(
        re.fullmatch(pattern, normalized)
        for pattern in _SMALL_TALK_PATTERNS
    )


# =========================================================
# Prompt
# =========================================================

SYSTEM_PROMPT = """
You convert a user's follow-up message into a concise
retrieval query for a knowledge base.

Use the conversation only to resolve references such as:
he, she, they, it, this, that, his, her, their, etc.

Preserve the user's actual intent.

Do not answer the question.
Do not add facts.
Do not invent names or entities.
Do not broaden or narrow the request.

Return only the retrieval query.
"""


def _build_prompt(
    question: str,
    history: list[dict],
) -> str:

    history_parts = []

    for message in history[-MAX_HISTORY_MESSAGES:]:
        role = str(
            message.get("role", "")
        ).strip().lower()

        content = str(
            message.get("content", "")
        ).strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        if role == "assistant":
            content = content[:MAX_ASSISTANT_CHARS]

        history_parts.append(
            f"{role.upper()}: {content}"
        )

    history_text = "\n".join(history_parts)

    return f"""
CONVERSATION:
{history_text or "(none)"}

CURRENT MESSAGE:
{question[:MAX_QUESTION_CHARS]}

Rewrite the current message as a concise retrieval query.
"""


# =========================================================
# Retrieval query
# =========================================================

def build_retrieval_query(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Convert a genuine follow-up message into a retrieval query.

    The caller decides whether the message is a follow-up.
    This function only performs query understanding.

    If Ollama fails, the original question is returned.
    """

    question = (question or "").strip()

    if not question:
        return ""

    history = history or []

    if not history:
        return question

    prompt = _build_prompt(
        question=question,
        history=history,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": build_ollama_options(
            temperature=0,
            num_predict=MAX_OUTPUT_TOKENS,
        ),
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            raise RuntimeError(
                f"Ollama returned an error: {data['error']}"
            )

        result = (
            data.get("response") or ""
        ).strip()

        # Remove accidental surrounding quotation marks.
        result = result.strip("\"'").strip()

        if result:
            logger.info(
                "Query understanding: %r -> %r",
                question,
                result,
            )

            return result

    except httpx.TimeoutException:
        logger.warning(
            "Query understanding timed out; "
            "using original question."
        )

    except httpx.HTTPError as exc:
        logger.warning(
            "Query understanding HTTP error: %s; "
            "using original question.",
            exc,
        )

    except Exception:
        logger.exception(
            "Query understanding failed; "
            "using original question."
        )

    return question