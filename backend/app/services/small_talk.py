from __future__ import annotations

import logging

import httpx

from backend.app.services.ollama_client import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    _get_http_client,
)


logger = logging.getLogger("ask_alcor.small_talk")

FALLBACK_REPLY = (
    "I'm here to help with questions about Alcor Solutions. "
    "What would you like to know?"
)


async def generate_small_talk_reply(
    message: str,
    history: list[dict],
) -> str:
    conversation = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in history[-10:]
        if item.get("role") in {"user", "assistant"}
        and str(item.get("content", "")).strip()
    )
    prompt = (
        f"Conversation history:\n{conversation or '(none)'}\n\n"
        f"Current message: {message.strip()}\n\n"
        "Respond naturally and briefly. For factual questions about Alcor, "
        "do not invent information; invite the user to ask a question "
        "about Alcor if needed."
    )

    try:
        response = await _get_http_client().post(
            "/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": "You are Ask Alcor, a concise and helpful assistant.",
                "prompt": prompt,
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 120,
                },
            },
        )
        response.raise_for_status()
        answer = str(response.json().get("response") or "").strip()
        return answer or FALLBACK_REPLY
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Ollama small-talk generation failed: %s", exc)
        return FALLBACK_REPLY