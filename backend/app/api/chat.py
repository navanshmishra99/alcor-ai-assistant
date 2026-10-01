from __future__ import annotations

import logging
import os
import random
import re
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.services.ai import (
    FALLBACK_ANSWER,
    UNAVAILABLE_ANSWER,
    generate_answer,
)
from backend.app.services.grounding import validate_answer
from backend.app.services.rag import build_context


router = APIRouter()

logger = logging.getLogger("ask_alcor.chat")


# =========================================================
# Configuration
# =========================================================

MAX_MESSAGE_CHARS = int(os.getenv("CHAT_MAX_MESSAGE_CHARS", "1000"))
MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
MAX_HISTORY_MESSAGE_CHARS = int(os.getenv("CHAT_MAX_HISTORY_MESSAGE_CHARS", "2000"))
MAX_HISTORY_TOTAL_CHARS = int(os.getenv("CHAT_MAX_HISTORY_TOTAL_CHARS", "8000"))
MAX_RETRIEVAL_USER_MESSAGES = int(os.getenv("CHAT_MAX_RETRIEVAL_USER_MESSAGES", "3"))
MAX_ASSISTANT_SNIPPET_CHARS = int(os.getenv("CHAT_MAX_ASSISTANT_SNIPPET_CHARS", "400"))
MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "3"))
GENERATION_ATTEMPTS = int(os.getenv("CHAT_GENERATION_ATTEMPTS", "1"))

# Names used in conversational replies (change via env, not code)
ASSISTANT_NAME = os.getenv("CHAT_ASSISTANT_NAME", "Ask Alcor")
COMPANY_NAME = os.getenv("CHAT_COMPANY_NAME", "Alcor")

# What the assistant tells people it can help with
CAPABILITY_POINTS = [
    f"What {COMPANY_NAME} does and the services it offers",
    f"{COMPANY_NAME}'s CEO and leadership team",
    f"Other company information in the {COMPANY_NAME} knowledge base",
]

EMPTY_MESSAGE_RESPONSE = os.getenv(
    "CHAT_EMPTY_MESSAGE_RESPONSE",
    "How can I help you today?",
).strip()

UNCLEAR_MESSAGE_RESPONSE = (
    "I didn't quite catch that. Could you tell me what you'd like to know?"
)

# The AI layer's exact fallback text. Keep in one place so source handling
# uses exactly the same text as the AI layer.
KNOWLEDGE_BASE_FALLBACK = FALLBACK_ANSWER

# What the user actually sees when nothing relevant is found.
FRIENDLY_FALLBACK = os.getenv(
    "CHAT_FRIENDLY_FALLBACK",
    f"I couldn't find that in the {COMPANY_NAME} knowledge base. "
    "You could try rephrasing your question, or ask me about "
    f"{COMPANY_NAME}'s services, leadership or company information.",
).strip()


# Replies that must never be fed back to the model as history. If they
# stay in the conversation, small models start repeating them.
_NON_ANSWER_PREFIXES = (
    FALLBACK_ANSWER,
    FRIENDLY_FALLBACK,
    UNAVAILABLE_ANSWER,
    "I couldn't find that in the ",
)


def _is_non_answer(text: str) -> bool:
    cleaned = (text or "").strip()
    return any(cleaned.startswith(prefix) for prefix in _NON_ANSWER_PREFIXES)


def _drop_failed_exchanges(messages: list[dict]) -> list[dict]:
    """Remove "not found" / "unavailable" replies and the question before them."""

    drop: set[int] = set()

    for index, message in enumerate(messages):
        if message["role"] == "assistant" and _is_non_answer(message["content"]):
            drop.add(index)

            if index > 0 and messages[index - 1]["role"] == "user":
                drop.add(index - 1)

    return [m for i, m in enumerate(messages) if i not in drop]


# =========================================================
# Conversation models (tolerant of missing / null fields)
# =========================================================

class ChatMessage(BaseModel):
    role: str = ""
    content: str = ""


class ChatRequest(BaseModel):
    message: str = ""
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


# =========================================================
# Input helpers
# =========================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_message(text: str | None) -> str:
    """Remove control characters, normalise whitespace, enforce length."""

    text = _CONTROL_CHARS.sub("", text or "")
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text[:MAX_MESSAGE_CHARS].strip()


def _clean_history(history: list[ChatMessage] | None) -> list[dict]:
    """
    Keep only valid user/assistant messages, trim very long ones, bound the
    total size, and make sure the conversation starts with a user message.
    """

    cleaned: list[dict] = []

    for message in (history or [])[-MAX_HISTORY_TURNS:]:
        role = (message.role or "").strip().lower()
        content = (message.content or "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content[:MAX_HISTORY_MESSAGE_CHARS],
            }
        )

    cleaned = _drop_failed_exchanges(cleaned)

    # Bound total size, keeping the most recent messages.
    total = 0
    bounded: list[dict] = []

    for message in reversed(cleaned):
        total += len(message["content"])

        if total > MAX_HISTORY_TOTAL_CHARS and bounded:
            break

        bounded.append(message)

    bounded.reverse()

    # Conversations should start with the user.
    while bounded and bounded[0]["role"] != "user":
        bounded.pop(0)

    return bounded


# =========================================================
# Small talk (answered instantly, no retrieval, no AI call)
#
# A message only counts as small talk when the WHOLE message matches,
# so "hi, who is the CEO?" still goes through the knowledge base.
# =========================================================

def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


_ADDRESS = r"(?: (?:there|team|everyone|alcor|ask alcor))?"

SMALL_TALK_PATTERNS: dict[str, re.Pattern] = {
    "greeting": re.compile(
        r"(?:hi+|hii+|hello+|hey+|hiya|howdy|yo|greetings|namaste|"
        r"good (?:morning|afternoon|evening|day))" + _ADDRESS
    ),
    "how_are_you": re.compile(
        r"(?:(?:hi|hello|hey) )?(?:how are you(?: doing| today)?|how are u|"
        r"how's it going|hows it going|how is it going|what's up|whats up|"
        r"sup|how do you do)" + _ADDRESS
    ),
    "thanks": re.compile(
        r"(?:(?:ok(?:ay)?|great|awesome|perfect|nice|cool) )?"
        r"(?:thanks?|thank you|thank u|thx|ty|many thanks|much appreciated|"
        r"appreciate it)(?: (?:a lot|so much|very much|again|alcor))*"
    ),
    "goodbye": re.compile(
        r"(?:bye+|goodbye|good bye|see you(?: later| soon)?|see ya|cya|"
        r"take care|have a (?:good|great|nice) (?:day|one|evening)|"
        r"that's all|that is all|thats all|nothing else|no thanks|"
        r"no thank you|i'm done|im done|talk later|talk to you later)"
    ),
    "acknowledge": re.compile(
        r"(?:ok(?:ay)?|cool|great|nice|perfect|awesome|got it|understood|"
        r"alright|all right|sounds good|fine|makes sense|i see|good|"
        r"wonderful|excellent)"
    ),
    "identity": re.compile(
        r"(?:who are you|what are you|what is your name|what's your name|"
        r"whats your name|your name|introduce yourself|tell me about yourself|"
        r"are you (?:an? )?(?:bot|robot|ai|human|real|chatbot|person|"
        r"real person|real human))"
    ),
    "capabilities": re.compile(
        r"(?:help|help me|what can you do|what do you do|"
        r"what can you help (?:me )?with|how can you help(?: me)?|"
        r"what can i ask(?: you)?|what do you know|what (?:topics|questions) "
        r"(?:can|do) (?:you|i) (?:cover|know|ask)|what questions can i ask|"
        r"how do you work|how does this work|how to use this)"
    ),
}


def _capabilities_text() -> str:
    points = "\n".join(f"- {point}" for point in CAPABILITY_POINTS)
    return f"I can help you with:\n{points}"


def _small_talk_reply(intent: str, has_history: bool) -> str:
    if intent == "greeting":
        if has_history:
            return random.choice(
                (
                    "Hello again! What would you like to know?",
                    "Hi! What else can I help you with?",
                )
            )
        return random.choice(
            (
                f"Hello, and welcome! I'm {ASSISTANT_NAME}. How can I help you today?",
                f"Hi there! I'm {ASSISTANT_NAME}. What would you like to know?",
            )
        )

    if intent == "how_are_you":
        return (
            "I'm doing well, thank you for asking. "
            "How can I help you today?"
        )

    if intent == "thanks":
        return random.choice(
            (
                "You're welcome! Is there anything else I can help you with?",
                "Happy to help! Let me know if you have any other questions.",
                "My pleasure. Feel free to ask if there's anything else.",
            )
        )

    if intent == "goodbye":
        return random.choice(
            (
                "Thank you for chatting. Have a great day!",
                "Goodbye! Feel free to come back any time you have a question.",
            )
        )

    if intent == "acknowledge":
        return random.choice(
            (
                "Glad that was helpful. Is there anything else you'd like to know?",
                "Great. Let me know if you have any other questions.",
            )
        )

    if intent == "identity":
        return (
            f"I'm {ASSISTANT_NAME}, the AI assistant for {COMPANY_NAME}. "
            f"I answer questions using the {COMPANY_NAME} knowledge base.\n\n"
            + _capabilities_text()
        )

    if intent == "capabilities":
        return _capabilities_text() + "\n\nWhat would you like to know?"

    return EMPTY_MESSAGE_RESPONSE


def _detect_small_talk(message: str, history: list[dict]) -> str | None:
    normalized = _normalize(message)

    if not normalized or len(normalized) > 60:
        return None

    for intent, pattern in SMALL_TALK_PATTERNS.items():
        if not pattern.fullmatch(normalized):
            continue

        # "ok" / "sure"-style replies to a direct question from the
        # assistant ("Would you like more detail?") are real answers and
        # must reach the AI with the conversation context.
        if intent in {"acknowledge", "goodbye"} and history:
            last = history[-1]

            if last["role"] == "assistant" and last["content"].rstrip().endswith("?"):
                if intent == "acknowledge":
                    return None

        return intent

    return None


# =========================================================
# Follow-up detection and retrieval query
# =========================================================

_FOLLOW_UP_STARTERS = (
    "and ", "also ", "what about", "how about", "why", "how so", "tell me more",
    "more", "elaborate", "explain", "go on", "continue", "yes", "yeah", "yep",
    "sure", "please", "ok", "okay", "but ", "so ", "then ", "what else",
)

_REFERENCE_WORDS = {
    "he", "she", "they", "it", "him", "her", "them", "his", "hers", "their",
    "theirs", "its", "this", "that", "these", "those", "there", "such",
    "former", "latter", "same", "above", "previous",
}


def _is_follow_up(message: str, history: list[dict]) -> bool:
    """
    Decide whether the message depends on earlier conversation.

    Short messages, messages that start with a continuation phrase, and
    messages containing reference words ("he", "that", "their"...) are
    treated as follow-ups. Everything else is treated as a new,
    self-contained question so old topics do not pollute retrieval.
    """

    if not history:
        return False

    normalized = _normalize(message)
    words = normalized.split()

    if not words:
        return False

    if len(words) <= 4:
        return True

    if normalized.startswith(_FOLLOW_UP_STARTERS):
        return True

    if len(words) <= 14 and any(word in _REFERENCE_WORDS for word in words):
        return True

    return False


def _contextual_query(current_message: str, history: list[dict]) -> str:
    """
    Current message plus recent user messages and a short snippet of the
    last assistant answer, so follow-ups keep their topic.
    """

    user_messages: list[str] = []

    for message in reversed(history):
        if message["role"] != "user":
            continue

        user_messages.append(message["content"])

        if len(user_messages) >= MAX_RETRIEVAL_USER_MESSAGES:
            break

    user_messages.reverse()

    parts = list(user_messages)

    for message in reversed(history):
        if message["role"] == "assistant":
            parts.append(message["content"][:MAX_ASSISTANT_SNIPPET_CHARS])
            break

    parts.append(current_message)

    return "\n".join(parts)


# =========================================================
# Answer helpers
# =========================================================

def _is_knowledge_base_fallback(answer: str) -> bool:
    return (answer or "").strip() == KNOWLEDGE_BASE_FALLBACK


def _is_unavailable_response(answer: str) -> bool:
    return (answer or "").strip() == UNAVAILABLE_ANSWER.strip()


def _clean_sources(sources: list[dict]) -> list[dict]:
    """Keep valid source objects, remove duplicate URLs, cap the count."""

    cleaned: list[dict] = []
    seen_urls: set[str] = set()

    for source in sources or []:
        if not isinstance(source, dict):
            continue

        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or "").strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        cleaned.append({"title": title or "Source", "url": url})

        if len(cleaned) >= MAX_SOURCES:
            break

    return cleaned


async def _validated_sources(
    answer: str,
    context: str,
    sources: list[dict],
) -> list[dict]:
    """
    Return sources only when the answer is supported by retrieved knowledge.
    Fallback / unavailable answers never expose sources.
    """

    if (
        not answer
        or not context
        or not sources
        or _is_knowledge_base_fallback(answer)
        or _is_unavailable_response(answer)
    ):
        return []

    try:
        validated = await run_in_threadpool(
            validate_answer,
            answer=answer,
            context=context,
        )
    except Exception:
        logger.exception("Source validation failed.")
        return []

    if not validated or validated.strip() != answer.strip():
        return []

    return sources


async def _generate_with_retry(
    question: str,
    context: str,
    history: list[dict],
) -> str:
    """
    Call the AI layer, retrying briefly on transient failures.
    Returns UNAVAILABLE_ANSWER if every attempt fails.
    """

    answer = UNAVAILABLE_ANSWER

    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            answer = await generate_answer(
                question=question,
                context=context,
                history=history,
            )
        except Exception:
            logger.exception(
                "AI generation failed (attempt %d/%d).",
                attempt,
                GENERATION_ATTEMPTS,
            )
            answer = UNAVAILABLE_ANSWER

        if answer and not _is_unavailable_response(answer):
            return answer

    return UNAVAILABLE_ANSWER


async def _retrieve(queries: list[str]) -> tuple[str, list[dict], str]:
    """
    Try each query in order and return the first one that finds context:
    (context, sources, query_used).
    """

    last_query = queries[0] if queries else ""

    for query in queries:
        last_query = query

        result = await run_in_threadpool(build_context, query)
        result = result or {}

        context = result.get("context", "") or ""

        if context.strip():
            return context, result.get("sources", []) or [], query

    return "", [], last_query


# =========================================================
# Chat endpoint
# =========================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:

    started = time.perf_counter()

    # ---- 1. Input --------------------------------------------------

    current_message = _sanitize_message(request.message)

    if not current_message:
        return ChatResponse(answer=EMPTY_MESSAGE_RESPONSE, sources=[])

    if not re.search(r"\w", current_message):
        # Only punctuation / symbols / emoji
        return ChatResponse(answer=UNCLEAR_MESSAGE_RESPONSE, sources=[])

    history = _clean_history(request.history)

    # ---- 2. Small talk: instant answer, no retrieval or AI call ----

    intent = _detect_small_talk(current_message, history)

    if intent:
        return ChatResponse(
            answer=_small_talk_reply(intent, bool(history)),
            sources=[],
        )

    # ---- 3. Retrieval ----------------------------------------------

    follow_up = _is_follow_up(current_message, history)

    if follow_up:
        queries = [_contextual_query(current_message, history)]
    else:
        queries = [current_message]

        # Standalone question found nothing: maybe it relied on context.
        if history:
            queries.append(_contextual_query(current_message, history))

    try:
        context, retrieved_sources, query_used = await _retrieve(queries)
    except Exception:
        logger.exception("Knowledge retrieval failed.")
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    # Nothing relevant and no conversation to lean on: answer directly,
    # no AI call needed (faster and cannot hallucinate).
    if not context.strip() and not history:
        return ChatResponse(answer=FRIENDLY_FALLBACK, sources=[])

    # ---- 4. Generation ---------------------------------------------

    answer = await _generate_with_retry(
        question=current_message,
        context=context,
        history=history,
    )

    if _is_unavailable_response(answer):
        # Transient failure: a 503 lets the widget retry quietly.
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    answer = (answer or "").strip()

    # ---- 5. Sources ------------------------------------------------

    sources = await _validated_sources(
        answer=answer,
        context=context,
        sources=_clean_sources(retrieved_sources),
    )

    # ---- 6. Present the fallback in a friendlier way ---------------

    if not answer or _is_knowledge_base_fallback(answer):
        answer = FRIENDLY_FALLBACK
        sources = []

    # ---- 7. Diagnostics --------------------------------------------

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "ask_alcor follow_up=%s history=%d total=%.3fs context_chars=%d "
            "answer_chars=%d sources=%d query=%r",
            follow_up,
            len(history),
            time.perf_counter() - started,
            len(context),
            len(answer),
            len(sources),
            query_used[:200],
        )

    return ChatResponse(answer=answer, sources=sources)