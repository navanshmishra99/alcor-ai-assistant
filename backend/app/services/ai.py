"""
Answer generation for Ask Alcor.

Responsibilities
----------------
- Generate an answer from supplied knowledge only.
- Keep conversation history separate from factual evidence.
- Handle small talk.
- Avoid unnecessary LLM calls when retrieval has no evidence.
- Use exactly one generation call for normal questions.
- Validate generated answers against supplied knowledge.
- Handle Ollama cold starts and transient failures.
- Keep behaviour configurable through environment variables.

Public API
----------
    await generate_answer(question, context, history=None) -> str
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from math import ceil

import httpx
from dotenv import load_dotenv

from .grounding import validate_answer
from .query_understanding import is_small_talk


load_dotenv()

logger = logging.getLogger("ask_alcor.generation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    logger.addHandler(_handler)
logger.propagate = False


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# ASSISTANT / RESPONSES
# ============================================================

ASSISTANT_NAME = os.getenv(
    "AI_ASSISTANT_NAME",
    "AI Assistant",
).strip() or "AI Assistant"

COMPANY_NAME = os.getenv(
    "AI_COMPANY_NAME",
    "the organization",
).strip() or "the organization"

FALLBACK_ANSWER = os.getenv(
    "AI_FALLBACK_ANSWER",
    f"I don't have that information in the {COMPANY_NAME} knowledge base.",
).strip()

EMPTY_KB_ANSWER = os.getenv(
    "AI_EMPTY_KB_ANSWER",
    f"The {COMPANY_NAME} knowledge base has not been populated yet. "
    "Please ingest content before asking questions.",
).strip()

UNAVAILABLE_ANSWER = os.getenv(
    "AI_UNAVAILABLE_ANSWER",
    f"{ASSISTANT_NAME} is temporarily unavailable. "
    "Please try again in a moment.",
).strip()

SMALL_TALK_ANSWER = os.getenv(
    "AI_SMALL_TALK_ANSWER",
    f"Hello! I can answer questions about {COMPANY_NAME}. "
    "What would you like to know?",
).strip()


NON_ANSWERS = {
    FALLBACK_ANSWER,
    EMPTY_KB_ANSWER,
    UNAVAILABLE_ANSWER,
}


# ============================================================
# INPUT LIMITS
# ============================================================

MAX_QUESTION_CHARS = _env_int(
    "AI_MAX_QUESTION_CHARS",
    1000,
)

MAX_CONTEXT_CHARS = _env_int(
    "AI_MAX_CONTEXT_CHARS",
    6000,
)


# ============================================================
# HISTORY
# ============================================================

MAX_HISTORY_MESSAGES = _env_int(
    "AI_MAX_HISTORY_MESSAGES",
    4,
)

MAX_HISTORY_CHARS = _env_int(
    "AI_MAX_HISTORY_CHARS",
    1800,
)


# ============================================================
# OUTPUT
# ============================================================

MIN_OUTPUT_TOKENS = _env_int(
    "AI_MIN_OUTPUT_TOKENS",
    80,
)

MAX_OUTPUT_TOKENS = _env_int(
    "AI_MAX_OUTPUT_TOKENS",
    350,
)


# ============================================================
# OLLAMA
# ============================================================

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
    "30m",
)

TEMPERATURE = _env_float(
    "AI_TEMPERATURE",
    0.1,
)

NUM_CTX = _env_int(
    "AI_NUM_CTX",
    4096,
)


# ============================================================
# HTTP
# ============================================================

REQUEST_TIMEOUT_SECONDS = _env_float(
    "AI_REQUEST_TIMEOUT_SECONDS",
    45,
)

MAX_RETRIES = _env_int(
    "AI_MAX_RETRIES",
    1,
)


# ============================================================
# CACHE
# ============================================================

CACHE_ENABLED = _env_bool(
    "AI_RESPONSE_CACHE_ENABLED",
    False,
)

CACHE_MAX_ITEMS = _env_int(
    "AI_RESPONSE_CACHE_SIZE",
    128,
)


# ============================================================
# GROUNDING
# ============================================================

GROUNDING_ENABLED = _env_bool(
    "AI_GROUNDING_ENABLED",
    True,
)

GROUNDING_STRICT = _env_bool(
    "AI_GROUNDING_STRICT",
    False,
)

# Small talk does not require retrieved knowledge. By default it uses
# the configured static response so a greeting never needs an LLM call.
# Set AI_SMALL_TALK_GENERATE=true only when model-generated small talk
# is explicitly desired.
SMALL_TALK_GENERATE = _env_bool(
    "AI_SMALL_TALK_GENERATE",
    False,
)


# ============================================================
# HTTP CLIENT
# ============================================================

HTTP_MAX_CONNECTIONS = _env_int(
    "AI_HTTP_MAX_CONNECTIONS",
    10,
)

HTTP_MAX_KEEPALIVE_CONNECTIONS = _env_int(
    "AI_HTTP_MAX_KEEPALIVE_CONNECTIONS",
    5,
)

HTTP_CONNECT_TIMEOUT_SECONDS = _env_float(
    "AI_CONNECT_TIMEOUT_SECONDS",
    5.0,
)

_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        REQUEST_TIMEOUT_SECONDS,
        connect=HTTP_CONNECT_TIMEOUT_SECONDS,
    ),
    trust_env=False,
    limits=httpx.Limits(
        max_connections=max(1, HTTP_MAX_CONNECTIONS),
        max_keepalive_connections=max(
            0,
            min(
                HTTP_MAX_KEEPALIVE_CONNECTIONS,
                HTTP_MAX_CONNECTIONS,
            ),
        ),
    ),
)


# ============================================================
# CACHE
# ============================================================

_response_cache: OrderedDict[str, str] = OrderedDict()


def _cache_key(
    question: str,
    context: str,
    history: list[dict],
) -> str:

    history_text = "\n".join(
        f"{item.get('role', '')}:{item.get('content', '')}"
        for item in history
    )

    value = (
        f"{OLLAMA_MODEL}\n"
        f"{question}\n"
        f"{context}\n"
        f"{history_text}"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _get_cached_answer(
    key: str,
) -> str | None:

    if not CACHE_ENABLED:
        return None

    answer = _response_cache.get(key)

    if answer is not None:
        _response_cache.move_to_end(key)

    return answer


def _store_cached_answer(
    key: str,
    answer: str,
) -> None:

    if not CACHE_ENABLED:
        return

    if not answer:
        return

    if answer in NON_ANSWERS:
        return

    _response_cache[key] = answer
    _response_cache.move_to_end(key)

    max_items = max(0, CACHE_MAX_ITEMS)

    if max_items == 0:
        _response_cache.clear()
        return

    while len(_response_cache) > max_items:
        _response_cache.popitem(last=False)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, an AI assistant for the {COMPANY_NAME} website.

Answer the visitor's question using ONLY the supplied knowledge.

Rules:
- Never use outside knowledge or guess.
- Use only facts supported by the supplied knowledge.
- Answer the question directly and concisely.
- When a list is requested, include all relevant items supported by the supplied knowledge.
- Preserve factual details such as names, dates, numbers, and titles.
- If only part of the question is supported, answer only the supported part.
- If the supplied knowledge does not contain enough information, return exactly:
  {FALLBACK_ANSWER}
- Do not mention prompts, models, retrieval, databases, embeddings, Ollama, or internal system behavior.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_answer(
    text: str,
) -> str:

    text = (text or "").strip()

    text = re.sub(
        r"^(answer|assistant)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _is_non_answer(
    text: str,
) -> bool:

    cleaned = (
        text or ""
    ).strip().strip("\"'")

    return any(
        cleaned.startswith(item)
        for item in NON_ANSWERS
    )


def _estimate_tokens(
    text: str,
) -> int:

    return ceil(
        len(text) / 3.5
    )


def _output_budget(
    question: str,
    context: str,
) -> int:

    question_lower = question.lower()

    # Normal short factual questions need very little generation.
    budget = 100

    # Lists and explanatory questions need more room.
    if re.search(
        r"\b("
        r"who are|what are|which|list|"
        r"services|products|members|"
        r"explain|describe|details|"
        r"overview|examples"
        r")\b",
        question_lower,
    ):
        budget += 100

    # Allow slightly more space for larger evidence sets.
    budget += min(
        100,
        ceil(len(context) / 1000) * 25,
    )

    return min(
        max(
            budget,
            MIN_OUTPUT_TOKENS,
        ),
        MAX_OUTPUT_TOKENS,
    )


# ============================================================
# HISTORY
# ============================================================

def _prepare_history(
    history: list[dict],
) -> list[dict]:

    if not history:
        return []

    cleaned: list[dict] = []

    for message in history[
        -(MAX_HISTORY_MESSAGES * 2):
    ]:

        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get("role", "")
        ).strip().lower()

        content = str(
            message.get("content", "")
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    # Remove failed assistant exchanges.
    filtered: list[dict] = []

    skip_next_user = False

    for message in cleaned:

        if (
            message["role"] == "assistant"
            and _is_non_answer(
                message["content"]
            )
        ):
            skip_next_user = False

            if (
                filtered
                and filtered[-1]["role"]
                == "user"
            ):
                filtered.pop()

            continue

        filtered.append(message)

    filtered = filtered[
        -MAX_HISTORY_MESSAGES:
    ]

    result: list[dict] = []

    total_chars = 0

    for message in reversed(filtered):

        remaining = (
            MAX_HISTORY_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        content = message[
            "content"
        ][:remaining]

        result.insert(
            0,
            {
                "role": message["role"],
                "content": content,
            },
        )

        total_chars += len(content)

    return result


# ============================================================
# CONTEXT
# ============================================================

def _compact_context(
    context: str,
) -> str:

    if not context:
        return ""

    # URLs are already returned separately by the API.
    # Keeping them in the LLM prompt wastes context.
    lines = [
        line
        for line in context.splitlines()
        if not line.strip().lower().startswith(
            "url:"
        )
    ]

    return "\n".join(lines).strip()[
        :MAX_CONTEXT_CHARS
    ]


# ============================================================
# MODEL MESSAGES
# ============================================================

def _build_messages(
    question: str,
    context: str,
    history: list[dict],
    small_talk: bool,
) -> list[dict]:

    if small_talk:

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

    messages: list[dict] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # History is included only as reference.
    # It must never compete with the evidence for factual authority.
    if history:

        history_lines = []

        for item in history:

            role = item["role"].upper()

            history_lines.append(
                f"{role}: {item['content']}"
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "CONVERSATION REFERENCE:\n"
                    + "\n".join(history_lines)
                    + "\n\n"
                    "Use this only to understand references "
                    "such as 'he', 'they', 'that', or 'it'. "
                    "Do not use it as a factual source."
                ),
            }
        )

    # IMPORTANT:
    # Evidence + question are the final message.
    # This makes the current retrieval result the strongest
    # information immediately before generation.
    messages.append(
        {
            "role": "user",
            "content": (
                "SUPPLIED KNOWLEDGE\n"
                "==================\n"
                f"{context}\n\n"
                "VISITOR QUESTION\n"
                "================\n"
                f"{question}\n\n"
                "Answer the visitor's question using only "
                "the supplied knowledge."
            ),
        }
    )

    return messages


# ============================================================
# OLLAMA
# ============================================================

def _log_ollama_metrics(
    request_id: str,
    data: dict,
    elapsed: float,
) -> None:

    def seconds(
        key: str,
    ) -> float:

        return (
            data.get(key, 0) or 0
        ) / 1e9

    logger.info(
        "ollama_result "
        "request_id=%s "
        "model=%s "
        "http=%.3fs "
        "server=%.3fs "
        "load=%.3fs "
        "prompt_eval=%.3fs "
        "generation=%.3fs "
        "prompt_tokens=%s "
        "output_tokens=%s "
        "done_reason=%s",
        request_id,
        data.get(
            "model",
            OLLAMA_MODEL,
        ),
        elapsed,
        seconds("total_duration"),
        seconds("load_duration"),
        seconds("prompt_eval_duration"),
        seconds("eval_duration"),
        data.get(
            "prompt_eval_count",
            0,
        ),
        data.get(
            "eval_count",
            0,
        ),
        data.get("done_reason"),
    )


def _is_transient(
    error: Exception,
) -> bool:

    if isinstance(
        error,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
        ),
    ):
        return True

    if isinstance(
        error,
        httpx.HTTPStatusError,
    ):
        return (
            error.response.status_code
            in {
                408,
                425,
                429,
                500,
                502,
                503,
                504,
            }
        )

    return False


async def _call_ollama_once(
    messages: list[dict],
    output_budget: int,
    request_id: str,
) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": output_budget,
            "num_ctx": NUM_CTX,
        },
    }

    started = time.perf_counter()

    response = await _http_client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if response.status_code >= 400:

        logger.error(
            "ollama_http_error "
            "request_id=%s "
            "status=%s "
            "body=%s",
            request_id,
            response.status_code,
            response.text[:500],
        )

        response.raise_for_status()

    data = response.json()

    if data.get("error"):

        raise RuntimeError(
            f"Ollama returned an error: "
            f"{data['error']}"
        )

    _log_ollama_metrics(
        request_id,
        data,
        elapsed,
    )

    return _clean_answer(
        (
            data.get("message") or {}
        ).get(
            "content",
            "",
        )
    )


async def _call_ollama(
    messages: list[dict],
    output_budget: int,
    request_id: str,
) -> str:

    last_error: Exception | None = None

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            return await _call_ollama_once(
                messages=messages,
                output_budget=output_budget,
                request_id=request_id,
            )

        except Exception as error:

            last_error = error

            if (
                not _is_transient(error)
                or attempt >= MAX_RETRIES
            ):
                break

            delay = 1.5 * (
                attempt + 1
            )

            logger.warning(
                "ollama_retry "
                "request_id=%s "
                "attempt=%d/%d "
                "delay=%.1fs "
                "error=%s: %s",
                request_id,
                attempt + 1,
                MAX_RETRIES,
                delay,
                type(error).__name__,
                error,
            )

            await asyncio.sleep(
                delay
            )

    assert last_error is not None

    raise last_error


# ============================================================
# GROUNDING
# ============================================================

async def _run_grounding(
    answer: str,
    context: str,
) -> str:

    if inspect.iscoroutinefunction(
        validate_answer
    ):
        return await validate_answer(
            answer=answer,
            context=context,
        )

    result = await asyncio.to_thread(
        validate_answer,
        answer=answer,
        context=context,
    )

    if inspect.isawaitable(result):
        result = await result

    return result


# ============================================================
# SMALL TALK
# ============================================================

def _safe_is_small_talk(
    question: str,
) -> bool:

    try:
        return bool(
            is_small_talk(question)
        )

    except Exception:

        logger.exception(
            "is_small_talk_failed "
            "question=%r",
            question,
        )

        return False


# ============================================================
# PUBLIC API
# ============================================================

async def generate_answer(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> str:

    total_start = time.perf_counter()

    request_id = uuid.uuid4().hex[:12]

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if not question or not question.strip():
        return FALLBACK_ANSWER

    safe_question = (
        question.strip()
        [:MAX_QUESTION_CHARS]
    )

    safe_context = _compact_context(
        context
    )

    safe_history = _prepare_history(
        history or []
    )

    small_talk = _safe_is_small_talk(
        safe_question
    )

    # --------------------------------------------------------
    # NO RETRIEVED KNOWLEDGE
    # --------------------------------------------------------

    if (
        not safe_context
        and not small_talk
    ):

        logger.warning(
            "no_context "
            "request_id=%s "
            "question=%r",
            request_id,
            safe_question,
        )

        return FALLBACK_ANSWER

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    cache_key = _cache_key(
        safe_question,
        safe_context,
        safe_history,
    )

    cached = _get_cached_answer(
        cache_key
    )

    if cached:

        logger.info(
            "cache_hit "
            "request_id=%s",
            request_id,
        )

        return cached

    # --------------------------------------------------------
    # SMALL TALK
    # --------------------------------------------------------

    if small_talk:

        # Default path: no model call for greetings/thanks.
        # This avoids unnecessary latency and avoids making cold-start
        # behaviour depend on a non-factual conversational request.
        if not SMALL_TALK_GENERATE:
            return SMALL_TALK_ANSWER

        messages = _build_messages(
            question=safe_question,
            context="",
            history=[],
            small_talk=True,
        )

        try:

            answer = await _call_ollama(
                messages=messages,
                output_budget=120,
                request_id=request_id,
            )

        except Exception as error:

            logger.error(
                "small_talk_generation_failed "
                "request_id=%s "
                "error=%s: %s",
                request_id,
                type(error).__name__,
                error,
            )

            return SMALL_TALK_ANSWER

        if (
            not answer
            or _is_non_answer(answer)
        ):
            answer = SMALL_TALK_ANSWER

        _store_cached_answer(
            cache_key,
            answer,
        )

        return answer

    # --------------------------------------------------------
    # NORMAL QUESTION
    # --------------------------------------------------------

    messages = _build_messages(
        question=safe_question,
        context=safe_context,
        history=safe_history,
        small_talk=False,
    )

    output_budget = _output_budget(
        question=safe_question,
        context=safe_context,
    )

    prompt_tokens = sum(
        _estimate_tokens(
            str(
                message.get(
                    "content",
                    "",
                )
            )
        )
        for message in messages
    )

    logger.info(
        "generation_start "
        "request_id=%s "
        "question=%r "
        "context_chars=%d "
        "history=%d "
        "prompt_tokens~%d "
        "output_budget=%d "
        "num_ctx=%d "
        "model=%s",
        request_id,
        safe_question,
        len(safe_context),
        len(safe_history),
        prompt_tokens,
        output_budget,
        NUM_CTX,
        OLLAMA_MODEL,
    )

    # --------------------------------------------------------
    # ONE LLM CALL
    # --------------------------------------------------------

    generation_start = time.perf_counter()

    try:

        answer = await _call_ollama(
            messages=messages,
            output_budget=output_budget,
            request_id=request_id,
        )

    except Exception as error:

        logger.error(
            "generation_failed "
            "request_id=%s "
            "exception=%s "
            "error=%s "
            "question=%r",
            request_id,
            type(error).__name__,
            error,
            safe_question,
        )

        return UNAVAILABLE_ANSWER

    generation_elapsed = (
        time.perf_counter()
        - generation_start
    )

    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not answer:

        logger.warning(
            "generation_empty "
            "request_id=%s",
            request_id,
        )

        return FALLBACK_ANSWER

    # --------------------------------------------------------
    # MODEL FALLBACK
    # --------------------------------------------------------

    if _is_non_answer(answer):

        logger.info(
            "model_declined "
            "request_id=%s "
            "question=%r "
            "context_chars=%d",
            request_id,
            safe_question,
            len(safe_context),
        )

        return FALLBACK_ANSWER

    # --------------------------------------------------------
    # GROUNDING
    # --------------------------------------------------------

    grounding_start = time.perf_counter()

    if not GROUNDING_ENABLED:
        validated = answer
        grounding_elapsed = 0.0
    else:
        try:
            validated = await _run_grounding(
                answer=answer,
                context=safe_context,
            )

        except Exception as error:

            logger.exception(
                "grounding_failed "
                "request_id=%s "
                "exception=%s "
                "error=%s",
                request_id,
                type(error).__name__,
                error,
            )

            if GROUNDING_STRICT:
                return FALLBACK_ANSWER

            # Preserve availability if the validator itself fails.
            validated = answer

        grounding_elapsed = (
            time.perf_counter()
            - grounding_start
        )

    # --------------------------------------------------------
    # GROUNDING REJECTED
    # --------------------------------------------------------

    if (
        not validated
        or not str(validated).strip()
    ):

        logger.warning(
            "grounding_rejected "
            "request_id=%s "
            "question=%r "
            "answer=%r",
            request_id,
            safe_question,
            answer[:500],
        )

        return FALLBACK_ANSWER

    validated = str(
        validated
    ).strip()

    if _is_non_answer(
        validated
    ):

        return FALLBACK_ANSWER

    # --------------------------------------------------------
    # CACHE SUCCESSFUL ANSWER
    # --------------------------------------------------------

    _store_cached_answer(
        cache_key,
        validated,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    logger.info(
        "generation_success "
        "request_id=%s "
        "total=%.3fs "
        "generation=%.3fs "
        "grounding=%.3fs "
        "grounding_enabled=%s "
        "answer_chars=%d",
        request_id,
        time.perf_counter()
        - total_start,
        generation_elapsed,
        grounding_elapsed,
        GROUNDING_ENABLED,
        len(validated),
    )

    return validated

# ============================================================
# LIFECYCLE
# ============================================================

async def close_http_client() -> None:
    """Close the shared Ollama HTTP client during application shutdown."""
    if not _http_client.is_closed:
        await _http_client.aclose()


def clear_response_cache() -> None:
    """Clear the optional local generation cache."""
    _response_cache.clear()



