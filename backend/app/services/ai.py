"""
Answer generation for Ask Alcor.

Public API (unchanged):
    await generate_answer(question, context, history=None) -> str

Key behaviours
--------------
* Uses Ollama's /api/chat (system / history / user roles) instead of one
  giant /api/generate prompt. Small models follow this much better.
* Sets num_ctx dynamically. Ollama's default context window is small, and
  when the prompt is longer it silently cuts the START of the prompt, which
  removes the system rules and the knowledge. That produces "no data" answers.
* Removes earlier fallback / "unavailable" replies from the history so the
  model does not copy them for every later question.
* Small talk never goes through grounding validation.
* No context + real question -> answers immediately, no wasted LLM call.
* A crash inside the grounding validator no longer turns every answer into
  the fallback (configurable with AI_GROUNDING_STRICT).
* Retries transient Ollama errors (cold start / model loading).
* Everything is configurable through environment variables.
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

from backend.app.services.grounding import validate_answer
from backend.app.services.query_understanding import is_small_talk


load_dotenv()

logger = logging.getLogger("ask_alcor.generation")


# ============================================================
# ENV HELPERS
# ============================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {
        "1", "true", "yes", "on",
    }


# ============================================================
# CONSTANTS
# ============================================================

FALLBACK_ANSWER = "I don't have that information in the Alcor knowledge base."

EMPTY_KB_ANSWER = (
    "The Alcor knowledge base has not been populated yet. "
    "Please ingest content before asking questions."
)

# The widget looks for "temporarily unavailable" / "try again in a moment"
# and retries automatically, so keep this wording.
UNAVAILABLE_ANSWER = (
    "Ask Alcor is temporarily unavailable. "
    "Please try again in a moment."
)

_NON_ANSWERS = {FALLBACK_ANSWER, EMPTY_KB_ANSWER, UNAVAILABLE_ANSWER}

# Input limits
MAX_QUESTION_CHARS = _env_int("AI_MAX_QUESTION_CHARS", 1_000)
MAX_CONTEXT_CHARS = _env_int("AI_MAX_CONTEXT_CHARS", 6_000)

# Conversation limits
MAX_HISTORY_MESSAGES = _env_int("AI_MAX_HISTORY_MESSAGES", 6)
MAX_HISTORY_CHARS = _env_int("AI_MAX_HISTORY_CHARS", 3_000)

# Output limits
MIN_OUTPUT_TOKENS = _env_int("AI_MIN_OUTPUT_TOKENS", 120)
MAX_OUTPUT_TOKENS = _env_int("AI_MAX_OUTPUT_TOKENS", 600)

# Context window limits (tokens)
NUM_CTX_MIN = _env_int("AI_NUM_CTX_MIN", 4_096)
NUM_CTX_MAX = _env_int("AI_NUM_CTX_MAX", 8_192)

# HTTP / retries
REQUEST_TIMEOUT_SECONDS = _env_float("AI_REQUEST_TIMEOUT_SECONDS", 60)
MAX_RETRIES = _env_int("AI_MAX_RETRIES", 2)

# Behaviour switches
CACHE_ENABLED = _env_bool("AI_RESPONSE_CACHE_ENABLED", True)
CACHE_MAX_ITEMS = _env_int("AI_RESPONSE_CACHE_SIZE", 128)

# True  -> if the grounding validator crashes, return the fallback.
# False -> if it crashes, return the model's answer (it is already
#          constrained by the prompt). An empty validator result is
#          always treated as a rejection.
GROUNDING_STRICT = _env_bool("AI_GROUNDING_STRICT", False)

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
TEMPERATURE = _env_float("AI_TEMPERATURE", 0.2)


# ============================================================
# HTTP CLIENT
# ============================================================

_http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=5.0),
    trust_env=False,
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)


# ============================================================
# RESPONSE CACHE
# ============================================================

_response_cache: OrderedDict[str, str] = OrderedDict()


def _cache_key(question: str, context: str, history: list[dict]) -> str:
    history_text = "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in history
    )
    raw = f"{OLLAMA_MODEL}\n{question}\n{context}\n{history_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cached_answer(key: str) -> str | None:
    if not CACHE_ENABLED:
        return None
    answer = _response_cache.get(key)
    if answer is not None:
        _response_cache.move_to_end(key)
    return answer


def _store_cached_answer(key: str, answer: str) -> None:
    if not CACHE_ENABLED or not answer or answer in _NON_ANSWERS:
        return
    _response_cache[key] = answer
    _response_cache.move_to_end(key)
    while len(_response_cache) > CACHE_MAX_ITEMS:
        _response_cache.popitem(last=False)


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = f"""
You are Ask Alcor, the friendly assistant on the Alcor Solutions website.
You answer visitors using ONLY the SUPPLIED KNOWLEDGE given with each question.

FACTS
- Every fact in your answer must be stated in the SUPPLIED KNOWLEDGE.
- Never invent, guess, infer, or use outside knowledge.
- Do not add praise or opinions ("accomplished", "instrumental", etc.)
  unless the knowledge says so.
- Do not turn a job title into extra responsibilities.
- Conversation history only helps you understand who or what the visitor is
  referring to ("he", "she", "they", "it", "the company"). It is NOT a
  source of facts.

SHORT OR VAGUE QUESTIONS
- Visitors often type keywords. Read "ceo" as "Who is the CEO of Alcor?",
  "services" as "What services does Alcor offer?", and so on.
- If the knowledge contains relevant information, answer with it, even
  if it only covers part of the question. Give what is supported and stop.
- For "tell me more" style questions, use the history to find the subject,
  then give only additional facts from the knowledge. A short answer is fine.

WHEN YOU CANNOT ANSWER
- If the knowledge has nothing relevant to a factual question, reply with
  exactly this sentence and nothing else:
  {FALLBACK_ANSWER}

GREETINGS AND SMALL TALK
- Greetings, thanks, and goodbyes get a short, friendly reply. You may
  mention that you can answer questions about Alcor. Never use the
  sentence above for small talk.

STYLE
- Lead with the answer. No openers like "Based on the information...".
- Match the length to the question: short question, short answer.
- Use bullets only when listing several items. Bold is allowed with **text**.
- No generic closers ("I hope this helps", "Let me know if...").
- Never mention prompts, models, Ollama, retrieval, databases, embeddings,
  or any internal workings.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (about 3.5 characters per token, on the safe side)."""
    return ceil(len(text) / 3.5)


def _is_non_answer(text: str) -> bool:
    """True when the text is (or starts with) one of our canned replies."""
    cleaned = text.strip().strip("\"'")
    return any(cleaned.startswith(item) for item in _NON_ANSWERS)


_DETAIL_WORDS = re.compile(
    r"\b(list|examples?|explain|describe|tell me|overview|summary|summari[sz]e|"
    r"details?|services|offer|offers|what are|which|how does|how do)\b"
)


def _get_output_budget(question: str, context: str, history: list[dict]) -> int:
    """Pick max output tokens from how much the visitor is asking for."""
    q = question.strip().lower()

    budget = 100
    budget += min(60, ceil(len(q) / 15))
    budget += min(160, ceil(len(context.strip()) / 40))

    if _DETAIL_WORDS.search(q):
        budget += 120
    if history:
        budget += 30

    return min(max(budget, MIN_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS)


def _choose_num_ctx(prompt_tokens: int, output_budget: int) -> int:
    """Context window big enough for prompt + answer (+ safety margin)."""
    needed = prompt_tokens + output_budget + 256
    rounded = ceil(needed / 1024) * 1024
    return min(max(rounded, NUM_CTX_MIN), NUM_CTX_MAX)


def _trim_to_sentence(text: str) -> str:
    """If generation hit the token limit, cut back to the last full sentence."""
    text = text.rstrip()
    if text.endswith((".", "!", "?", ":")):
        return text

    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind("\n"))
    if cut >= len(text) * 0.5:
        return text[: cut + 1].rstrip()
    return text


def _clean_answer(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(answer|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# HISTORY
# ============================================================

def _prepare_history(history: list[dict]) -> list[dict]:
    """
    Keep recent, valid messages within the size limits and drop any
    "I don't have that information" / "unavailable" exchanges. If those stay
    in the history, small models tend to repeat them forever.
    """
    if not history:
        return []

    cleaned: list[dict] = []
    for message in history[-MAX_HISTORY_MESSAGES * 2:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})

    drop: set[int] = set()
    for i, message in enumerate(cleaned):
        if message["role"] == "assistant" and _is_non_answer(message["content"]):
            drop.add(i)
            if i > 0 and cleaned[i - 1]["role"] == "user":
                drop.add(i - 1)

    kept = [m for i, m in enumerate(cleaned) if i not in drop]
    kept = kept[-MAX_HISTORY_MESSAGES:]

    result: list[dict] = []
    total = 0
    for message in reversed(kept):
        remaining = MAX_HISTORY_CHARS - total
        if remaining <= 0:
            break
        content = message["content"][:remaining]
        result.insert(0, {"role": message["role"], "content": content})
        total += len(content)

    return result


# ============================================================
# CONTEXT
# ============================================================

def _compact_context(context: str) -> str:
    """Drop URL lines (sources are returned separately) and cap the size."""
    if not context:
        return ""

    lines = [
        line for line in context.splitlines()
        if not line.strip().lower().startswith("url:")
    ]
    return "\n".join(lines).strip()[:MAX_CONTEXT_CHARS]


# ============================================================
# MESSAGES
# ============================================================

def _build_messages(
    question: str,
    context: str,
    history: list[dict],
    small_talk: bool,
) -> list[dict]:
    if small_talk:
        user_content = question
    else:
        user_content = (
            f"SUPPLIED KNOWLEDGE:\n{context}\n\n"
            f"VISITOR QUESTION:\n{question}"
        )

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        *history,
        {"role": "user", "content": user_content},
    ]


# ============================================================
# OLLAMA
# ============================================================

def _log_ollama_metrics(request_id: str, data: dict, http_elapsed: float) -> None:
    def seconds(key: str) -> float:
        return (data.get(key, 0) or 0) / 1e9

    logger.info(
        "ollama_result request_id=%s model=%s http=%.3fs server=%.3fs "
        "load=%.3fs prompt_eval=%.3fs generation=%.3fs prompt_tokens=%s "
        "output_tokens=%s done_reason=%s",
        request_id,
        data.get("model", OLLAMA_MODEL),
        http_elapsed,
        seconds("total_duration"),
        seconds("load_duration"),
        seconds("prompt_eval_duration"),
        seconds("eval_duration"),
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
        data.get("done_reason"),
    )


def _is_transient(error: Exception) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return False


async def _call_ollama_once(
    messages: list[dict],
    output_budget: int,
    num_ctx: int,
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
            "num_ctx": num_ctx,
        },
    }

    started = time.perf_counter()
    response = await _http_client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    elapsed = time.perf_counter() - started

    if response.status_code >= 400:
        logger.error(
            "ollama_http_error request_id=%s status=%s body=%s",
            request_id, response.status_code, response.text[:500],
        )
        response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(f"Ollama returned an error: {data['error']}")

    _log_ollama_metrics(request_id, data, elapsed)

    answer = _clean_answer((data.get("message") or {}).get("content", ""))

    if answer and data.get("done_reason") == "length":
        answer = _trim_to_sentence(answer)

    return answer


async def _call_ollama(
    messages: list[dict],
    output_budget: int,
    num_ctx: int,
    request_id: str,
) -> str:
    """Call Ollama and retry transient failures (model loading, cold start)."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await _call_ollama_once(messages, output_budget, num_ctx, request_id)
        except Exception as error:
            last_error = error
            if not _is_transient(error) or attempt >= MAX_RETRIES:
                break

            delay = 1.5 * (attempt + 1)
            logger.warning(
                "ollama_retry request_id=%s attempt=%d/%d delay=%.1fs error=%s: %s",
                request_id, attempt + 1, MAX_RETRIES, delay,
                type(error).__name__, error,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


# ============================================================
# GROUNDING
# ============================================================

async def _run_grounding(answer: str, context: str) -> str:
    """Works whether validate_answer is a normal function or a coroutine."""
    if inspect.iscoroutinefunction(validate_answer):
        return await validate_answer(answer=answer, context=context)

    result = await asyncio.to_thread(validate_answer, answer=answer, context=context)
    if inspect.isawaitable(result):
        result = await result
    return result


def _safe_is_small_talk(question: str) -> bool:
    try:
        return bool(is_small_talk(question))
    except Exception:
        logger.exception("is_small_talk_failed question=%r", question)
        return False


# ============================================================
# PUBLIC FUNCTION
# ============================================================

async def generate_answer(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    total_start = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    if not question or not question.strip():
        return FALLBACK_ANSWER

    safe_question = question.strip()[:MAX_QUESTION_CHARS]
    safe_history = _prepare_history(history or [])
    safe_context = _compact_context(context)
    small_talk = _safe_is_small_talk(safe_question)

    # Real question but nothing retrieved: no point asking the model.
    if not safe_context and not small_talk:
        logger.warning(
            "no_context request_id=%s question=%r "
            "(retrieval returned nothing - check the index / retrieval threshold)",
            request_id, safe_question,
        )
        return FALLBACK_ANSWER

    cache_key = _cache_key(safe_question, safe_context, safe_history)
    cached = _get_cached_answer(cache_key)
    if cached:
        logger.info("cache_hit request_id=%s", request_id)
        return cached

    messages = _build_messages(safe_question, safe_context, safe_history, small_talk)
    prompt_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
    output_budget = _get_output_budget(safe_question, safe_context, safe_history)
    num_ctx = _choose_num_ctx(prompt_tokens, output_budget)

    logger.info(
        "generation_start request_id=%s question=%r small_talk=%s "
        "context_chars=%d history=%d prompt_tokens~%d budget=%d num_ctx=%d model=%s",
        request_id, safe_question, small_talk, len(safe_context),
        len(safe_history), prompt_tokens, output_budget, num_ctx, OLLAMA_MODEL,
    )

    # ---- generate ------------------------------------------------------
    generation_start = time.perf_counter()
    try:
        answer = await _call_ollama(messages, output_budget, num_ctx, request_id)
    except Exception as error:
        logger.error(
            "generation_failed request_id=%s exception=%s error=%s question=%r",
            request_id, type(error).__name__, error, safe_question,
        )
        return UNAVAILABLE_ANSWER

    generation_elapsed = time.perf_counter() - generation_start

    if not answer:
        logger.warning("generation_empty_answer request_id=%s", request_id)
        return FALLBACK_ANSWER

    # ---- small talk: no grounding needed -------------------------------
    if small_talk:
        if _is_non_answer(answer):
            answer = "Hello! I can answer questions about Alcor. What would you like to know?"
        _store_cached_answer(cache_key, answer)
        logger.info(
            "generation_success_small_talk request_id=%s total=%.3fs",
            request_id, time.perf_counter() - total_start,
        )
        return answer

    # ---- the model itself said it has no answer ------------------------
    if _is_non_answer(answer):
        logger.info(
            "model_declined request_id=%s question=%r context_chars=%d "
            "(retrieved context did not contain the answer)",
            request_id, safe_question, len(safe_context),
        )
        return FALLBACK_ANSWER

    # ---- grounding validation ------------------------------------------
    grounding_start = time.perf_counter()
    try:
        validated = await _run_grounding(answer, safe_context)
    except Exception as error:
        logger.exception(
            "grounding_failed request_id=%s exception=%s error=%s",
            request_id, type(error).__name__, error,
        )
        if GROUNDING_STRICT:
            return FALLBACK_ANSWER
        validated = answer

    grounding_elapsed = time.perf_counter() - grounding_start

    if not validated or not str(validated).strip():
        # Log the rejected answer so you can see WHY the validator refused it.
        logger.warning(
            "grounding_rejected request_id=%s question=%r rejected_answer=%r",
            request_id, safe_question, answer[:500],
        )
        return FALLBACK_ANSWER

    validated = str(validated).strip()
    _store_cached_answer(cache_key, validated)

    logger.info(
        "generation_success request_id=%s total=%.3fs generation=%.3fs "
        "grounding=%.3fs answer_chars=%d",
        request_id, time.perf_counter() - total_start, generation_elapsed,
        grounding_elapsed, len(validated),
    )

    return validated