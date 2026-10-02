"""Ask Alcor chat endpoint: POST /chat

What happens to a visitor message
---------------------------------
1. Clean the input; answer greetings and thanks instantly (no AI call).
2. Repeat standalone questions are answered from a short-lived cache.
3. Otherwise: find relevant passages, ask the model, check the answer against
   the passages, and return the answer with its sources.
4. If the knowledge base has nothing, say so politely and point to a human
   contact. Those questions are logged so the team can add the missing content.

Business-friendly settings (all optional, set in .env)
-------------------------------------------------------
Wording (change what the assistant says without touching code)
    CHAT_ASSISTANT_NAME, CHAT_COMPANY_NAME
    CHAT_CAPABILITY_POINTS      "point one|point two|point three"
    CHAT_FRIENDLY_FALLBACK      shown when nothing relevant is found
    CHAT_CONTACT_URL            added to that message, e.g. your contact page
    CHAT_BUSY_RESPONSE, CHAT_RATE_LIMIT_RESPONSE, CHAT_EMPTY_MESSAGE_RESPONSE

Speed and capacity
    CHAT_ANSWER_CACHE_TTL_SECONDS   repeat questions answered instantly (300, 0=off)
    CHAT_MODEL_HISTORY_MESSAGES     chat history sent to the model on follow-ups (6)
    CHAT_MAX_CONCURRENT_GENERATIONS model calls at once (2)
    CHAT_QUEUE_TIMEOUT_SECONDS      max wait for a free slot before "busy" (25)
    CHAT_REQUEST_TIMEOUT_SECONDS    hard limit for one request (60, 0=off)
    CHAT_GROUNDING_ENABLED          check answers against sources (true)
    CHAT_RATE_LIMIT_PER_MINUTE      messages per visitor per minute (0=off)
    CHAT_TRUST_PROXY                use X-Forwarded-For for the visitor (false)

Insight and diagnostics
    One short line per request is logged to "ask_alcor.analytics" (INFO).
    A detailed diagnosis block is logged (WARNING) only when something goes
    wrong, or for every request when CHAT_DEBUG=true.
    CHAT_UNANSWERED_LOG_PATH    append unanswered questions to this .jsonl file
    CHAT_LOG_QUESTIONS          false hides visitor text from logs (true)
    CHAT_DEBUG_RESPONSE         add a "debug" object to the JSON (development only)
    CHAT_DEBUG_CONTEXT_CHARS / CHAT_DEBUG_ANSWER_CHARS   preview lengths

Operations
    warm_up()           await at startup so the first visitor is not slow
    clear_answer_cache() call after re-ingesting the knowledge base
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..services.ai import (
    FALLBACK_ANSWER,
    UNAVAILABLE_ANSWER,
    generate_answer,
)
from ..services.grounding import validate_answer
from ..services.query_understanding import build_retrieval_query
from ..services.rag import build_context


router = APIRouter()

logger = logging.getLogger("ask_alcor.chat")
analytics_logger = logging.getLogger("ask_alcor.analytics")


# =========================================================
# Configuration (read once at start-up)
# =========================================================

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_int(name: str, default: int, minimum: int | None = None,
             maximum: int | None = None) -> int:
    try:
        value = int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError, OverflowError):
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value != value or value in (float("inf"), float("-inf")):
        return default
    return max(minimum, value) if minimum is not None else value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _Config:
    # Input limits
    max_message_chars: int
    max_history_turns: int
    max_history_message_chars: int
    max_history_total_chars: int
    max_retrieval_user_messages: int
    max_assistant_snippet_chars: int
    max_sources: int
    generation_attempts: int
    model_history_messages: int

    # Wording
    assistant_name: str
    company_name: str
    capability_points: tuple
    contact_url: str
    empty_message_response: str
    unclear_message_response: str
    friendly_fallback: str
    busy_response: str
    rate_limit_response: str

    # Speed and capacity
    answer_cache_ttl_seconds: int
    max_concurrent_generations: int
    queue_timeout_seconds: float
    request_timeout_seconds: float
    grounding_enabled: bool
    rate_limit_per_minute: int
    trust_proxy: bool

    # Diagnostics
    debug_logs: bool
    debug_response: bool
    debug_context_chars: int
    debug_answer_chars: int
    log_questions: bool
    unanswered_log_path: str


def _load_config() -> _Config:
    assistant = _env_str("CHAT_ASSISTANT_NAME", "Ask Alcor")
    company = _env_str("CHAT_COMPANY_NAME", "Alcor")

    default_points = (
        f"What {company} does and the services it offers",
        f"{company}'s CEO and leadership team",
        f"Other company information in the {company} knowledge base",
    )
    raw_points = os.getenv("CHAT_CAPABILITY_POINTS", "")
    points = tuple(p.strip() for p in raw_points.split("|") if p.strip())

    return _Config(
        max_message_chars=_env_int("CHAT_MAX_MESSAGE_CHARS", 1000, 20),
        max_history_turns=_env_int("CHAT_MAX_HISTORY_TURNS", 10, 0),
        max_history_message_chars=_env_int("CHAT_MAX_HISTORY_MESSAGE_CHARS", 2000, 50),
        max_history_total_chars=_env_int("CHAT_MAX_HISTORY_TOTAL_CHARS", 8000, 100),
        max_retrieval_user_messages=_env_int("CHAT_MAX_RETRIEVAL_USER_MESSAGES", 3, 1),
        max_assistant_snippet_chars=_env_int("CHAT_MAX_ASSISTANT_SNIPPET_CHARS", 400, 0),
        max_sources=_env_int("CHAT_MAX_SOURCES", 3, 0),
        generation_attempts=_env_int("CHAT_GENERATION_ATTEMPTS", 1, 1, 5),
        model_history_messages=_env_int("CHAT_MODEL_HISTORY_MESSAGES", 6, 0, 50),

        assistant_name=assistant,
        company_name=company,
        capability_points=points or default_points,
        contact_url=_env_str("CHAT_CONTACT_URL", ""),
        empty_message_response=_env_str(
            "CHAT_EMPTY_MESSAGE_RESPONSE", "How can I help you today?"
        ),
        unclear_message_response=_env_str(
            "CHAT_UNCLEAR_MESSAGE_RESPONSE",
            "I didn't quite catch that. Could you tell me what you'd like to know?",
        ),
        friendly_fallback=_env_str(
            "CHAT_FRIENDLY_FALLBACK",
            f"I couldn't find that in the {company} knowledge base. "
            "You could try rephrasing your question, or ask me about "
            f"{company}'s services, leadership or company information.",
        ),
        busy_response=_env_str(
            "CHAT_BUSY_RESPONSE",
            "I'm helping a lot of people right now. Please try again in a moment.",
        ),
        rate_limit_response=_env_str(
            "CHAT_RATE_LIMIT_RESPONSE",
            "You're sending messages quite quickly. "
            "Please wait a moment and try again.",
        ),

        answer_cache_ttl_seconds=_env_int("CHAT_ANSWER_CACHE_TTL_SECONDS", 300, 0, 86400),
        max_concurrent_generations=_env_int("CHAT_MAX_CONCURRENT_GENERATIONS", 1, 1, 64),
        queue_timeout_seconds=_env_float("CHAT_QUEUE_TIMEOUT_SECONDS", 25.0, 0.05),
        request_timeout_seconds=_env_float("CHAT_REQUEST_TIMEOUT_SECONDS", 120.0, 0.0),
        grounding_enabled=_env_bool("CHAT_GROUNDING_ENABLED", True),
        rate_limit_per_minute=_env_int("CHAT_RATE_LIMIT_PER_MINUTE", 0, 0, 10000),
        trust_proxy=_env_bool("CHAT_TRUST_PROXY", False),

        debug_logs=_env_bool("CHAT_DEBUG", False),
        debug_response=_env_bool("CHAT_DEBUG_RESPONSE", False),
        debug_context_chars=_env_int("CHAT_DEBUG_CONTEXT_CHARS", 600, 0),
        debug_answer_chars=_env_int("CHAT_DEBUG_ANSWER_CHARS", 300, 0),
        log_questions=_env_bool("CHAT_LOG_QUESTIONS", True),
        unanswered_log_path=_env_str("CHAT_UNANSWERED_LOG_PATH", ""),
    )


CFG = _load_config()

# The AI layer's exact fallback text (kept in one place so source handling
# uses exactly the same text as the AI layer).
KNOWLEDGE_BASE_FALLBACK = FALLBACK_ANSWER


def _friendly_fallback() -> str:
    """What the visitor sees when nothing relevant is found."""
    text = CFG.friendly_fallback
    if CFG.contact_url:
        text += f" You can also contact the team here: {CFG.contact_url}"
    return text


# Replies that must never be fed back to the model as history. If they stay
# in the conversation, small models start repeating them.
def _non_answer_prefixes() -> tuple:
    return (
        FALLBACK_ANSWER,
        CFG.friendly_fallback,
        UNAVAILABLE_ANSWER,
        CFG.busy_response,
        CFG.rate_limit_response,
        "I couldn't find that in the ",
    )


def _is_non_answer(text: str) -> bool:
    cleaned = (text or "").strip()
    return any(cleaned.startswith(prefix) for prefix in _non_answer_prefixes())


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
# Request trace and diagnostics
#
# One _Trace follows each request. Healthy requests produce a single short
# analytics line. When something goes wrong (or CHAT_DEBUG=true) a detailed
# SUMMARY with a likely cause is logged as well.
# =========================================================

_OK_OUTCOMES = {
    "success",
    "answer_cache_hit",
    "empty_message",
    "symbol_only_message",
    "retrieval_skipped",
}

# Questions the knowledge base could not answer: valuable for the business.
_UNANSWERED_OUTCOMES = {"kb_fallback_from_model", "no_context"}


def _is_ok_outcome(outcome: str) -> bool:
    return outcome in _OK_OUTCOMES or outcome.startswith("small_talk")


class _Trace:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.started = time.perf_counter()
        self.outcome = "unknown"
        self.data: dict = {}
        self.stages: dict[str, float] = {}
        self.notes: list[str] = []

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def note(self, text: str) -> None:
        self.notes.append(text)
        logger.debug("[%s] NOTE: %s", self.request_id, text)

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started


def _one_line(text: str | None, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _describe_item(item: dict) -> str:
    """Compact 'key=value' view of a source dict (title, url, scores)."""

    parts: list[str] = []

    for key, value in item.items():
        if isinstance(value, bool) or isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        elif key in {"title", "url", "source", "heading"} and value:
            parts.append(f"{key}={_one_line(str(value), 90)}")

    return " ".join(parts) or "(no scalar fields)"


_COVERAGE_STOPWORDS = frozenset(
    """
    the and for with from that this these those what who whom whose which when
    where why how are was were been being does did done have has had can could
    should would will may might must shall about into over under than then
    there here not yes tell please give show list name describe explain any
    you your our their his her its them they she him
    """.split()
)


def _term_coverage(question: str, context: str) -> tuple[list[str], list[str]]:
    """Which meaningful question words appear in the retrieved context?
    Separates "retrieval missed the topic" from "the model refused anyway"."""

    terms: list[str] = []

    for token in re.findall(r"\w+", (question or "").lower()):
        if len(token) < 3 or token in _COVERAGE_STOPWORDS or token in terms:
            continue
        terms.append(token)

    haystack = (context or "").lower()
    found: list[str] = []
    missing: list[str] = []

    for term in terms:
        stem = term[:5] if len(term) > 5 else term
        (found if stem in haystack else missing).append(term)

    return found, missing


def _diagnose(trace: _Trace) -> tuple[str, str]:
    """Turn the trace into a most-likely cause and a next step."""

    d = trace.data
    outcome = trace.outcome
    context_chars = d.get("context_chars", 0)
    found = d.get("terms_found", [])
    missing = d.get("terms_missing", [])
    total_terms = len(found) + len(missing)
    coverage = (len(found) / total_terms) if total_terms else None
    attempts = d.get("retrieval_attempts", [])

    if outcome == "kb_fallback_from_model":
        if context_chars == 0:
            return (
                "NO_CONTEXT_REACHED_MODEL",
                f"Retrieval found nothing across {len(attempts)} attempt(s), but "
                "the model was still called (a follow-up with chat history) and "
                "returned the fallback. Check the retrieval attempts: which "
                "query ran, and did thresholds filter everything out?",
            )

        if coverage is not None and coverage < 0.5:
            return (
                "RETRIEVAL_MISS",
                f"Context was supplied ({context_chars} chars) but it lacks most "
                f"question terms (missing: {missing}). The right passage was "
                "probably not retrieved or ranked too low. Run "
                "debug_retrieval with --expect to see where it is lost.",
            )

        return (
            "MODEL_OR_PROMPT",
            f"Context was supplied ({context_chars} chars) and appears to "
            f"contain the question terms (found: {found}), yet the model (or "
            "the AI layer's own validation) returned the fallback. Read the "
            "context preview: if the answer is clearly there, look at ai.py "
            "(prompt wording, context truncation, small-model refusal).",
        )

    if outcome == "no_context":
        return (
            "RETRIEVAL_EMPTY",
            f"No context from {len(attempts)} retrieval attempt(s). The model "
            "call was skipped to save time. Check the retrieval attempts and "
            "thresholds; if the content exists, run debug_retrieval.",
        )

    if outcome == "busy":
        return (
            "BUSY",
            f"No free model slot within {CFG.queue_timeout_seconds:.0f}s "
            f"(CHAT_MAX_CONCURRENT_GENERATIONS={CFG.max_concurrent_generations}). "
            "Raise it only if the model server can really run more in parallel.",
        )

    if outcome == "rate_limited":
        return ("RATE_LIMITED", "This visitor exceeded CHAT_RATE_LIMIT_PER_MINUTE.")

    if outcome == "timeout":
        return (
            "TIMEOUT",
            f"The request exceeded CHAT_REQUEST_TIMEOUT_SECONDS="
            f"{CFG.request_timeout_seconds:.0f}s. See the stage timings to find "
            "the slow stage (usually generation or a cold model start).",
        )

    if outcome.startswith("http_") or outcome in {
        "unhandled_exception",
        "generation_unavailable",
    }:
        return (
            "PIPELINE_ERROR",
            "A stage raised or the model was unavailable. See the exception "
            "lines above for the traceback and the stage timings for how far "
            "the request got.",
        )

    if outcome == "answer_cache_hit":
        return ("OK_CACHED", "Served from the answer cache.")

    if outcome == "success":
        grounding = d.get("grounding", {})

        if grounding.get("skipped_reason") == "no_sources":
            return (
                "OK_NO_SOURCES",
                "Answered, but retrieval returned no source objects, so no "
                "links are shown.",
            )

        if grounding.get("ran") and grounding.get("validated_same") is False:
            return (
                "OK_UNGROUNDED_SOURCES_HIDDEN",
                "Answered, but the grounding check returned different text than "
                "the answer, so sources were hidden.",
            )

        return ("OK", "Answered normally.")

    return ("N/A", "Handled without retrieval or generation.")


_HIDDEN = "[hidden]"


def _log_view(data: dict) -> dict:
    """Copy of the trace data with visitor text hidden when configured."""

    if CFG.log_questions:
        return data

    view = dict(data)

    if "question" in view:
        view["question"] = _HIDDEN
    if "queries" in view:
        view["queries"] = [_HIDDEN for _ in view["queries"]]
    if "query_used" in view:
        view["query_used"] = _HIDDEN
    if "retrieval_attempts" in view:
        view["retrieval_attempts"] = [
            {**a, "query": _HIDDEN} for a in view["retrieval_attempts"]
        ]
    if isinstance(view.get("understanding"), dict):
        view["understanding"] = {
            k: (_HIDDEN if k == "rewritten_query" else v)
            for k, v in view["understanding"].items()
        }

    return view


def _debug_payload(trace: _Trace, data: dict | None = None) -> dict:
    code, explanation = _diagnose(trace)

    return {
        "request_id": trace.request_id,
        "outcome": trace.outcome,
        "diagnosis": code,
        "explanation": explanation,
        "total_seconds": round(trace.total, 3),
        "stages": {k: round(v, 3) for k, v in trace.stages.items()},
        "notes": trace.notes,
        **(trace.data if data is None else data),
    }


def _log_summary(trace: _Trace, code: str, why: str, problem: bool) -> None:
    """Detailed block: only when something went wrong (or CHAT_DEBUG=true)."""

    d = _log_view(trace.data)
    rid = trace.request_id
    level = logging.WARNING if problem else logging.INFO

    lines: list[str] = [
        "================ SUMMARY ================",
        f"outcome    : {trace.outcome}",
        f"diagnosis  : {code}",
        f"why/next   : {why}",
        f"question   : {_one_line(d.get('question'), 200)!r}",
        f"history    : in={d.get('history_in', 0)} "
        f"after_cleanup={d.get('history_out', 0)} "
        f"dropped_failed_exchanges={d.get('history_dropped_failed', 0)} "
        f"sent_to_model={d.get('model_history', 0)}",
    ]

    if "follow_up" in d:
        lines.append(
            f"follow_up  : {d['follow_up']} (reason: {d.get('follow_up_reason')}) "
            f"cache={d.get('cache', '-')}"
        )

    if d.get("understanding"):
        lines.append(f"understand : {d['understanding']}")

    for i, query in enumerate(d.get("queries", []), start=1):
        lines.append(f"query {i}    : {_one_line(query, 200)!r}")

    for i, attempt in enumerate(d.get("retrieval_attempts", []), start=1):
        lines.append(
            f"retrieval {i}: context_chars={attempt.get('context_chars')} "
            f"sources={attempt.get('source_count')} {attempt.get('seconds')}s"
            + (f" ERROR={attempt['error']}" if attempt.get("error") else "")
        )

    if "context_chars" in d:
        lines.append(
            f"context    : chars={d['context_chars']} "
            f"sources_raw={d.get('source_count_raw', 0)} "
            f"sources_cleaned={d.get('source_count_cleaned', 0)} "
            f"query_used={_one_line(d.get('query_used'), 120)!r}"
        )

        for item in d.get("source_list", []):
            lines.append(f"  source   : {item}")

        if "terms_found" in d:
            total = len(d["terms_found"]) + len(d["terms_missing"])
            lines.append(
                f"coverage   : {len(d['terms_found'])}/{total} question terms "
                f"found in context (approx.); missing={d['terms_missing']}"
            )

        if d.get("context_preview"):
            lines.append(f"ctx preview: {d['context_preview']!r}")

    generation = d.get("generation")
    if generation:
        lines.append(
            f"generation : attempts={generation.get('attempts')} "
            f"{generation.get('seconds')}s "
            f"answer_chars={generation.get('answer_chars')} "
            f"is_kb_fallback={generation.get('is_kb_fallback')} "
            f"is_unavailable={generation.get('is_unavailable')}"
        )
        if generation.get("answer_preview") is not None:
            lines.append(f"answer     : {generation['answer_preview']!r}")

    if d.get("grounding"):
        lines.append(f"grounding  : {d['grounding']}")

    if "final_answer_chars" in d:
        lines.append(
            f"final      : answer_chars={d['final_answer_chars']} "
            f"sources={d.get('final_sources', 0)}"
        )

    for note in trace.notes:
        lines.append(f"note       : {note}")

    stage_text = " ".join(f"{k}={v:.2f}s" for k, v in trace.stages.items())
    lines.append(f"stages     : {stage_text or '-'}")
    lines.append(f"total      : {trace.total:.2f}s")
    lines.append("=========================================")

    for line in lines:
        logger.log(level, "[%s] %s", rid, line)

    try:
        logger.log(
            level,
            "[%s] TRACE_JSON %s",
            rid,
            json.dumps(
                _debug_payload(trace, d), default=str, ensure_ascii=False
            ),
        )
    except Exception:
        logger.exception("[%s] TRACE_JSON: could not serialise trace", rid)


_UNANSWERED_LOCK = threading.Lock()


def _record_unanswered(trace: _Trace) -> None:
    """Append questions the knowledge base could not answer to a .jsonl file
    so the team can see which content is missing. Opt-in via env."""

    path = CFG.unanswered_log_path
    if not path:
        return

    d = _log_view(trace.data)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": trace.request_id,
        "question": d.get("question", ""),
        "outcome": trace.outcome,
        "diagnosis": trace.data.get("diagnosis"),
        "follow_up": trace.data.get("follow_up"),
        "sources_seen": [
            s for s in d.get("source_list", [])
        ][:3],
    }

    try:
        with _UNANSWERED_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("[%s] could not write unanswered log", trace.request_id)


def _log_request(trace: _Trace) -> None:
    """Called once per request on every exit path."""

    code, why = _diagnose(trace)
    trace.data["diagnosis"] = code
    problem = not _is_ok_outcome(trace.outcome)
    d = _log_view(trace.data)

    record = {
        "id": trace.request_id,
        "outcome": trace.outcome,
        "diagnosis": code,
        "ms": round(trace.total * 1000),
        "stages_ms": {k: round(v * 1000) for k, v in trace.stages.items()},
        "follow_up": trace.data.get("follow_up"),
        "cache": trace.data.get("cache"),
        "sources": trace.data.get("final_sources"),
        "answer_chars": trace.data.get("final_answer_chars"),
        "question": d.get("question"),
    }

    try:
        analytics_logger.info(
            "CHAT %s", json.dumps(record, ensure_ascii=False, default=str)
        )
    except Exception:
        logger.exception("[%s] analytics line failed", trace.request_id)

    if problem or CFG.debug_logs:
        _log_summary(trace, code, why, problem)

    if trace.outcome in _UNANSWERED_OUTCOMES:
        _record_unanswered(trace)


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
    # Only filled when CHAT_DEBUG_RESPONSE=true; omitted from JSON otherwise.
    debug: dict | None = None


# =========================================================
# Input helpers
# =========================================================

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_message(text: str | None) -> str:
    """Remove control characters, normalise whitespace, enforce length."""

    text = _CONTROL_CHARS.sub("", text or "")
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text[: CFG.max_message_chars].strip()


def _clean_history(
    history: list[ChatMessage] | None,
    stats: dict | None = None,
) -> list[dict]:
    """Keep valid user/assistant messages, trim long ones, bound the total
    size, and make sure the conversation starts with a user message."""

    cleaned: list[dict] = []

    recent = (history or [])[-CFG.max_history_turns:] if CFG.max_history_turns else []

    for message in recent:
        role = (message.role or "").strip().lower()
        content = (message.content or "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        cleaned.append(
            {"role": role, "content": content[: CFG.max_history_message_chars]}
        )

    before_drop = len(cleaned)
    cleaned = _drop_failed_exchanges(cleaned)

    if stats is not None:
        stats["dropped_failed"] = before_drop - len(cleaned)

    total = 0
    bounded: list[dict] = []

    for message in reversed(cleaned):
        total += len(message["content"])

        if total > CFG.max_history_total_chars and bounded:
            break

        bounded.append(message)

    bounded.reverse()

    while bounded and bounded[0]["role"] != "user":
        bounded.pop(0)

    return bounded


def _history_for_model(history: list[dict], follow_up: bool) -> list[dict]:
    """Chat history sent to the model.

    A standalone question needs none: old turns only add noise and slow the
    model down. A follow-up gets the last few messages.
    """

    if not follow_up or CFG.model_history_messages <= 0:
        return []

    trimmed = history[-CFG.model_history_messages:]

    while trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]

    return trimmed


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


def _build_small_talk_patterns(cfg: _Config) -> dict[str, re.Pattern]:
    """Patterns use the configured assistant/company names, never fixed ones."""

    names = sorted(
        {n for n in (_normalize(cfg.assistant_name), _normalize(cfg.company_name)) if n},
        key=len,
        reverse=True,
    )
    name_alt = "|".join(re.escape(n) for n in names)
    address_alt = "|".join(
        ["there", "team", "everyone"] + ([name_alt] if name_alt else [])
    )
    address = f"(?: (?:{address_alt}))?"
    thanks_tail = (
        f"(?: (?:a lot|so much|very much|again"
        + (f"|{name_alt}" if name_alt else "")
        + "))*"
    )

    return {
        "greeting": re.compile(
            r"(?:hi+|hii+|hello+|hey+|hiya|howdy|yo|greetings|namaste|"
            r"good (?:morning|afternoon|evening|day))" + address
        ),
        "how_are_you": re.compile(
            r"(?:(?:hi|hello|hey) )?(?:how are you(?: doing| today)?|how are u|"
            r"how's it going|hows it going|how is it going|what's up|whats up|"
            r"sup|how do you do)" + address
        ),
        "thanks": re.compile(
            r"(?:(?:ok(?:ay)?|great|awesome|perfect|nice|cool) )?"
            r"(?:thanks?|thank you|thank u|thx|ty|many thanks|much appreciated|"
            r"appreciate it)" + thanks_tail
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


SMALL_TALK_PATTERNS = _build_small_talk_patterns(CFG)


def _capabilities_text() -> str:
    points = "\n".join(f"- {point}" for point in CFG.capability_points)
    return f"I can help you with:\n{points}"


def _small_talk_reply(intent: str, has_history: bool) -> str:
    name = CFG.assistant_name
    company = CFG.company_name

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
                f"Hello, and welcome! I'm {name}. How can I help you today?",
                f"Hi there! I'm {name}. What would you like to know?",
            )
        )

    if intent == "how_are_you":
        return "I'm doing well, thank you for asking. How can I help you today?"

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
            f"I'm {name}, the AI assistant for {company}. "
            f"I answer questions using the {company} knowledge base.\n\n"
            + _capabilities_text()
        )

    if intent == "capabilities":
        return _capabilities_text() + "\n\nWhat would you like to know?"

    return CFG.empty_message_response


def _detect_small_talk(message: str, history: list[dict]) -> str | None:
    normalized = _normalize(message)

    if not normalized or len(normalized) > 60:
        return None

    for intent, pattern in SMALL_TALK_PATTERNS.items():
        if not pattern.fullmatch(normalized):
            continue

        # "ok" replies to a direct question from the assistant ("Would you
        # like more detail?") are real answers and must reach the AI.
        if intent == "acknowledge" and history:
            last = history[-1]

            if last["role"] == "assistant" and last["content"].rstrip().endswith("?"):
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


def _follow_up_reason(message: str, history: list[dict]) -> tuple[bool, str]:
    """Decide whether the message depends on earlier conversation, and say WHY
    (the reason is logged, so a wrong decision is easy to spot)."""

    if not history:
        return False, "no_history"

    normalized = _normalize(message)
    words = normalized.split()

    if not words:
        return False, "empty_after_normalize"

    if len(words) <= 4:
        return True, f"short_message({len(words)}_words)"

    if normalized.startswith(_FOLLOW_UP_STARTERS):
        return True, "starts_with_continuation_phrase"

    if len(words) <= 14:
        for word in words:
            if word in _REFERENCE_WORDS:
                return True, f"reference_word({word!r})"

    return False, "standalone_question"


def _is_follow_up(message: str, history: list[dict]) -> bool:
    return _follow_up_reason(message, history)[0]


def _contextual_query(current_message: str, history: list[dict]) -> str:
    """Current message plus recent user messages and a short snippet of the
    last assistant answer, so follow-ups keep their topic."""

    user_messages: list[str] = []

    for message in reversed(history):
        if message["role"] != "user":
            continue

        user_messages.append(message["content"])

        if len(user_messages) >= CFG.max_retrieval_user_messages:
            break

    user_messages.reverse()

    parts = list(user_messages)

    for message in reversed(history):
        if message["role"] == "assistant":
            parts.append(message["content"][: CFG.max_assistant_snippet_chars])
            break

    parts.append(current_message)

    return "\n".join(parts)


_QUESTION_STARTERS = (
    "who", "what", "when", "where", "why", "which", "how", "can", "could",
    "do", "does", "is", "are", "tell", "explain", "list", "give", "show",
    "describe", "name", "any",
)


def _looks_like_question(message: str) -> bool:
    """Used to override a wrong 'no retrieval needed' decision."""

    if "?" in message:
        return True

    words = _normalize(message).split()

    return bool(words) and words[0] in _QUESTION_STARTERS


def _usable_query(query: str | None, original: str) -> str | None:
    """Reject empty, over-long or answer-like output from the model."""

    query = (query or "").strip()

    if not query or len(query) > 200 or len(query.split()) > 30:
        return None

    if query.lower() == original.strip().lower():
        return None

    return query


async def _plan_retrieval(
    current_message: str,
    history: list[dict],
    follow_up: bool,
    trace: _Trace | None = None,
) -> tuple[list[str], bool]:
    """Returns (queries, skip_retrieval).

    * queries: ordered search queries; the first one that finds context wins.
    * skip_retrieval: True when query understanding decided the message does
      not need the knowledge base (and it does not look like a question).

    The language-model query rewrite is only used for follow-ups. Standalone
    questions are searched as typed, which saves a model call on most turns.
    """

    queries: list[str] = []
    rid = trace.request_id if trace else "-"

    if follow_up and history:
        info: dict = {"called": True}

        try:
            understood = await run_in_threadpool(
                build_retrieval_query, current_message, history
            )
            info["rewritten_query"] = (
                None if understood is None else _one_line(understood, 200)
            )
        except Exception:
            logger.exception("[%s] Query understanding failed.", rid)
            info["error"] = "build_retrieval_query raised (see traceback)"
            understood = current_message

        if understood is not None and not understood.strip():
            info["model_said_no_retrieval"] = True

            if not _looks_like_question(current_message):
                info["decision"] = "skip_retrieval"
                if trace:
                    trace.data["understanding"] = info
                return [], True

            info["decision"] = "ignored_empty_because_it_looks_like_a_question"
            understood = None

        usable = _usable_query(understood, current_message)
        info["rewritten_query_usable"] = bool(usable)

        if usable:
            queries.append(usable)

        if trace:
            trace.data["understanding"] = info

    if follow_up:
        queries.append(_contextual_query(current_message, history))
    else:
        queries.append(current_message)

        # Standalone question found nothing: maybe it relied on context.
        if history:
            queries.append(_contextual_query(current_message, history))

    seen: set[str] = set()
    unique: list[str] = []

    for query in queries:
        key = query.strip().lower()

        if key and key not in seen:
            seen.add(key)
            unique.append(query)

    return unique, False


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

        if len(cleaned) >= CFG.max_sources:
            break

    return cleaned


async def _validated_sources(
    answer: str,
    context: str,
    sources: list[dict],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> list[dict]:
    """Return sources only when the answer is supported by retrieved
    knowledge. Fallback / unavailable answers never expose sources."""

    grounding: dict = {"ran": False}

    if trace:
        trace.data["grounding"] = grounding

    skipped_reason = None

    if not answer:
        skipped_reason = "no_answer"
    elif not context:
        skipped_reason = "no_context"
    elif _is_knowledge_base_fallback(answer):
        skipped_reason = "answer_is_kb_fallback"
    elif _is_unavailable_response(answer):
        skipped_reason = "answer_is_unavailable"
    elif not sources:
        skipped_reason = "no_sources"
    elif not CFG.grounding_enabled:
        grounding["skipped_reason"] = "disabled"
        return sources

    if skipped_reason:
        grounding["skipped_reason"] = skipped_reason
        logger.debug("[%s] GROUNDING: skipped (%s)", request_id, skipped_reason)
        return []

    started = time.perf_counter()

    try:
        validated = await run_in_threadpool(
            validate_answer, answer=answer, context=context
        )
    except Exception:
        logger.exception("[%s] SOURCE VALIDATION: FAILED", request_id)
        grounding["ran"] = True
        grounding["error"] = "validate_answer raised (see traceback)"
        return []

    grounding["ran"] = True
    grounding["seconds"] = round(time.perf_counter() - started, 3)

    validated_text = (validated or "").strip()
    same = bool(validated) and validated_text == answer.strip()

    grounding["validated_same"] = same
    grounding["validator_returned_empty"] = not validated_text

    if not same:
        grounding["validated_preview"] = _one_line(
            validated_text, CFG.debug_answer_chars
        )
        grounding["validated_chars"] = len(validated_text)
        grounding["answer_chars"] = len(answer.strip())
        logger.warning(
            "[%s] GROUNDING: validator output differs from answer "
            "(validated_chars=%d answer_chars=%d) -> sources hidden.",
            request_id,
            len(validated_text),
            len(answer.strip()),
        )
        return []

    return sources


async def _generate_with_retry(
    question: str,
    context: str,
    history: list[dict],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> str:
    """Call the AI layer, retrying on transient failures.
    Returns UNAVAILABLE_ANSWER if every attempt fails."""

    answer = UNAVAILABLE_ANSWER
    attempts_made = 0
    started = time.perf_counter()

    logger.debug(
        "[%s] GENERATION: question_chars=%d context_chars=%d history_messages=%d",
        request_id, len(question), len(context), len(history),
    )

    for attempt in range(1, CFG.generation_attempts + 1):
        attempts_made = attempt
        attempt_started = time.perf_counter()

        try:
            answer = await generate_answer(
                question=question, context=context, history=history
            )
            logger.debug(
                "[%s] GENERATION: attempt %d/%d returned in %.3fs chars=%d",
                request_id, attempt, CFG.generation_attempts,
                time.perf_counter() - attempt_started, len(answer or ""),
            )
        except Exception:
            logger.exception(
                "[%s] GENERATION: attempt %d/%d FAILED after %.3fs.",
                request_id, attempt, CFG.generation_attempts,
                time.perf_counter() - attempt_started,
            )
            answer = UNAVAILABLE_ANSWER

        if answer and not _is_unavailable_response(answer):
            break
    else:
        answer = UNAVAILABLE_ANSWER

    if not answer or _is_unavailable_response(answer):
        answer = UNAVAILABLE_ANSWER

    if trace:
        trace.data["generation"] = {
            "attempts": attempts_made,
            "seconds": round(time.perf_counter() - started, 3),
            "answer_chars": len(answer or ""),
            "is_kb_fallback": _is_knowledge_base_fallback(answer or ""),
            "is_unavailable": _is_unavailable_response(answer or ""),
            "answer_preview": (
                _one_line(answer, CFG.debug_answer_chars)
                if CFG.debug_answer_chars > 0
                else None
            ),
        }

    return answer


# One slot per concurrent model call. Created on first use inside the running
# event loop. Visitors beyond the limit wait briefly, then get a friendly
# "busy" message instead of a request that hangs.
_generation_gate: asyncio.Semaphore | None = None


def _get_gate() -> asyncio.Semaphore:
    global _generation_gate

    if _generation_gate is None:
        _generation_gate = asyncio.Semaphore(CFG.max_concurrent_generations)

    return _generation_gate


async def _generate_gated(
    question: str,
    context: str,
    history: list[dict],
    request_id: str,
    trace: _Trace,
) -> str | None:
    """Returns the answer, or None when no model slot became free in time."""

    gate = _get_gate()

    try:
        with trace.stage("queue"):
            await asyncio.wait_for(
                gate.acquire(), timeout=CFG.queue_timeout_seconds
            )
    except asyncio.TimeoutError:
        return None

    try:
        with trace.stage("generation"):
            return await _generate_with_retry(
                question=question,
                context=context,
                history=history,
                request_id=request_id,
                trace=trace,
            )
    finally:
        gate.release()


async def _retrieve(
    queries: list[str],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> tuple[str, list[dict], str]:
    """Try each query in order and return the first one that finds context:
    (context, sources, query_used)."""

    last_query = queries[0] if queries else ""
    attempts: list[dict] = []

    if trace:
        trace.data["retrieval_attempts"] = attempts

    for index, query in enumerate(queries, start=1):
        last_query = query
        attempt: dict = {"query": _one_line(query, 200)}
        attempts.append(attempt)
        started = time.perf_counter()

        try:
            result = await run_in_threadpool(build_context, query)
        except Exception as exc:
            attempt["seconds"] = round(time.perf_counter() - started, 3)
            attempt["error"] = f"{type(exc).__name__}: {_one_line(str(exc), 160)}"
            raise

        attempt["seconds"] = round(time.perf_counter() - started, 3)
        result = result or {}

        context = result.get("context", "") or ""
        sources = result.get("sources", []) or []

        attempt["context_chars"] = len(context)
        attempt["source_count"] = len(sources)

        logger.debug(
            "[%s] RETRIEVAL: query %d/%d %.3fs context_chars=%d sources=%d",
            request_id, index, len(queries), attempt["seconds"],
            len(context), len(sources),
        )

        if context.strip():
            return context, sources, query

    return "", [], last_query


# =========================================================
# Answer cache (repeat standalone questions are answered instantly)
# =========================================================

_ANSWER_CACHE_MAX = 256
_ANSWER_CACHE_LOCK = threading.Lock()
_ANSWER_CACHE: OrderedDict[str, tuple[float, str, list[dict]]] = OrderedDict()


def _answer_cache_get(key: str) -> tuple[str, list[dict]] | None:
    ttl = CFG.answer_cache_ttl_seconds

    if ttl <= 0 or not key:
        return None

    with _ANSWER_CACHE_LOCK:
        entry = _ANSWER_CACHE.get(key)

        if entry is None:
            return None

        stored_at, answer, sources = entry

        if time.monotonic() - stored_at > ttl:
            _ANSWER_CACHE.pop(key, None)
            return None

        _ANSWER_CACHE.move_to_end(key)
        return answer, [dict(s) for s in sources]


def _answer_cache_put(key: str, answer: str, sources: list[dict]) -> None:
    if CFG.answer_cache_ttl_seconds <= 0 or not key:
        return

    with _ANSWER_CACHE_LOCK:
        _ANSWER_CACHE[key] = (time.monotonic(), answer, [dict(s) for s in sources])
        _ANSWER_CACHE.move_to_end(key)

        while len(_ANSWER_CACHE) > _ANSWER_CACHE_MAX:
            _ANSWER_CACHE.popitem(last=False)


def clear_answer_cache() -> None:
    """Call after re-ingesting the knowledge base so answers refresh at once."""
    with _ANSWER_CACHE_LOCK:
        _ANSWER_CACHE.clear()


# =========================================================
# Per-visitor rate limit (optional)
# =========================================================

_RATE_LOCK = threading.Lock()
_RATE_HITS: dict[str, deque] = {}


def _client_key(http_request: Request) -> str:
    if CFG.trust_proxy:
        forwarded = (http_request.headers.get("x-forwarded-for") or "").strip()
        if forwarded:
            return forwarded.split(",")[0].strip()

    client = getattr(http_request, "client", None)
    return getattr(client, "host", None) or "unknown"


def _rate_limited(key: str) -> bool:
    limit = CFG.rate_limit_per_minute

    if limit <= 0:
        return False

    now = time.monotonic()
    cutoff = now - 60.0

    with _RATE_LOCK:
        hits = _RATE_HITS.setdefault(key, deque())

        while hits and hits[0] < cutoff:
            hits.popleft()

        if len(hits) >= limit:
            return True

        hits.append(now)

        if len(_RATE_HITS) > 5000:
            for stale in [k for k, v in _RATE_HITS.items() if not v or v[-1] < cutoff]:
                _RATE_HITS.pop(stale, None)

    return False


# =========================================================
# The pipeline
# =========================================================

async def _run_chat(request: ChatRequest, trace: _Trace) -> ChatResponse:
    """Every return/raise sets trace.outcome first."""

    request_id = trace.request_id

    # ---- 1. Input --------------------------------------------------
    try:
        with trace.stage("input"):
            current_message = _sanitize_message(request.message)

            trace.data["question"] = current_message
            trace.data["history_in"] = len(request.history or [])

            if not current_message:
                trace.outcome = "empty_message"
                return ChatResponse(answer=CFG.empty_message_response, sources=[])

            if not re.search(r"\w", current_message):
                trace.outcome = "symbol_only_message"
                return ChatResponse(
                    answer=CFG.unclear_message_response, sources=[]
                )

            history_stats: dict = {}
            history = _clean_history(request.history, stats=history_stats)

            trace.data["history_out"] = len(history)
            trace.data["history_dropped_failed"] = history_stats.get(
                "dropped_failed", 0
            )
    except Exception:
        logger.exception("[%s] INPUT: FAILED", request_id)
        trace.outcome = "http_500_input_failed"
        raise HTTPException(status_code=500, detail=UNAVAILABLE_ANSWER)

    # ---- 2. Small talk ---------------------------------------------
    try:
        with trace.stage("small_talk"):
            intent = _detect_small_talk(current_message, history)

        if intent:
            trace.outcome = f"small_talk:{intent}"
            return ChatResponse(
                answer=_small_talk_reply(intent, bool(history)), sources=[]
            )
    except Exception:
        logger.exception("[%s] SMALL_TALK: FAILED", request_id)
        trace.outcome = "http_500_small_talk_failed"
        raise HTTPException(status_code=500, detail=UNAVAILABLE_ANSWER)

    # ---- 3. Follow-up decision + answer cache ----------------------
    follow_up, follow_up_why = _follow_up_reason(current_message, history)

    trace.data["follow_up"] = follow_up
    trace.data["follow_up_reason"] = follow_up_why
    cache_key = "" if follow_up else _normalize(current_message)

    if follow_up:
        trace.data["cache"] = "skip(follow_up)"
    else:
        with trace.stage("cache"):
            cached = _answer_cache_get(cache_key)

        if cached is not None:
            answer, cached_sources = cached
            trace.data["cache"] = "hit"
            trace.data["final_answer_chars"] = len(answer)
            trace.data["final_sources"] = len(cached_sources)
            trace.outcome = "answer_cache_hit"
            return ChatResponse(answer=answer, sources=cached_sources)

        trace.data["cache"] = "miss"

    # ---- 4. Query planning ------------------------------------------
    try:
        with trace.stage("query_plan"):
            queries, skip_retrieval = await _plan_retrieval(
                current_message, history, follow_up, trace=trace
            )
            trace.data["queries"] = queries
    except Exception:
        logger.exception("[%s] QUERY_PLAN: FAILED", request_id)
        trace.outcome = "http_503_query_plan_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    if skip_retrieval:
        trace.outcome = "retrieval_skipped"
        return ChatResponse(
            answer=(
                "Thanks for letting me know. "
                f"What would you like to know about {CFG.company_name}?"
            ),
            sources=[],
        )

    # ---- 5. Retrieval ------------------------------------------------
    try:
        with trace.stage("retrieval"):
            context, retrieved_sources, query_used = await _retrieve(
                queries, request_id=request_id, trace=trace
            )

        found, missing = _term_coverage(current_message, context)

        trace.data.update(
            {
                "context_chars": len(context),
                "source_count_raw": len(retrieved_sources),
                "query_used": query_used,
                "terms_found": found,
                "terms_missing": missing,
                "source_list": [
                    _describe_item(s)
                    for s in retrieved_sources[:8]
                    if isinstance(s, dict)
                ],
                "context_preview": (
                    _one_line(context, CFG.debug_context_chars)
                    if CFG.debug_context_chars > 0 and context.strip()
                    else None
                ),
            }
        )
    except Exception:
        logger.exception("[%s] RAG: FAILED", request_id)
        trace.outcome = "http_503_retrieval_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    # ---- 6. Nothing found: skip the model, it cannot help ---------
    if not context.strip():
        if not history or not follow_up:
            trace.note(
                "No relevant context for a standalone question; the model "
                "call was skipped (it could only have returned the fallback)."
            )
            trace.data["final_answer_chars"] = len(_friendly_fallback())
            trace.data["final_sources"] = 0
            trace.outcome = "no_context"
            return ChatResponse(answer=_friendly_fallback(), sources=[])

        # A follow-up may still be answerable from the conversation itself.
        trace.note(
            "No context was retrieved for a follow-up; calling the model "
            "with the conversation only."
        )

    # ---- 7. Generation ----------------------------------------------
    model_history = _history_for_model(history, follow_up)
    trace.data["model_history"] = len(model_history)

    try:
        answer = await _generate_gated(
            question=current_message,
            context=context,
            history=model_history,
            request_id=request_id,
            trace=trace,
        )
    except Exception:
        logger.exception("[%s] GENERATION_PIPELINE: FAILED", request_id)
        trace.outcome = "http_503_generation_pipeline_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    if answer is None:
        trace.outcome = "busy"
        return ChatResponse(answer=CFG.busy_response, sources=[])

    if _is_unavailable_response(answer):
        logger.error("[%s] GENERATION: unavailable response", request_id)
        trace.outcome = "generation_unavailable"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    answer = (answer or "").strip()

    # ---- 8. Source validation / grounding -----------------------------
    try:
        with trace.stage("grounding"):
            cleaned_sources = _clean_sources(retrieved_sources)
            trace.data["source_count_cleaned"] = len(cleaned_sources)

            sources = await _validated_sources(
                answer=answer,
                context=context,
                sources=cleaned_sources,
                request_id=request_id,
                trace=trace,
            )
    except Exception:
        logger.exception("[%s] SOURCE_PIPELINE: FAILED", request_id)
        sources = []

    # ---- 9. Final decision ---------------------------------------------
    if not answer or _is_knowledge_base_fallback(answer):
        answer = _friendly_fallback()
        sources = []
        trace.outcome = "kb_fallback_from_model"
    else:
        trace.outcome = "success"

        # Only cache clean, standalone answers whose retrieval used the
        # question as typed (not the conversation-augmented query).
        if cache_key and query_used == current_message:
            _answer_cache_put(cache_key, answer, sources)

    trace.data["final_answer_chars"] = len(answer)
    trace.data["final_sources"] = len(sources)

    return ChatResponse(answer=answer, sources=sources)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Main Ask Alcor pipeline.

    Internal exceptions are logged with a request id and full traceback.
    Visitors receive only the safe public error/fallback response.
    """

    trace = _Trace(uuid.uuid4().hex[:8])

    if _rate_limited(_client_key(http_request)):
        trace.outcome = "rate_limited"
        trace.data["question"] = _sanitize_message(request.message)
        _log_request(trace)
        return ChatResponse(answer=CFG.rate_limit_response, sources=[])

    timeout = CFG.request_timeout_seconds or None

    try:
        response = await asyncio.wait_for(_run_chat(request, trace), timeout=timeout)
    except asyncio.TimeoutError:
        trace.outcome = "timeout"
        logger.error("[%s] request exceeded %.0fs", trace.request_id, timeout or 0)
        _log_request(trace)
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)
    except HTTPException as exc:
        if trace.outcome == "unknown":
            trace.outcome = f"http_{exc.status_code}"
        _log_request(trace)
        raise
    except Exception:
        trace.outcome = "unhandled_exception"
        logger.exception("[%s] UNHANDLED EXCEPTION", trace.request_id)
        _log_request(trace)
        raise

    _log_request(trace)

    if CFG.debug_response:
        response.debug = _debug_payload(trace)

    return response


# =========================================================
# Start-up warm-up
# =========================================================

async def warm_up() -> dict:
    """Load the search models and wake the language model so the first
    visitor does not wait for a cold start. Never raises.

    Example (main.py):
        @app.on_event("startup")
        async def _warm():
            asyncio.create_task(chat_router_module.warm_up())
    """

    started = time.perf_counter()

    async def _search() -> None:
        try:
            await run_in_threadpool(build_context, "company overview")
        except Exception:
            logger.exception("warm-up: retrieval failed")

    async def _model() -> None:
        try:
            await generate_answer(question="Hello", context="Warm-up.", history=[])
        except Exception:
            logger.exception("warm-up: model call failed")

    await asyncio.gather(_search(), _model())

    seconds = round(time.perf_counter() - started, 2)
    logger.info("chat warm-up finished in %.2fs", seconds)
    return {"seconds": seconds}