from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.services.ai import (
    FALLBACK_ANSWER,
    UNAVAILABLE_ANSWER,
    generate_answer,
)
from backend.app.services.grounding import validate_answer
from backend.app.services.query_understanding import build_retrieval_query
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

# ---- Debugging (all optional, all via env) ---------------------------------
# How many characters of retrieved context / model answer to show in logs.
# Set to 0 to hide them (e.g. if you do not want content in production logs).
DEBUG_CONTEXT_CHARS = int(os.getenv("CHAT_DEBUG_CONTEXT_CHARS", "600"))
DEBUG_ANSWER_CHARS = int(os.getenv("CHAT_DEBUG_ANSWER_CHARS", "300"))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# When true, the JSON response also carries a "debug" object (diagnosis,
# stage timings, queries, sources...). Use only in development/testing and
# turn it off in production.
DEBUG_RESPONSE = _env_bool("CHAT_DEBUG_RESPONSE", False)

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
# Debug trace
#
# One _Trace object follows each request. Every stage records what it saw
# and how long it took. When the request ends (on ANY path, including
# errors) a single summary block is logged with a diagnosis of what most
# likely went wrong, plus one machine-readable TRACE_JSON line you can grep
# and collect across many failing requests.
# =========================================================

# Outcomes that are normal and do not need a WARNING-level summary.
_OK_OUTCOMES = {
    "success",
    "empty_message",
    "symbol_only_message",
    "retrieval_skipped",
}


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
        logger.info("[%s] NOTE: %s", self.request_id, text)

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started


def _one_line(text: str | None, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _describe_item(item: dict) -> str:
    """Compact 'key=value' view of a source/chunk dict (scores, title, url)."""

    parts: list[str] = []

    for key, value in item.items():
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        elif key in {"title", "url", "source", "heading"} and value:
            parts.append(f"{key}={_one_line(str(value), 90)}")

    return " ".join(parts) or "(no scalar fields)"


def _describe_extra(result: dict) -> list[str]:
    """
    Anything build_context() returns besides 'context' and 'sources'
    (chunk lists, scores, thresholds...) is shown too, so the retrieval layer
    can explain itself without this file knowing its exact shape.
    """

    lines: list[str] = []

    for key, value in result.items():
        if key in {"context", "sources"}:
            continue

        if isinstance(value, list) and value and isinstance(value[0], dict):
            for item in value[:5]:
                lines.append(f"{key}: {_describe_item(item)}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}={_one_line(str(value), 120)}")

        if len(lines) >= 12:
            break

    return lines


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
    """
    Approximate check: which meaningful words of the question appear in the
    retrieved context? Separates "retrieval missed the topic" (terms absent)
    from "the model refused even though the topic is there" (terms present).
    Uses a 5-letter prefix so plurals/stems still match.
    """

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
                "the model was still called with EMPTY context (the "
                "'no context' short-circuit only applies when there is no "
                "chat history). The model then returned the fallback. Look at "
                "the RETRIEVAL lines and retriever timing: which query was "
                "used, and did thresholds filter everything out?",
            )

        if coverage is not None and coverage < 0.5:
            return (
                "RETRIEVAL_MISS",
                f"Context was supplied ({context_chars} chars) but it does not "
                f"contain most question terms (missing: {missing}). The right "
                "chunk was probably not retrieved or ranked too low. Compare the "
                "source titles/URLs above with where the answer lives on the "
                "site; check thresholds and the query that was actually used.",
            )

        return (
            "MODEL_OR_PROMPT",
            f"Context was supplied ({context_chars} chars) and it appears to "
            f"contain the question terms (found: {found}). The model "
            "(or the AI layer's own validation inside generate_answer) still "
            "returned the fallback. Read the context preview: if the answer is "
            "clearly in it, the cause is in ai.py - prompt wording, context "
            "truncation, a small model refusing, or an internal grounding step. "
            "Add logging there (see notes) to see which branch returned the "
            "fallback.",
        )

    if outcome == "no_context_first_turn":
        return (
            "RETRIEVAL_EMPTY",
            f"No context from {len(attempts)} retrieval attempt(s) on a first "
            "turn, so the friendly fallback was returned without calling the "
            "model. Check the RETRIEVAL lines and retriever thresholds.",
        )

    if outcome.startswith("http_") or outcome in {
        "unhandled_exception",
        "generation_unavailable",
    }:
        return (
            "PIPELINE_ERROR",
            "A stage raised or the model was unavailable. See the "
            "FAILED/exception lines above for the traceback, and the "
            "'stages' timing line to see how far the request got.",
        )

    if outcome == "success":
        grounding = d.get("grounding", {})

        if grounding.get("skipped_reason") == "no_sources":
            return (
                "OK_NO_SOURCES",
                "Answered, but retrieval returned no source objects, so no "
                "links are shown and grounding was skipped.",
            )

        if grounding.get("ran") and grounding.get("validated_same") is False:
            return (
                "OK_UNGROUNDED_SOURCES_HIDDEN",
                "Answered, but the grounding validator returned different text "
                "than the answer, so sources were hidden. See the GROUNDING "
                "lines for what the validator returned.",
            )

        return ("OK", "Answered normally.")

    return ("N/A", "Handled without retrieval or generation.")


def _debug_payload(trace: _Trace) -> dict:
    code, explanation = _diagnose(trace)

    return {
        "request_id": trace.request_id,
        "outcome": trace.outcome,
        "diagnosis": code,
        "explanation": explanation,
        "total_seconds": round(trace.total, 3),
        "stages": {k: round(v, 3) for k, v in trace.stages.items()},
        "notes": trace.notes,
        **trace.data,
    }


def _log_summary(trace: _Trace) -> None:
    """One readable block per request, on every exit path."""

    d = trace.data
    rid = trace.request_id
    code, explanation = _diagnose(trace)
    trace.data["diagnosis"] = code

    is_ok = trace.outcome in _OK_OUTCOMES or trace.outcome.startswith("small_talk")
    level = logging.INFO if is_ok else logging.WARNING

    lines: list[str] = [
        "================ SUMMARY ================",
        f"outcome    : {trace.outcome}",
        f"diagnosis  : {code}",
        f"why/next   : {explanation}",
        f"question   : {_one_line(d.get('question'), 200)!r}",
        f"history    : in={d.get('history_in', 0)} "
        f"after_cleanup={d.get('history_out', 0)} "
        f"dropped_failed_exchanges={d.get('history_dropped_failed', 0)}",
    ]

    if "follow_up" in d:
        lines.append(
            f"follow_up  : {d['follow_up']} (reason: {d.get('follow_up_reason')})"
        )

    understanding = d.get("understanding")
    if understanding:
        lines.append(f"understand : {understanding}")

    for i, query in enumerate(d.get("queries", []), start=1):
        lines.append(f"query {i}    : {_one_line(query, 200)!r}")

    for i, attempt in enumerate(d.get("retrieval_attempts", []), start=1):
        lines.append(
            f"retrieval {i}: context_chars={attempt.get('context_chars')} "
            f"sources={attempt.get('source_count')} "
            f"{attempt.get('seconds')}s"
            + (f" ERROR={attempt['error']}" if attempt.get("error") else "")
        )

    if "context_chars" in d:
        lines.append(
            f"context    : chars={d['context_chars']} "
            f"sources_raw={d.get('source_count_raw', 0)} "
            f"sources_cleaned={d.get('source_count_cleaned', 0)} "
            f"query_used={_one_line(d.get('query_used'), 120)!r}"
        )

        for title_url in d.get("source_list", []):
            lines.append(f"  source   : {title_url}")

        if "terms_found" in d:
            total_terms = len(d["terms_found"]) + len(d["terms_missing"])
            lines.append(
                f"coverage   : {len(d['terms_found'])}/{total_terms} question "
                f"terms found in context (approx.); "
                f"missing={d['terms_missing']}"
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

    grounding = d.get("grounding")
    if grounding:
        lines.append(f"grounding  : {grounding}")

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
            json.dumps(_debug_payload(trace), default=str, ensure_ascii=False),
        )
    except Exception:
        logger.exception("[%s] TRACE_JSON: could not serialise trace", rid)


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

    return text[:MAX_MESSAGE_CHARS].strip()


def _clean_history(
    history: list[ChatMessage] | None,
    stats: dict | None = None,
) -> list[dict]:
    """
    Keep only valid user/assistant messages, trim very long ones, bound the
    total size, and make sure the conversation starts with a user message.

    ``stats`` (optional) receives debugging counters.
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

    before_drop = len(cleaned)
    cleaned = _drop_failed_exchanges(cleaned)

    if stats is not None:
        stats["dropped_failed"] = before_drop - len(cleaned)

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


def _follow_up_reason(message: str, history: list[dict]) -> tuple[bool, str]:
    """
    Decide whether the message depends on earlier conversation, and say WHY
    (the reason is logged, so a wrong follow-up decision is easy to spot).

    Short messages, messages that start with a continuation phrase, and
    messages containing reference words ("he", "that", "their"...) are
    treated as follow-ups. Everything else is treated as a new,
    self-contained question so old topics do not pollute retrieval.
    """

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
    """
    Returns (queries, skip_retrieval).

    * queries: ordered search queries; the first one that finds context wins.
    * skip_retrieval: True when query understanding decided the message does
      not need the knowledge base (and it does not look like a question).

    The language-model query understanding (build_retrieval_query) is only
    called for follow-ups. Self-contained questions are searched as typed,
    which saves a model call on most turns.
    """

    queries: list[str] = []
    rid = trace.request_id if trace else "-"

    if follow_up and history:
        info: dict = {"called": True}

        try:
            understood = await run_in_threadpool(
                build_retrieval_query,
                current_message,
                history,
            )
            info["rewritten_query"] = (
                None if understood is None else _one_line(understood, 200)
            )
        except Exception:
            logger.exception("[%s] Query understanding failed.", rid)
            info["error"] = "build_retrieval_query raised (see traceback)"
            understood = current_message

        if understood is not None and not understood.strip():
            # Model says: no retrieval needed.
            info["model_said_no_retrieval"] = True

            if not _looks_like_question(current_message):
                info["decision"] = "skip_retrieval"
                if trace:
                    trace.data["understanding"] = info
                return [], True

            info["decision"] = "ignored_empty_because_it_looks_like_a_question"
            understood = None  # looks like a question: do not trust "empty"

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

    # Remove duplicates, keep order.
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

        if len(cleaned) >= MAX_SOURCES:
            break

    return cleaned


async def _validated_sources(
    answer: str,
    context: str,
    sources: list[dict],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> list[dict]:
    """
    Return sources only when the answer is supported by retrieved knowledge.
    Fallback / unavailable answers never expose sources.
    """

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

    if skipped_reason:
        grounding["skipped_reason"] = skipped_reason
        logger.info("[%s] GROUNDING: skipped (%s)", request_id, skipped_reason)
        return []

    started = time.perf_counter()

    try:
        validated = await run_in_threadpool(
            validate_answer,
            answer=answer,
            context=context,
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
            validated_text, DEBUG_ANSWER_CHARS
        )
        grounding["validated_chars"] = len(validated_text)
        grounding["answer_chars"] = len(answer.strip())
        logger.warning(
            "[%s] GROUNDING: validator output differs from answer "
            "(validated_chars=%d answer_chars=%d) -> sources hidden. "
            "validated=%r",
            request_id,
            len(validated_text),
            len(answer.strip()),
            _one_line(validated_text, DEBUG_ANSWER_CHARS),
        )
        return []

    logger.info(
        "[%s] GROUNDING: validated OK in %.3fs",
        request_id,
        grounding["seconds"],
    )

    return sources


async def _generate_with_retry(
    question: str,
    context: str,
    history: list[dict],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> str:
    """
    Call the AI layer, retrying briefly on transient failures.
    Returns UNAVAILABLE_ANSWER if every attempt fails.
    """

    answer = UNAVAILABLE_ANSWER
    attempts_made = 0
    started = time.perf_counter()

    logger.info(
        "[%s] GENERATION: input question_chars=%d context_chars=%d "
        "history_messages=%d",
        request_id,
        len(question),
        len(context),
        len(history),
    )

    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        attempts_made = attempt
        attempt_started = time.perf_counter()

        try:
            logger.info(
                "[%s] GENERATION: attempt %d/%d started",
                request_id,
                attempt,
                GENERATION_ATTEMPTS,
            )
            answer = await generate_answer(
                question=question,
                context=context,
                history=history,
            )
            logger.info(
                "[%s] GENERATION: attempt %d/%d returned in %.3fs "
                "chars=%d is_kb_fallback=%s preview=%r",
                request_id,
                attempt,
                GENERATION_ATTEMPTS,
                time.perf_counter() - attempt_started,
                len(answer or ""),
                _is_knowledge_base_fallback(answer or ""),
                _one_line(answer, DEBUG_ANSWER_CHARS),
            )
        except Exception:
            # NOTE: the original call passed `attempt` twice, which made the
            # logging module raise a formatting error and hide the traceback.
            logger.exception(
                "[%s] GENERATION: attempt %d/%d FAILED after %.3fs.",
                request_id,
                attempt,
                GENERATION_ATTEMPTS,
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
                _one_line(answer, DEBUG_ANSWER_CHARS)
                if DEBUG_ANSWER_CHARS > 0
                else None
            ),
        }

    return answer


async def _retrieve(
    queries: list[str],
    request_id: str = "-",
    trace: _Trace | None = None,
) -> tuple[str, list[dict], str]:
    """
    Try each query in order and return the first one that finds context:
    (context, sources, query_used).
    """

    last_query = queries[0] if queries else ""
    attempts: list[dict] = []

    if trace:
        trace.data["retrieval_attempts"] = attempts

    logger.info("[%s] RETRIEVAL: started queries=%d", request_id, len(queries))

    for index, query in enumerate(queries, start=1):
        last_query = query
        logger.info(
            "[%s] RETRIEVAL: query %d/%d=%r",
            request_id,
            index,
            len(queries),
            query[:300],
        )

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
        attempt["result_keys"] = sorted(result.keys())

        logger.info(
            "[%s] RETRIEVAL: query %d/%d finished in %.3fs "
            "context_chars=%d sources=%d result_keys=%s",
            request_id,
            index,
            len(queries),
            attempt["seconds"],
            len(context),
            len(sources),
            attempt["result_keys"],
        )

        for rank, source in enumerate(sources[:8], start=1):
            if isinstance(source, dict):
                logger.info(
                    "[%s] RETRIEVAL:   source %d: %s",
                    request_id,
                    rank,
                    _describe_item(source),
                )

        for line in _describe_extra(result):
            logger.info("[%s] RETRIEVAL:   %s", request_id, line)

        if DEBUG_CONTEXT_CHARS > 0 and context.strip():
            logger.info(
                "[%s] RETRIEVAL:   context_preview=%r",
                request_id,
                _one_line(context, DEBUG_CONTEXT_CHARS),
            )

        if context.strip():
            logger.info(
                "[%s] RETRIEVAL: SUCCESS query %d/%d",
                request_id,
                index,
                len(queries),
            )
            return context, sources, query

        logger.warning(
            "[%s] RETRIEVAL: query %d/%d returned EMPTY context",
            request_id,
            index,
            len(queries),
        )

    logger.warning("[%s] RETRIEVAL: no context from any query", request_id)
    return "", [], last_query


# =========================================================
# Chat endpoint
# =========================================================

async def _run_chat(request: ChatRequest, trace: _Trace) -> ChatResponse:
    """The pipeline itself. Every return/raise sets trace.outcome first."""

    request_id = trace.request_id
    started = trace.started

    # ---- 1. Input --------------------------------------------------
    try:
        with trace.stage("input"):
            current_message = _sanitize_message(request.message)

            logger.info(
                "[%s] INPUT: raw_chars=%d sanitized_chars=%d history=%d",
                request_id,
                len(request.message or ""),
                len(current_message),
                len(request.history or []),
            )
            logger.info(
                "[%s] INPUT: question=%r",
                request_id,
                _one_line(current_message, 300),
            )

            trace.data["question"] = current_message
            trace.data["history_in"] = len(request.history or [])

            if not current_message:
                logger.info("[%s] INPUT: empty message", request_id)
                trace.outcome = "empty_message"
                return ChatResponse(answer=EMPTY_MESSAGE_RESPONSE, sources=[])

            if not re.search(r"\w", current_message):
                logger.info(
                    "[%s] INPUT: symbol/punctuation-only message", request_id
                )
                trace.outcome = "symbol_only_message"
                return ChatResponse(answer=UNCLEAR_MESSAGE_RESPONSE, sources=[])

            history_stats: dict = {}
            history = _clean_history(request.history, stats=history_stats)

            trace.data["history_out"] = len(history)
            trace.data["history_dropped_failed"] = history_stats.get(
                "dropped_failed", 0
            )

            logger.info(
                "[%s] INPUT: accepted history_after_cleanup=%d "
                "dropped_failed_exchanges=%d",
                request_id,
                len(history),
                history_stats.get("dropped_failed", 0),
            )
    except Exception:
        logger.exception("[%s] INPUT: FAILED", request_id)
        trace.outcome = "http_500_input_failed"
        raise HTTPException(status_code=500, detail=UNAVAILABLE_ANSWER)

    # ---- 2. Small talk ---------------------------------------------
    try:
        with trace.stage("small_talk"):
            intent = _detect_small_talk(current_message, history)
            logger.info("[%s] SMALL_TALK: intent=%r", request_id, intent)

        if intent:
            answer = _small_talk_reply(intent, bool(history))
            logger.info(
                "[%s] FINAL: small_talk total=%.3fs",
                request_id,
                time.perf_counter() - started,
            )
            logger.info("==============================================")
            trace.outcome = f"small_talk:{intent}"
            return ChatResponse(answer=answer, sources=[])
    except Exception:
        logger.exception("[%s] SMALL_TALK: FAILED", request_id)
        trace.outcome = "http_500_small_talk_failed"
        raise HTTPException(status_code=500, detail=UNAVAILABLE_ANSWER)

    # ---- 3. Query planning / query understanding ------------------
    try:
        with trace.stage("query_plan"):
            follow_up, follow_up_why = _follow_up_reason(current_message, history)

            trace.data["follow_up"] = follow_up
            trace.data["follow_up_reason"] = follow_up_why

            logger.info(
                "[%s] QUERY_PLAN: follow_up=%s reason=%s",
                request_id,
                follow_up,
                follow_up_why,
            )

            queries, skip_retrieval = await _plan_retrieval(
                current_message,
                history,
                follow_up,
                trace=trace,
            )

            trace.data["queries"] = queries

        logger.info(
            "[%s] QUERY_PLAN: queries=%d skip_retrieval=%s understanding=%s",
            request_id,
            len(queries),
            skip_retrieval,
            trace.data.get("understanding"),
        )

        for index, query in enumerate(queries, start=1):
            logger.info(
                "[%s] QUERY_PLAN: query_%d=%r",
                request_id,
                index,
                query[:300],
            )
    except Exception:
        logger.exception("[%s] QUERY_PLAN: FAILED", request_id)
        trace.outcome = "http_503_query_plan_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    if skip_retrieval:
        logger.info("[%s] FINAL: retrieval skipped", request_id)
        logger.info("==============================================")
        trace.outcome = "retrieval_skipped"
        return ChatResponse(
            answer=(
                "Thanks for letting me know. "
                f"What would you like to know about {COMPANY_NAME}?"
            ),
            sources=[],
        )

    # ---- 4. Retrieval / RAG ----------------------------------------
    try:
        with trace.stage("retrieval"):
            context, retrieved_sources, query_used = await _retrieve(
                queries,
                request_id=request_id,
                trace=trace,
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
                    _one_line(context, DEBUG_CONTEXT_CHARS)
                    if DEBUG_CONTEXT_CHARS > 0 and context.strip()
                    else None
                ),
            }
        )

        logger.info(
            "[%s] RAG: context_chars=%d sources=%d query_used=%r",
            request_id,
            len(context),
            len(retrieved_sources),
            query_used[:300],
        )
        logger.info(
            "[%s] RAG: question-term coverage %d/%d (approx.) found=%s "
            "missing=%s",
            request_id,
            len(found),
            len(found) + len(missing),
            found,
            missing,
        )
    except Exception:
        logger.exception("[%s] RAG: FAILED", request_id)
        logger.info(
            "[%s] FINAL: unavailable_after_retrieval total=%.3fs",
            request_id,
            time.perf_counter() - started,
        )
        logger.info("==============================================")
        trace.outcome = "http_503_retrieval_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    if not context.strip() and not history:
        logger.warning("[%s] RAG: NO CONTEXT -> fallback", request_id)
        logger.info(
            "[%s] FINAL: no_context total=%.3fs",
            request_id,
            time.perf_counter() - started,
        )
        logger.info("==============================================")
        trace.outcome = "no_context_first_turn"
        return ChatResponse(answer=FRIENDLY_FALLBACK, sources=[])

    if not context.strip():
        # This path is easy to miss: there is history, so the model is
        # called with EMPTY context and will usually return the fallback.
        trace.note(
            "No context was retrieved, but chat history exists, so the model "
            "is being called with empty context."
        )
        logger.warning(
            "[%s] RAG: NO CONTEXT but history=%d -> calling model with "
            "EMPTY context",
            request_id,
            len(history),
        )

    # ---- 5. Generation ---------------------------------------------
    try:
        with trace.stage("generation"):
            answer = await _generate_with_retry(
                question=current_message,
                context=context,
                history=history,
                request_id=request_id,
                trace=trace,
            )
    except Exception:
        logger.exception("[%s] GENERATION_PIPELINE: FAILED", request_id)
        trace.outcome = "http_503_generation_pipeline_failed"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    if _is_unavailable_response(answer):
        logger.error("[%s] GENERATION: unavailable response", request_id)
        logger.info("==============================================")
        trace.outcome = "generation_unavailable"
        raise HTTPException(status_code=503, detail=UNAVAILABLE_ANSWER)

    answer = (answer or "").strip()

    # ---- 6. Source validation / grounding --------------------------
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

    # ---- 7. Final fallback -----------------------------------------
    if not answer or _is_knowledge_base_fallback(answer):
        found = trace.data.get("terms_found", [])
        missing = trace.data.get("terms_missing", [])

        logger.warning(
            "[%s] FINAL: model returned knowledge-base fallback | "
            "context_chars=%d sources_raw=%d query_used=%r "
            "terms_found=%s terms_missing=%s history=%d",
            request_id,
            len(context),
            len(retrieved_sources),
            query_used[:200],
            found,
            missing,
            len(history),
        )
        answer = FRIENDLY_FALLBACK
        sources = []
        trace.outcome = "kb_fallback_from_model"
    else:
        trace.outcome = "success"

    trace.data["final_answer_chars"] = len(answer)
    trace.data["final_sources"] = len(sources)

    logger.info(
        "[%s] FINAL: %s total=%.3fs context_chars=%d "
        "answer_chars=%d sources=%d",
        request_id,
        "SUCCESS" if trace.outcome == "success" else "FALLBACK",
        time.perf_counter() - started,
        len(context),
        len(answer),
        len(sources),
    )
    logger.info("==============================================")

    return ChatResponse(answer=answer, sources=sources)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main Ask Alcor pipeline with per-stage diagnostics.

    Internal exceptions are logged with a request id and full traceback.
    Visitors receive only the safe public error/fallback response.
    A SUMMARY block with a diagnosis is logged for every request.
    """

    request_id = uuid.uuid4().hex[:8]
    trace = _Trace(request_id)

    logger.info("")
    logger.info("========== ASK ALCOR REQUEST [%s] ==========", request_id)

    try:
        response = await _run_chat(request, trace)
    except HTTPException as exc:
        if trace.outcome == "unknown":
            trace.outcome = f"http_{exc.status_code}"
        _log_summary(trace)
        raise
    except Exception:
        trace.outcome = "unhandled_exception"
        logger.exception("[%s] UNHANDLED EXCEPTION", request_id)
        _log_summary(trace)
        raise

    _log_summary(trace)

    if DEBUG_RESPONSE:
        response.debug = _debug_payload(trace)

    return response