from __future__ import annotations

import os
import re


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Configurable fallback. No company/domain name is hard-coded.
FALLBACK_ANSWER = _env_str(
    "GROUNDING_FALLBACK_ANSWER",
    "I don't have that information in the knowledge base.",
)


# Share of meaningful words that must be supported by retrieved knowledge.
MIN_SUPPORT = _env_float("GROUNDING_MIN_SUPPORT", 0.5)


# Optional terms that are allowed even when they do not occur in context.
# Configure through:
# GROUNDING_ALLOWED_TERMS="term1,term2,term3"
ALLOWED_TERMS = {
    term.strip().lower()
    for term in os.getenv("GROUNDING_ALLOWED_TERMS", "").split(",")
    if term.strip()
}


STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers him
    his how i if in into is it its itself just me more most my no nor not of
    off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with would you your
    """.split()
)


def _stem(word: str) -> str:
    """Light stemming for common word-form variations."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w][\w'’-]*", text.lower())


def _context_vocabulary(context: str) -> tuple[set[str], set[str]]:
    """
    Build vocabulary from the complete retrieved knowledge.

    Returns:
        stems: stemmed vocabulary
        raw: exact vocabulary
    """
    stems: set[str] = set()
    raw: set[str] = set()

    for block in context.split("\n\n---\n\n"):
        for token in _tokens(block):
            raw.add(token)
            stems.add(_stem(token))

    return stems, raw


def _strip_markdown(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
    line = line.replace("**", "").replace("__", "")
    return line.strip()


def _is_supported(
    line: str,
    stems: set[str],
    raw: set[str],
) -> bool:
    text = _strip_markdown(line)
    tokens = _tokens(text)

    meaningful = [
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and token not in ALLOWED_TERMS
            and len(token) > 1
        )
    ]

    # Conversational / formatting-only text.
    if len(meaningful) < 2:
        return True

    # Numbers and years must be present exactly in retrieved knowledge.
    for token in meaningful:
        if any(ch.isdigit() for ch in token) and token not in raw:
            return False

    # Explicit names must exist in the retrieved knowledge.
    words = re.findall(r"[A-Za-z][\w'’-]*", text)

    for index, word in enumerate(words):
        if index == 0 or not word[0].isupper() or word.isupper():
            continue

        lowered = word.lower()

        if lowered in STOPWORDS or lowered in ALLOWED_TERMS:
            continue

        if lowered not in raw and _stem(lowered) not in stems:
            return False

    supported = sum(
        1 for token in meaningful
        if _stem(token) in stems
    )

    return supported / len(meaningful) >= MIN_SUPPORT


def validate_answer(
    answer: str,
    context: str,
) -> str:
    """
    Validate generated text against retrieved knowledge.

    Unsupported factual sentences are removed individually.
    If nothing supportable remains, return the configured fallback.
    """

    if not answer or not answer.strip():
        return FALLBACK_ANSWER

    if not context or not context.strip():
        return FALLBACK_ANSWER

    stems, raw = _context_vocabulary(context)

    if not stems:
        return FALLBACK_ANSWER

    kept: list[str] = []
    supported_claims = 0

    for line in answer.splitlines():
        stripped = line.strip()

        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        # Preserve markdown structure.
        if stripped.startswith(("#", "|")) or stripped in {"---", "***"}:
            kept.append(line.rstrip())
            continue

        # Validate individual sentences.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            stripped,
        )

        good = [
            sentence
            for sentence in sentences
            if _is_supported(sentence, stems, raw)
        ]

        if good:
            kept.append(" ".join(good))
            supported_claims += 1

    if supported_claims == 0:
        return FALLBACK_ANSWER

    return "\n".join(kept).strip()