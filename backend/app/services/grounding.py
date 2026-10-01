from __future__ import annotations

import os
import re


FALLBACK_ANSWER = (
    "I don't have that information in the Alcor knowledge base."
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Share of a line's meaningful words that must appear in the retrieved
# knowledge for the line to count as supported.
MIN_SUPPORT = _env_float("GROUNDING_MIN_SUPPORT", 0.5)

# Words that may appear in answers even if the knowledge never uses them
# (the company / assistant name, for example). Comma separated.
ALLOWED_TERMS = {
    term.strip().lower()
    for term in os.getenv("GROUNDING_ALLOWED_TERMS", "alcor,solutions").split(",")
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
    """Very light stemming so 'leads' / 'leading' / 'led' style variants match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w][\w'’-]*", text.lower())


def _context_vocabulary(context: str) -> tuple[set[str], set[str]]:
    """
    Return (stems, raw_words) found in the retrieved knowledge.
    Both the title and the content of every chunk count as knowledge.
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


def _is_supported(line: str, stems: set[str], raw: set[str]) -> bool:
    text = _strip_markdown(line)
    tokens = _tokens(text)

    meaningful = [
        token for token in tokens
        if token not in STOPWORDS
        and token not in ALLOWED_TERMS
        and len(token) > 1
    ]

    # Too short to contain a checkable claim ("Sure!", "Here you go:").
    if len(meaningful) < 2:
        return True

    # Numbers and years must appear in the knowledge exactly.
    for token in meaningful:
        if any(ch.isdigit() for ch in token) and token not in raw:
            return False

    # Names (capitalised words inside the sentence) must appear in the knowledge.
    words = re.findall(r"[A-Za-z][\w'’-]*", text)
    for index, word in enumerate(words):
        if index == 0 or not word[0].isupper() or word.isupper():
            continue
        lowered = word.lower()
        if lowered in STOPWORDS or lowered in ALLOWED_TERMS:
            continue
        if lowered not in raw and _stem(lowered) not in stems:
            return False

    supported = sum(1 for token in meaningful if _stem(token) in stems)

    return supported / len(meaningful) >= MIN_SUPPORT


def validate_answer(
    answer: str,
    context: str,
) -> str:
    """
    Return the answer with any unsupported lines removed.

    * Every line is checked against the retrieved knowledge.
    * Unsupported lines are dropped instead of rejecting the whole answer,
      so one weak sentence does not throw away a good response.
    * If nothing supported remains, the fallback answer is returned.
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
            # keep paragraph spacing, but never two blank lines in a row
            if kept and kept[-1] != "":
                kept.append("")
            continue

        # Formatting-only lines are not factual claims.
        if stripped.startswith(("#", "|")) or stripped in {"---", "***"}:
            kept.append(line.rstrip())
            continue

        # Check sentence by sentence so one wrong sentence does not
        # remove the correct ones next to it.
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        good = [sentence for sentence in sentences if _is_supported(sentence, stems, raw)]

        if good:
            kept.append(" ".join(good))
            supported_claims += 1

    if supported_claims == 0:
        return FALLBACK_ANSWER

    return "\n".join(kept).strip()