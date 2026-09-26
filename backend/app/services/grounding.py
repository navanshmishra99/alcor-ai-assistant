from __future__ import annotations

import re


FALLBACK_ANSWER = (
    "I don't have that information in the Alcor knowledge base."
)


def _normalise(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def _extract_source_sentences(
    context: str,
) -> list[str]:

    sentences: list[str] = []

    for block in context.split("\n\n---\n\n"):

        content_match = re.search(
            r"Content:\s*(.*)",
            block,
            flags=re.DOTALL,
        )

        if not content_match:
            continue

        content = content_match.group(1)

        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            content,
        )

        for part in parts:

            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


def validate_answer(
    answer: str,
    context: str,
) -> str:

    if not answer.strip():
        return FALLBACK_ANSWER

    if not context.strip():
        return FALLBACK_ANSWER

    source_sentences = (
        _extract_source_sentences(context)
    )

    if not source_sentences:
        return FALLBACK_ANSWER

    normalised_sources = [
        _normalise(sentence)
        for sentence in source_sentences
    ]

    answer_lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    unsupported_lines: list[str] = []

    for line in answer_lines:

        # Formatting-only lines are not factual claims.
        if line.startswith("#"):
            continue

        if line.startswith("|"):
            continue

        if line in {"---", "***"}:
            continue

        normalised_line = _normalise(line)

        if not normalised_line:
            continue

        # A line is considered supported when a meaningful
        # portion of it is represented in the retrieved
        # knowledge.
        words = re.findall(
            r"\b[\w’'-]+\b",
            normalised_line,
        )

        if len(words) < 4:
            continue

        matching_words = 0

        for word in words:

            if any(
                word in source
                for source in normalised_sources
            ):
                matching_words += 1

        support_ratio = (
            matching_words / len(words)
        )

        if support_ratio < 0.45:

            unsupported_lines.append(
                line
            )

    if unsupported_lines:
        return FALLBACK_ANSWER

    return answer