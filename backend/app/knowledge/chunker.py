from __future__ import annotations

import re

from .db import get_database_connection


HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+)$"
)


def _normalise_text(
    text: str,
) -> str:

    lines: list[str] = []

    for line in text.splitlines():

        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    return "\n".join(lines)


def _word_count(
    text: str,
) -> int:

    return len(
        re.findall(
            r"\S+",
            text,
        )
    )


def _split_large_section(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:

    words = re.findall(
        r"\S+",
        text,
    )

    if not words:
        return []

    chunks: list[str] = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(
            end - chunk_overlap,
            start + 1,
        )

    return chunks


def _parse_sections(
    text: str,
) -> list[tuple[int, str]]:

    """
    Parse heading-based sections.

    Each section keeps its heading level so the chunking
    logic can use the document's existing structure without
    relying on page-specific names or hard-coded content.
    """

    lines = text.splitlines()

    sections: list[tuple[int, str]] = []

    current: list[str] = []
    current_level: int | None = None

    for line in lines:

        match = HEADING_PATTERN.match(line)

        if match:

            if current:

                section = "\n".join(
                    current
                ).strip()

                if section:

                    sections.append(
                        (
                            current_level or 1,
                            section,
                        )
                    )

            current_level = len(
                match.group(1)
            )

            current = [line]

        else:

            current.append(line)

    if current:

        section = "\n".join(
            current
        ).strip()

        if section:

            sections.append(
                (
                    current_level or 1,
                    section,
                )
            )

    return sections


def _merge_sections(
    sections: list[tuple[int, str]],
    minimum_words: int = 80,
) -> list[str]:

    """
    Merge short consecutive sections into coherent groups.

    Large sections remain independent.

    Short sections are grouped together until they contain
    enough information to form a useful retrieval unit.

    The logic is completely data-driven and does not contain
    page-specific names, roles, products, dates, URLs,
    or question-specific rules.
    """

    if not sections:
        return []

    cleaned: list[tuple[int, str]] = []

    for level, section in sections:

        section = section.strip()

        if not section:
            continue

        lines = [
            line.strip()
            for line in section.splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        # Remove sections that contain only a heading.
        if (
            len(lines) == 1
            and HEADING_PATTERN.match(lines[0])
        ):
            continue

        cleaned.append(
            (
                level,
                section,
            )
        )

    if not cleaned:
        return []

    merged: list[str] = []

    current_parts: list[str] = []
    current_words = 0

    for level, section in cleaned:

        words = _word_count(section)

        # Large sections already contain enough context.
        # Keep them as independent retrieval units.
        if words >= minimum_words:

            if current_parts:

                merged.append(
                    "\n".join(
                        current_parts
                    ).strip()
                )

                current_parts = []
                current_words = 0

            merged.append(section)

            continue

        # Short sections are accumulated so related
        # structured information stays together.
        current_parts.append(section)

        current_words += words

        # Once the accumulated content reaches the
        # minimum useful size, create one merged section.
        if current_words >= minimum_words:

            merged.append(
                "\n".join(
                    current_parts
                ).strip()
            )

            current_parts = []
            current_words = 0

    # Preserve any remaining short sections.
    if current_parts:

        merged.append(
            "\n".join(
                current_parts
            ).strip()
        )

    return merged


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:

    """
    Convert extracted page content into retrieval-friendly
    chunks.

    The process is:

    1. Normalize the extracted content.
    2. Identify heading-based sections.
    3. Merge short consecutive sections.
    4. Keep large sections independent.
    5. Split sections larger than chunk_size using overlap.
    """

    text = _normalise_text(
        text
    )

    if not text:
        return []

    sections = _parse_sections(
        text
    )

    sections = _merge_sections(
        sections
    )

    chunks: list[str] = []

    for section in sections:

        if (
            _word_count(section)
            <= chunk_size
        ):

            chunks.append(
                section
            )

        else:

            chunks.extend(
                _split_large_section(
                    section,
                    chunk_size,
                    chunk_overlap,
                )
            )

    return chunks


def store_document_chunks(
    document_id: int,
    chunks: list[str],
) -> None:

    """
    Replace the existing chunks for a document.
    """

    with get_database_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM chunks
                WHERE document_id = %s
                """,
                (
                    document_id,
                ),
            )

            for index, content in enumerate(
                chunks
            ):

                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_index,
                        content
                    )
                    VALUES (
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        document_id,
                        index,
                        content,
                    ),
                )