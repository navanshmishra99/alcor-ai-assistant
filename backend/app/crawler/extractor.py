from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class ExtractedPage:
    url: str
    title: str | None
    content: str


def _normalise_content(content: str) -> str:
    lines = []

    for line in content.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def _extract_main_content(soup: BeautifulSoup) -> str:
    """
    Extract readable page content while removing common
    site-wide and non-content HTML elements.

    Heading/list relationships are preserved where the HTML
    structure clearly represents a label followed by a heading.

    No page-specific names, URLs, or keywords are used.
    """

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "template",
            "svg",
            "iframe",
            "nav",
            "header",
            "footer",
            "form",
        ]
    ):
        element.decompose()

    for element in soup.find_all(
        attrs={
            "role": [
                "navigation",
                "banner",
                "contentinfo",
                "complementary",
            ]
        }
    ):
        element.decompose()

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(
            attrs={
                "role": "main",
            }
        )
        or soup.body
    )

    if root is None:
        return ""

    output: list[str] = []

    elements = root.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "blockquote",
            "pre",
        ]
    )

    index = 0

    while index < len(elements):
        element = elements[index]

        text = element.get_text(
            " ",
            strip=True,
        )

        if not text:
            index += 1
            continue

        tag = element.name

        # If this list item is immediately followed by a heading
        # whose text is contained in the list item's text, the
        # pair will be processed when the heading is reached.
        if tag == "li" and index + 1 < len(elements):
            next_element = elements[index + 1]

            if next_element.name in {
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }:
                next_text = next_element.get_text(
                    " ",
                    strip=True,
                )

                if next_text and next_text in text:
                    index += 1
                    continue

        # Convert a heading + preceding list item into one
        # clean heading/role structure.
        if (
            tag in {
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }
            and index > 0
        ):
            previous = elements[index - 1]

            if previous.name == "li":
                previous_text = previous.get_text(
                    " ",
                    strip=True,
                )

                if text and text in previous_text:
                    heading_level = int(tag[1])

                    role_text = (
                        previous_text
                        .replace(text, "", 1)
                        .strip(" :-–—")
                    )

                    output.append(
                        f"{'#' * heading_level} {text}"
                    )

                    if role_text:
                        output.append(role_text)

                    index += 1
                    continue

        if tag.startswith("h"):
            level = int(tag[1])

            output.append(
                f"{'#' * level} {text}"
            )

        elif tag == "li":
            output.append(
                f"- {text}"
            )

        else:
            output.append(text)

        index += 1

    return _normalise_content(
        "\n".join(output)
    )


def extract_page(
    url: str,
    timeout: int = 30,
) -> ExtractedPage:

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "AlcorAI-Crawler/0.1",
        },
    )

    response.raise_for_status()

    html = response.text

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    title = (
        soup.title.get_text(strip=True)
        if soup.title
        else None
    )

    content = _extract_main_content(
        soup
    )

    return ExtractedPage(
        url=url,
        title=title,
        content=content,
    )