from dataclasses import dataclass
from xml.etree import ElementTree

import requests


SITEMAP_NS = {
    "s": "http://www.sitemaps.org/schemas/sitemap/0.9"
}


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    last_modified: str | None = None


def fetch_xml(url: str, timeout: int = 30) -> ElementTree.Element:
    response = requests.get(
        url,
        timeout=timeout,
    )
    response.raise_for_status()

    return ElementTree.fromstring(response.content)


def discover_sitemaps(
    sitemap_index_url: str,
    timeout: int = 30,
) -> list[str]:
    root = fetch_xml(
        sitemap_index_url,
        timeout=timeout,
    )

    return [
        loc.text.strip()
        for loc in root.findall("s:sitemap/s:loc", SITEMAP_NS)
        if loc.text
    ]


def discover_urls(
    sitemap_url: str,
    timeout: int = 30,
) -> list[SitemapEntry]:
    root = fetch_xml(
        sitemap_url,
        timeout=timeout,
    )

    entries: list[SitemapEntry] = []

    for url_element in root.findall("s:url", SITEMAP_NS):
        location = url_element.findtext(
            "s:loc",
            namespaces=SITEMAP_NS,
        )

        if not location:
            continue

        last_modified = url_element.findtext(
            "s:lastmod",
            namespaces=SITEMAP_NS,
        )

        entries.append(
            SitemapEntry(
                url=location.strip(),
                last_modified=(
                    last_modified.strip()
                    if last_modified
                    else None
                ),
            )
        )

    return entries


def discover_all_urls(
    sitemap_index_url: str,
    timeout: int = 30,
) -> list[SitemapEntry]:
    sitemap_urls = discover_sitemaps(
        sitemap_index_url,
        timeout=timeout,
    )

    all_entries: list[SitemapEntry] = []

    for sitemap_url in sitemap_urls:
        all_entries.extend(
            discover_urls(
                sitemap_url,
                timeout=timeout,
            )
        )

    unique_entries: dict[str, SitemapEntry] = {}

    for entry in all_entries:
        unique_entries[entry.url] = entry

    return list(unique_entries.values())