from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .extractor import extract_page
from .sitemap import SitemapEntry, discover_all_urls


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str | None
    content: str
    last_modified: str | None
    content_hash: str
    crawled_at: str


def calculate_content_hash(content: str) -> str:
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def crawl_entry(entry: SitemapEntry) -> CrawledPage:
    page = extract_page(entry.url)

    return CrawledPage(
        url=page.url,
        title=page.title,
        content=page.content,
        last_modified=entry.last_modified,
        content_hash=calculate_content_hash(page.content),
        crawled_at=datetime.now(timezone.utc).isoformat(),
    )


def save_page(
    page: CrawledPage,
    output_directory: Path,
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        hashlib.sha256(
            page.url.encode("utf-8")
        ).hexdigest()
        + ".json"
    )

    output_path = output_directory / filename

    output_path.write_text(
        json.dumps(
            {
                "url": page.url,
                "title": page.title,
                "content": page.content,
                "last_modified": page.last_modified,
                "content_hash": page.content_hash,
                "crawled_at": page.crawled_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def crawl(
    sitemap_index_url: str,
    output_directory: Path,
    limit: int | None = None,
) -> list[CrawledPage]:
    entries = discover_all_urls(
        sitemap_index_url=sitemap_index_url,
    )

    if limit is not None:
        entries = entries[:limit]

    crawled_pages: list[CrawledPage] = []

    for entry in entries:
        try:
            page = crawl_entry(entry)

            save_page(
                page,
                output_directory,
            )

            crawled_pages.append(page)

        except Exception as error:
            print(
                f"Failed to crawl {entry.url}: {error}"
            )

    return crawled_pages


def crawl_url(
    url: str,
    output_directory: Path,
) -> CrawledPage:
    entry = SitemapEntry(url=url)

    page = crawl_entry(entry)

    save_page(
        page,
        output_directory,
    )

    return page