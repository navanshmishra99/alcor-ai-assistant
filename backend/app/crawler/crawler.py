from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .extractor import extract_page
from .sitemap import SitemapEntry, discover_all_urls


logger = logging.getLogger("ask_alcor.crawler")

MANIFEST_FILENAME = "_manifest.json"

# Politeness delay between requests, and retry settings for transient
# network failures. Kept simple deliberately — a small sequential
# crawler with a short delay, not a full retry framework.
REQUEST_DELAY_SECONDS = 0.2
MAX_FETCH_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str | None
    content: str
    last_modified: str | None
    content_hash: str
    crawled_at: str


@dataclass(frozen=True)
class CrawlFailure:
    url: str
    error: str


@dataclass(frozen=True)
class CrawlResult:
    """
    Returned by crawl(). Replaces the old list[CrawledPage] return value
    — see the note where crawl() is defined for what callers need to
    change.
    """
    pages: list[CrawledPage]
    failures: list[CrawlFailure]
    skipped: int  # pages reused from the manifest, not re-fetched


def calculate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _page_filename(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"


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


def _crawl_entry_with_retry(entry: SitemapEntry) -> CrawledPage:
    last_error: Exception | None = None

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            return crawl_entry(entry)
        except Exception as error:
            last_error = error
            logger.warning(
                "Fetch attempt %d/%d failed for %s: %s",
                attempt, MAX_FETCH_ATTEMPTS, entry.url, error,
            )
            if attempt < MAX_FETCH_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error


def save_page(page: CrawledPage, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / _page_filename(page.url)

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


def _load_manifest(output_directory: Path) -> dict[str, dict]:
    manifest_path = output_directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "Manifest at %s is unreadable; starting a fresh one.", manifest_path
        )
        return {}


def _save_manifest(output_directory: Path, manifest: dict[str, dict]) -> None:
    manifest_path = output_directory / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_cached_page(output_directory: Path, filename: str) -> CrawledPage | None:
    path = output_directory / filename
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CrawledPage(**data)
    except (json.JSONDecodeError, OSError, TypeError) as error:
        logger.warning("Could not reuse cached page at %s: %s", path, error)
        return None


def _manifest_entry_for(page: CrawledPage) -> dict:
    return {
        "filename": _page_filename(page.url),
        "content_hash": page.content_hash,
        "last_modified": page.last_modified,
        "crawled_at": page.crawled_at,
    }


def crawl(
    sitemap_index_url: str,
    output_directory: Path,
    limit: int | None = None,
    force: bool = False,
) -> CrawlResult:
    """
    Crawls every URL in the sitemap.

    BREAKING CHANGE from the previous version: this now returns a
    CrawlResult (pages, failures, skipped) instead of a bare
    list[CrawledPage]. Callers using `pages = crawl(...)` need to
    change to `result = crawl(...); pages = result.pages`.

    Pages whose sitemap last_modified matches what's recorded in the
    manifest from a previous run are reused from disk instead of being
    re-fetched. Pass force=True to always re-fetch everything (e.g. for
    a full rebuild after an ingestion-pipeline change like the
    boilerplate-stripping fix).
    """
    entries = discover_all_urls(sitemap_index_url=sitemap_index_url)

    if limit is not None:
        entries = entries[:limit]

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output_directory)

    crawled_pages: list[CrawledPage] = []
    failures: list[CrawlFailure] = []
    skipped = 0

    for entry in entries:
        manifest_entry = manifest.get(entry.url)

        can_skip = (
            not force
            and manifest_entry is not None
            and entry.last_modified is not None
            and manifest_entry.get("last_modified") == entry.last_modified
        )

        if can_skip:
            cached_page = _load_cached_page(output_directory, manifest_entry["filename"])
            if cached_page is not None:
                crawled_pages.append(cached_page)
                skipped += 1
                logger.debug("Skipping unchanged page: %s", entry.url)
                continue

        try:
            page = _crawl_entry_with_retry(entry)
        except Exception as error:
            logger.error("Failed to crawl %s: %s", entry.url, error)
            failures.append(CrawlFailure(url=entry.url, error=str(error)))
            continue

        save_page(page, output_directory)
        crawled_pages.append(page)
        manifest[entry.url] = _manifest_entry_for(page)

        time.sleep(REQUEST_DELAY_SECONDS)

    _save_manifest(output_directory, manifest)

    if failures:
        preview = ", ".join(f.url for f in failures[:10])
        if len(failures) > 10:
            preview += ", ..."
        logger.warning(
            "Crawl finished with %d failure(s) out of %d page(s): %s",
            len(failures), len(entries), preview,
        )

    logger.info(
        "Crawl complete | total=%d fetched=%d skipped=%d failed=%d",
        len(entries), len(crawled_pages) - skipped, skipped, len(failures),
    )

    return CrawlResult(pages=crawled_pages, failures=failures, skipped=skipped)


def crawl_url(url: str, output_directory: Path) -> CrawledPage:
    """
    Crawls a single URL, always fetching fresh — this is an explicit,
    on-demand action, so it deliberately ignores the manifest's skip
    logic rather than silently reusing a stale cached copy.
    """
    entry = SitemapEntry(url=url)
    page = crawl_entry(entry)

    output_directory.mkdir(parents=True, exist_ok=True)
    save_page(page, output_directory)

    manifest = _load_manifest(output_directory)
    manifest[url] = _manifest_entry_for(page)
    _save_manifest(output_directory, manifest)

    return page