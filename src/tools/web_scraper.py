"""
MICRA — Web Scraper Tool
==========================

LEARNING CONCEPT: Data ingestion for RAG

RAG has two phases:
  1. OFFLINE (ingestion): Fetch docs → clean → chunk → embed → store
  2. ONLINE (retrieval): Query → embed query → find similar chunks → generate

This file handles the FETCH and CLEAN part of offline ingestion.

The scraper's job is simple but important:
  - Get the page
  - Strip all the noise (nav, footer, ads, scripts)
  - Return clean prose text

Why clean aggressively? Because every token you embed and store costs money
and adds noise. A page might be 50,000 characters of HTML but only 3,000
characters of useful content. Embedding the nav bar 200 times is wasteful.

CONTENT EXTRACTION STRATEGY:
We use BeautifulSoup to:
  1. Remove script/style/nav/footer tags entirely
  2. Extract text from content-bearing tags (p, h1-h6, li, article, main)
  3. Collapse whitespace

This is a pragmatic approach. More sophisticated alternatives exist
(Mozilla Readability, newspaper3k) but for learning, DIY is better.
"""

import time
import logging
import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from src.config import SCRAPER_TIMEOUT, SCRAPER_MAX_CHARS, SCRAPER_DELAY

logger = logging.getLogger(__name__)

# Mimic a real browser — many sites block requests without a User-Agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that contain actual content (vs. navigation/boilerplate)
CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "article",
                "section", "main", "blockquote", "td", "th"]

# Tags to remove entirely before extracting text
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside",
              "form", "button", "iframe", "noscript", "meta", "link"]


@dataclass
class ScrapedPage:
    """Result of scraping one URL."""
    url: str
    title: str
    text: str           # Clean, extracted text
    char_count: int
    success: bool
    error: str = ""

    # Source metadata — stored alongside every chunk in ChromaDB
    # so we can trace any retrieved chunk back to its source
    source_type: str = "web"


def scrape_url(url: str, source_type: str = "web") -> ScrapedPage:
    """
    Fetch and clean a single URL.

    LEARNING: Why do we need a clean text, not raw HTML?

    If you embed raw HTML into a vector store, you embed things like:
      <div class="nav-item"><a href="/pricing">Pricing</a></div>
    That tells the embedding model almost nothing meaningful. Worse, the
    same nav bar appears on every page — you'd have hundreds of near-
    identical embeddings cluttering your vector space.

    Clean text = higher quality embeddings = better retrieval.

    Args:
        url: URL to scrape
        source_type: Label stored in chunk metadata ("web", "news", "regulatory")

    Returns:
        ScrapedPage with clean text, or ScrapedPage with success=False on error
    """
    try:
        # LEARNING: httpx is the modern alternative to requests.
        # We set a timeout to avoid hanging on slow/unresponsive sites.
        # follow_redirects=True handles HTTP → HTTPS redirects automatically.
        with httpx.Client(
            headers=HEADERS,
            timeout=SCRAPER_TIMEOUT,
            follow_redirects=True,
            verify=False  # for corporate environments with SSL interception
        ) as client:
            response = client.get(url)
            response.raise_for_status()  # raises on 4xx/5xx status codes

        # Only process HTML responses (skip PDFs, images, etc.)
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ScrapedPage(
                url=url, title="", text="", char_count=0, success=False,
                error=f"Non-HTML content type: {content_type}",
                source_type=source_type
            )

        text, title = _extract_text(response.text)

        # Truncate very long pages — beyond SCRAPER_MAX_CHARS is usually
        # boilerplate, duplicate content, or low-value filler text
        text = text[:SCRAPER_MAX_CHARS]

        if not text.strip():
            return ScrapedPage(
                url=url, title=title, text="", char_count=0, success=False,
                error="No extractable text content found",
                source_type=source_type
            )

        time.sleep(SCRAPER_DELAY)  # polite delay

        return ScrapedPage(
            url=url,
            title=title,
            text=text,
            char_count=len(text),
            success=True,
            source_type=source_type
        )

    except httpx.TimeoutException:
        return ScrapedPage(url=url, title="", text="", char_count=0,
                          success=False, error="Request timed out",
                          source_type=source_type)
    except httpx.HTTPStatusError as e:
        return ScrapedPage(url=url, title="", text="", char_count=0,
                          success=False, error=f"HTTP {e.response.status_code}",
                          source_type=source_type)
    except Exception as e:
        return ScrapedPage(url=url, title="", text="", char_count=0,
                          success=False, error=str(e),
                          source_type=source_type)


def _extract_text(html: str) -> tuple[str, str]:
    """
    Extract clean text and page title from HTML.

    LEARNING: BeautifulSoup parsing strategy

    soup.get_text() on the whole document gives you everything —
    including nav menus, cookie banners, and JS code as strings.

    Better approach:
      1. Remove known noise tags entirely (decompose())
      2. Extract text only from content-bearing tags
      3. Join with spaces, collapse whitespace

    Returns: (clean_text, page_title)
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract title before removing tags
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove noise tags entirely
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()  # removes the tag and all its children from the tree

    # Extract text from content-bearing tags
    text_parts = []
    for tag in soup.find_all(CONTENT_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 30:  # skip very short fragments (likely nav items)
            text_parts.append(text)

    # Fallback: if no content tags found, just get all text
    if not text_parts:
        text_parts = [soup.get_text(separator=" ", strip=True)]

    raw_text = " ".join(text_parts)

    # Collapse whitespace — BeautifulSoup often leaves lots of \n\n\n
    clean_text = re.sub(r"\s+", " ", raw_text).strip()

    return clean_text, title


def scrape_urls(
    urls: list[tuple[str, str]],  # list of (url, source_type)
    max_urls: int = 15
) -> list[ScrapedPage]:
    """
    Scrape multiple URLs, collecting successes and logging failures.

    LEARNING: Partial failure tolerance

    In a real pipeline, some URLs will always fail:
      - Site is down
      - Blocks scrapers
      - Returns empty content
      - Times out

    The right approach: log failures as non-fatal errors, keep going.
    Don't crash the whole pipeline because one competitor's site is slow.

    We return ALL results (success and failure) so the caller can:
      - Count successes
      - Log failures as state errors
      - Decide if enough data was collected to proceed
    """
    results = []
    for url, source_type in urls[:max_urls]:
        logger.info(f"Scraping: {url}")
        page = scrape_url(url, source_type)
        results.append(page)
        if not page.success:
            logger.warning(f"Failed to scrape {url}: {page.error}")

    return results
