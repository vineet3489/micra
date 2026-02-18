"""
MICRA — Web Search Tool
=========================

LEARNING CONCEPT: Agent Tools

A "tool" is a function an agent can call to interact with the outside world.
Tools are how agents escape the LLM's training data and access:
  - Live web pages
  - APIs
  - Databases
  - File systems

In LangChain/LangGraph, tools can be defined with the @tool decorator
(we'll use that in Phase 3). For now, they're plain functions that return
structured data — which is simpler to understand and test.

THIS TOOL: DuckDuckGo search (no API key required)
--------------------------------------------------
DuckDuckGo's search API is free, unofficial, and doesn't need credentials.
The `duckduckgo-search` library wraps it cleanly.

We use it for:
  1. Discovering competitor URLs from search queries
  2. Finding news articles about market developments
  3. Discovering regulatory/government sources

The alternative (Google Search API) requires a paid API key.
For learning and prototyping, DuckDuckGo is perfect.
"""

import time
import logging
from dataclasses import dataclass

from duckduckgo_search import DDGS

from src.config import SEARCH_RESULTS_PER_QUERY

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source_type: str  # "web", "news"


def search_web(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> list[SearchResult]:
    """
    Search DuckDuckGo and return URLs + snippets.

    LEARNING: Why return snippets?
    Snippets are the 1-2 sentence previews shown in search results.
    They're useful as a lightweight signal — if a snippet is clearly
    irrelevant, we can skip scraping that URL (saves time + tokens).
    In Phase 3 we'll add snippet-based URL filtering.

    Args:
        query: Search query string
        max_results: How many results to return

    Returns:
        List of SearchResult objects
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source_type="web"
                ))
    except Exception as e:
        logger.warning(f"Search failed for query '{query}': {e}")

    return results


def search_news(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> list[SearchResult]:
    """
    Search DuckDuckGo News for recent articles.

    LEARNING: Why separate web vs. news search?
    Web search returns any page — documentation, company sites, old blog posts.
    News search specifically returns recent articles, which is what we want
    for "market developments" and "recent funding" queries.
    We use different queries for each: news queries include time signals
    like "2024 funding round" or "market growth Q3 2024".
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("body", ""),
                    source_type="news"
                ))
    except Exception as e:
        logger.warning(f"News search failed for query '{query}': {e}")

    return results


def discover_urls_for_plan(
    search_queries: list[str],
    source_types: list[str],
    max_urls: int = 15
) -> list[SearchResult]:
    """
    Run multiple search queries and collect unique URLs.

    LEARNING: Deduplication with a seen set.
    Multiple queries often return the same URL (e.g. a competitor's homepage
    appears for every query about that competitor). We deduplicate by URL
    before returning, so we don't scrape the same page twice.

    Args:
        search_queries: The planner's generated search queries
        source_types: Which types to search ("web_competitors", "news", etc.)
        max_urls: Cap total URLs to avoid excessive scraping

    Returns:
        Unique search results up to max_urls
    """
    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for query in search_queries:
        if len(all_results) >= max_urls:
            break

        # Web search for competitor/market queries
        if any(t in source_types for t in ["web_competitors", "regulatory"]):
            for result in search_web(query):
                if result.url and result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
            time.sleep(0.5)  # polite delay between queries

        # News search for news queries
        if "news" in source_types:
            news_query = f"{query} 2024 2025"
            for result in search_news(news_query):
                if result.url and result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
            time.sleep(0.5)

    return all_results[:max_urls]
