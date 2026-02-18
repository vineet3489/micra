"""
MICRA — Academic Paper Fetcher
================================

LEARNING CONCEPT: Why academic papers matter for market intelligence

For technical markets (DERMS, AI infrastructure, biotech, etc.),
academic papers are a gold mine:

  - They define the problem space precisely
  - They cite existing solutions (your competitors)
  - They show what cutting-edge research is doing (future threats)
  - They quantify market problems (e.g. "grid instability costs $X/year")
  - They're written by domain experts, not marketers

Most market intelligence tools completely ignore papers. This is a
meaningful differentiation for MICRA.

TOOL: Semantic Scholar API (free, no key required for basic use)
---------------------------------------------------------------
Semantic Scholar indexes 200M+ academic papers across all fields.
Their API is free for up to 100 requests per 5 minutes.

Endpoints we use:
  - Paper search: /paper/search?query=...&fields=...
  - Returns: title, abstract, year, authors, citation count, URL

We only use the abstract (not the full paper) — abstracts are typically
150-300 words and contain the most information-dense content.
Full papers are behind paywalls; abstracts are always free.

LEARNING: "Abstract-only" RAG
For research intelligence, abstracts are usually sufficient.
They contain: problem statement, methodology, key findings, implications.
That's exactly what we need for market analysis.
"""

import time
import logging
from dataclasses import dataclass

import httpx

from src.config import PAPER_RESULTS_PER_QUERY

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Fields we want from the API
FIELDS = "title,abstract,year,authors,citationCount,externalIds,url"


@dataclass
class AcademicPaper:
    """A single academic paper with its metadata."""
    title: str
    abstract: str
    year: int
    authors: list[str]
    citation_count: int
    url: str
    source_type: str = "academic"


def fetch_papers(query: str, max_results: int = PAPER_RESULTS_PER_QUERY) -> list[AcademicPaper]:
    """
    Search Semantic Scholar and return papers with abstracts.

    LEARNING: API-based data fetching vs. web scraping

    Web scraping is brittle — page structure changes, sites block bots,
    content is hidden behind JS rendering. APIs are more reliable:
      - Structured JSON response (no HTML parsing)
      - Stable schema (versioned)
      - Rate limits are explicit and documented
      - No scraping ethics concerns

    When an API exists, always prefer it over scraping the website.

    Args:
        query: Search query (e.g. "distributed energy resource management USA")
        max_results: Max papers to return

    Returns:
        List of AcademicPaper objects with abstracts
    """
    papers = []

    try:
        params = {
            "query": query,
            "limit": max_results,
            "fields": FIELDS,
        }

        # LEARNING: We filter out papers without abstracts.
        # A paper with no abstract gives us nothing useful for RAG.
        # Quality over quantity — 5 papers with good abstracts beat
        # 20 papers with empty abstracts.

        with httpx.Client(timeout=15) as client:
            response = client.get(SEMANTIC_SCHOLAR_URL, params=params)
            response.raise_for_status()

        data = response.json()

        for item in data.get("data", []):
            abstract = item.get("abstract") or ""
            if not abstract.strip():
                continue  # skip papers with no abstract

            authors = [
                a.get("name", "") for a in item.get("authors", [])
            ]

            # Prefer the Semantic Scholar URL, fallback to a constructed one
            paper_id = item.get("paperId", "")
            url = item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}"

            papers.append(AcademicPaper(
                title=item.get("title", "Unknown title"),
                abstract=abstract,
                year=item.get("year") or 0,
                authors=authors[:5],  # cap at 5 authors for cleanliness
                citation_count=item.get("citationCount") or 0,
                url=url,
            ))

        # Sort by citation count descending — highly cited papers
        # are more likely to be foundational / trustworthy
        papers.sort(key=lambda p: p.citation_count, reverse=True)

    except httpx.TimeoutException:
        logger.warning(f"Semantic Scholar timed out for query: {query}")
    except Exception as e:
        logger.warning(f"Paper fetch failed for '{query}': {e}")

    return papers


def fetch_papers_for_queries(
    search_queries: list[str],
    max_per_query: int = PAPER_RESULTS_PER_QUERY
) -> list[AcademicPaper]:
    """
    Fetch papers for multiple queries, deduplicating by title.

    LEARNING: Why deduplicate by title and not URL?
    The same paper sometimes has multiple URLs (DOI, arXiv, S2 landing page).
    Deduplicating by normalized title is more robust.
    """
    all_papers: list[AcademicPaper] = []
    seen_titles: set[str] = set()

    for query in search_queries:
        papers = fetch_papers(query, max_per_query)
        for paper in papers:
            normalized = paper.title.lower().strip()
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                all_papers.append(paper)
        time.sleep(0.5)  # respect rate limits

    return all_papers


def paper_to_text(paper: AcademicPaper) -> str:
    """
    Convert a paper to a text block suitable for chunking.

    LEARNING: Formatting context for the LLM

    When we later retrieve this chunk during framework analysis,
    the LLM needs to know this came from an academic paper (not a
    competitor website). Including metadata like year and citation count
    helps the LLM weight the information appropriately:
      "This finding comes from a 2023 paper cited 847 times"
    is more trustworthy than "This is from a 2019 blog post".
    """
    author_str = ", ".join(paper.authors) if paper.authors else "Unknown authors"
    citation_str = f"Citations: {paper.citation_count}" if paper.citation_count else ""

    return (
        f"[ACADEMIC PAPER]\n"
        f"Title: {paper.title}\n"
        f"Authors: {author_str}\n"
        f"Year: {paper.year}\n"
        f"{citation_str}\n\n"
        f"{paper.abstract}"
    )
