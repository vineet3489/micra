"""
MICRA — Scraper Node
======================

LEARNING CONCEPT: The orchestrator node pattern

The scraper node doesn't scrape anything directly. It orchestrates tools:

    research_plan
        │
        ├─→ search tool (discover URLs)
        ├─→ web scraper tool (fetch pages)
        └─→ paper fetcher tool (fetch abstracts)

This separation is key:
  - TOOLS are reusable, stateless functions (they don't touch state)
  - NODES are graph participants that read/write state
  - Nodes call tools; tools return data; nodes store results in state

This is the same pattern used by OpenAI's function calling and
LangChain's AgentExecutor — the agent decides which tools to call,
calls them, and incorporates results.

LEARNING CONCEPT: What state does this node produce?

Input fields read:  research_plan
Output fields written: sources, messages, errors

The `sources` field uses operator.add — so this node APPENDS its
SourceDocuments to the list. If we run the scraper node twice (e.g.
on a re-run), sources accumulate (we may want dedup logic later).
"""

import logging
import hashlib
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.state import MICRAState, SourceDocument
from src.tools.search import discover_urls_for_plan
from src.tools.web_scraper import scrape_urls
from src.tools.paper_fetcher import fetch_papers_for_queries, paper_to_text
from src.config import SCRAPER_MAX_URLS, PAPER_RESULTS_PER_QUERY

logger = logging.getLogger(__name__)
console = Console()


def scraper_node(state: MICRAState) -> dict:
    """
    Orchestrates data ingestion from all source types.

    LEARNING: Notice this node does a lot — but each individual step is
    delegated to a focused tool. The node's job is coordination:
      1. Decide what to fetch (from research_plan)
      2. Call the right tools
      3. Convert results into SourceDocuments
      4. Handle and log errors

    The node does NOT do chunking or embedding — that's the embedder node's job.
    Single responsibility: fetch raw text, convert to SourceDocuments.
    """
    plan = state.get("research_plan", {})

    if not plan:
        return {
            "errors": ["[scraper] No research plan found. Planner node may have failed."],
            "messages": ["[scraper] Skipped — no research plan available."]
        }

    source_types = plan.get("source_types_to_query", ["web_competitors"])
    search_queries = plan.get("search_queries", [])
    known_competitors = plan.get("competitor_names_to_research", [])

    console.print("\n[bold cyan]Phase 2: Data Ingestion[/bold cyan]")

    sources: list[SourceDocument] = []
    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # ── Step 1: Discover and scrape web sources ──────────────────────
        if any(t in source_types for t in ["web_competitors", "news", "regulatory"]):
            task = progress.add_task("Searching for sources...", total=None)

            search_results = discover_urls_for_plan(
                search_queries=search_queries,
                source_types=source_types,
                max_urls=SCRAPER_MAX_URLS
            )

            progress.update(task, description=f"Found {len(search_results)} URLs. Scraping...")

            # Build (url, source_type) pairs for the scraper
            url_pairs = [(r.url, r.source_type) for r in search_results]

            # Add known competitor names as search queries to find their sites
            for competitor in known_competitors:
                competitor_results = discover_urls_for_plan(
                    search_queries=[f"{competitor} official website product features"],
                    source_types=["web_competitors"],
                    max_urls=3
                )
                url_pairs.extend((r.url, "web_competitors") for r in competitor_results)

            scraped_pages = scrape_urls(url_pairs, max_urls=SCRAPER_MAX_URLS)

            progress.update(task, description="Processing scraped pages...")

            for page in scraped_pages:
                if page.success and page.text.strip():
                    sources.append(SourceDocument(
                        url=page.url,
                        source_type=page.source_type,
                        title=page.title or page.url,
                        raw_text=page.text,
                        chunk_ids=[]   # filled in by embedder node
                    ))
                elif not page.success:
                    errors.append(f"[scraper] Failed to scrape {page.url}: {page.error}")

            progress.update(task, description=f"Web sources: {len(sources)} ingested", total=1, completed=1)

        # ── Step 2: Fetch academic papers ────────────────────────────────
        if "academic_papers" in source_types:
            task = progress.add_task("Fetching academic papers...", total=None)

            papers = fetch_papers_for_queries(
                search_queries=search_queries,
                max_per_query=PAPER_RESULTS_PER_QUERY
            )

            for paper in papers:
                text = paper_to_text(paper)
                sources.append(SourceDocument(
                    url=paper.url,
                    source_type="academic",
                    title=paper.title,
                    raw_text=text,
                    chunk_ids=[]
                ))

            progress.update(
                task,
                description=f"Academic papers: {len(papers)} fetched",
                total=1, completed=1
            )

    # ── Summary ──────────────────────────────────────────────────────────
    web_count = sum(1 for s in sources if s["source_type"] != "academic")
    paper_count = sum(1 for s in sources if s["source_type"] == "academic")
    total_chars = sum(len(s["raw_text"]) for s in sources)

    console.print(
        f"\n[green]✓[/green] Ingested [bold]{len(sources)} sources[/bold] "
        f"({web_count} web, {paper_count} papers) — "
        f"{total_chars:,} characters total"
    )

    if errors:
        console.print(f"[yellow]⚠[/yellow] {len(errors)} sources failed (see logs)")

    # LEARNING: Minimum viability check.
    # If we got fewer than 5 sources, research quality will be poor.
    # We log a warning but don't crash — partial data is better than none.
    if len(sources) < 5:
        errors.append(
            f"[scraper] Only {len(sources)} sources ingested. "
            "Research quality may be limited. Consider adding more URLs manually."
        )

    return {
        "sources": sources,          # operator.add — appends to existing list
        "messages": [
            f"[scraper] Ingested {len(sources)} sources "
            f"({web_count} web, {paper_count} papers, {total_chars:,} chars)."
        ],
        "errors": errors
    }
