"""
MICRA — Competitive Intelligence Node
========================================

LEARNING CONCEPT: Two-source intelligence gathering

The fundamental problem with RAG-only competitive profiling:
  If the scraper didn't find a page about Competitor X's pricing,
  the profiler has nothing to work with — and returns "Not found".

The fix: combine TWO sources of evidence per competitor:

  1. Vector DB retrieval  — what the scraper already found (broad context)
  2. Live targeted search — fresh searches specifically for THIS competitor

Live search queries are targeted to the exact intelligence we need:
  - Pricing: "{name} pricing cost license"
  - Reviews:  "{name} G2 Capterra customer reviews"
  - Position: "{name} market share revenue customers"
  - Features: "{name} product features capabilities"
  - News:     "{name} funding acquisition product launch 2024"

This way, even if the user said "research the Historian market" without
naming any competitors — we discover names from content, then immediately
go fetch detailed information about each one.

LEARNING CONCEPT: Competitor discovery

Two layers:
  1. Plan-specified: known competitors from clarification answers
  2. Auto-discovered: names extracted from scraped content

For auto-discovery we use two approaches:
  a. LLM extraction from vector DB chunks (fast, uses existing data)
  b. Dedicated market leader search if extraction returns too few names
     ("top historian software competitors market leaders 2024")

LEARNING CONCEPT: "One LLM call per entity" pattern

For N competitors we run N LLM calls, each producing one CompetitorProfile.
More reliable than asking for all profiles in one call (which causes the
LLM to confuse feature sets and pricing across competitors).
"""

import json
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.state import MICRAState, CompetitorProfile
from src.retriever import Retriever
from src.tools.search import search_web, search_news
from src.tools.web_scraper import scrape_urls
from src.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)
console = Console()


# ── Richer profile schema ────────────────────────────────────────────────────

class CompetitorProfileSchema(BaseModel):
    product_summary: str = Field(
        description="1-2 sentence description of what this company's product does and who it's for"
    )
    usp: str = Field(
        description="Their #1 unique selling proposition — the ONE thing they claim to be best at. "
                    "e.g. 'Never interpolates data — 100% lossless compression' or 'Easiest deployment in the market'"
    )
    core_features: list[str] = Field(
        description="6-10 specific product features, named as the company describes them"
    )
    pricing_model: str = Field(
        description="Pricing structure: e.g., 'Enterprise SaaS, per-site licensing' or 'Subscription, tag-based'"
    )
    pricing_specifics: str = Field(
        description="Any actual dollar figures found: '$200K initial license + $40K/yr maintenance', "
                    "'$40K/year subscription', 'Starts at $12K/year for 10K tags'. "
                    "Write 'Not publicly disclosed — quote-based' if not found."
    )
    market_position: str = Field(
        description="Their position in the market: 'Market Leader (~40-45% share)', "
                    "'Strong #2 in process industries', 'Fast-growing challenger with 20,000+ installs', "
                    "'Legacy player, declining market share'. Include any known % or rank."
    )
    market_share_estimate: str = Field(
        description="Market share % or qualitative position if % unavailable. "
                    "e.g. '40-45% of enterprise historian market' or 'Dominant in refining/petrochemical'"
    )
    revenue_estimate: str = Field(
        description="Annual revenue or ARR if publicly available or estimable from funding/customer count. "
                    "e.g. '$500M ARR (2023, estimated)' or 'Not publicly disclosed'"
    )
    notable_customers: list[str] = Field(
        description="Customer names, logos, or case studies mentioned in sources. "
                    "e.g. ['Shell', 'ExxonMobil', 'WBSEDCL', '65% of Global 500 companies']. "
                    "Empty list if none found."
    )
    customer_reviews_summary: str = Field(
        description="What customers actually say — synthesized from G2, Capterra, Reddit, forums, "
                    "or case studies found in sources. Include both praise and complaints. "
                    "e.g. 'Users love the 200+ integrations but frequently complain about the price "
                    "($200K+) and steep learning curve requiring specialized admins.'"
    )
    recent_developments: list[str] = Field(
        description="Recent news from 2023-2025: product launches, funding rounds, acquisitions, "
                    "executive changes, major customer wins. Empty list if none found."
    )
    target_segment: str = Field(
        description="Primary customer segment: e.g., 'Global Fortune 500 utilities and O&G, "
                    "enterprise with 1M+ tags' or 'Mid-market manufacturing, SMB-friendly pricing'"
    )
    tech_stack_signals: list[str] = Field(
        description="Technology signals from docs, job postings, or marketing. "
                    "e.g. ['OPC UA', 'REST API', 'AWS-native', 'Python SDK', 'SQL interface']"
    )
    release_velocity: str = Field(
        description="Estimated release cadence from changelog/news. "
                    "e.g. 'Monthly releases, major version annually' or 'Slow — last major release 2022'"
    )
    funding_stage: str = Field(
        description="Funding/ownership: '$45M Series C (2023)', 'Acquired by AVEVA (2021)', "
                    "'Public company (NASDAQ: AZPN)', or 'Bootstrapped / unknown'"
    )
    strengths: list[str] = Field(
        description="4-6 genuine competitive strengths, grounded in evidence from sources"
    )
    weaknesses: list[str] = Field(
        description="4-6 weaknesses or gaps — from customer reviews, missing features, "
                    "or analyst commentary found in sources"
    )
    differentiation_gaps: list[str] = Field(
        description="4-6 specific opportunities where a new entrant could beat this competitor. "
                    "Be specific: 'No renewable energy / DER-specific features' not 'lacks innovation'"
    )


# ── Live search per competitor ───────────────────────────────────────────────

def _live_search_competitor(name: str) -> str:
    """
    Fetch fresh web content about a competitor using targeted search queries.

    LEARNING: Why live search vs. vector DB only?

    The vector DB only contains what the scraper happened to find during the
    broad market research phase. That phase uses queries like "DERMS market
    overview" — which may mention competitor names but rarely surfaces their
    pricing pages, review aggregations, or market share data.

    By running targeted searches HERE, we guarantee fresh, specific context
    for each competitor regardless of what the scraper found.

    We cap at 6 scraped pages total to keep runtime reasonable (~30s per competitor).
    """
    queries = [
        f"{name} product features pricing 2024 2025",
        f"{name} market share customers revenue",
        f"{name} G2 Capterra reviews complaints strengths",
        f"{name} vs alternatives comparison",
    ]

    # Web search
    all_urls: list[tuple[str, str]] = []
    for query in queries:
        results = search_web(query, max_results=2)
        for r in results:
            all_urls.append((r.url, "web_competitors"))

    # Recent news
    news_results = search_news(f"{name} 2024 2025 product launch funding news", max_results=3)
    for r in news_results:
        all_urls.append((r.url, "news"))

    # Deduplicate URLs
    seen_urls: set[str] = set()
    unique_pairs = []
    for url, src_type in all_urls:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_pairs.append((url, src_type))

    if not unique_pairs:
        return ""

    # Scrape (cap at 6 pages to keep runtime reasonable)
    scraped = scrape_urls(unique_pairs[:6], max_urls=6)

    parts = []
    for page in scraped:
        if page.success and page.text.strip():
            parts.append(f"[Source: {page.url}]\n{page.text[:3000]}")

    return "\n\n---\n\n".join(parts)


# ── Competitor discovery ─────────────────────────────────────────────────────

def _discover_competitors_from_sources(brief: str, retriever: Retriever) -> list[str]:
    """
    Discover competitor names from vector DB + a dedicated market search.

    Two-pass approach:
      Pass 1: Ask LLM to extract names from existing vector DB chunks
      Pass 2: If too few found, do a dedicated "market leaders" web search
    """
    # Pass 1: vector DB extraction
    chunks = retriever.retrieve(
        "competitor company vendor product platform market leader provider",
        k=8
    )
    context = "\n\n".join(c.text for c in chunks[:6]) if chunks else ""

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    names: list[str] = []
    if context:
        response = llm.invoke([
            SystemMessage(content=(
                "Extract company or product names that are direct competitors in this market. "
                "Return a JSON array of name strings only. Max 6 names. "
                "Only include companies with a clear competing product — not analysts, consultants, or general tech companies."
            )),
            HumanMessage(content=f"Research context:\n{context}\n\nReturn JSON array of competitor names.")
        ])
        try:
            names = [n for n in json.loads(response.content) if isinstance(n, str) and len(n) > 1][:6]
        except (json.JSONDecodeError, TypeError):
            names = []

    # Pass 2: if we found fewer than 3, do a dedicated market leaders search
    if len(names) < 3:
        target_market = brief.split("\n")[0][:100]  # first line of brief as market hint
        search_results = search_web(
            f"top {target_market} software vendors market leaders competitors 2024",
            max_results=5
        )
        if search_results:
            snippets = "\n".join(f"{r.title}: {r.snippet}" for r in search_results)
            response2 = llm.invoke([
                SystemMessage(content=(
                    "From these search results, extract the names of software companies/products "
                    "that are competitors in this market. Return a JSON array of name strings. Max 6 names."
                )),
                HumanMessage(content=snippets)
            ])
            try:
                extra = [n for n in json.loads(response2.content)
                         if isinstance(n, str) and len(n) > 1]
                for name in extra:
                    if name not in names:
                        names.append(name)
            except (json.JSONDecodeError, TypeError):
                pass

    return names[:6]


# ── Per-competitor profiling ─────────────────────────────────────────────────

def _profile_competitor(
    name: str,
    brief: str,
    retriever: Retriever,
) -> CompetitorProfile | None:
    """
    Generate a rich profile for one competitor using two evidence sources:
      1. Vector DB chunks (broad market research already ingested)
      2. Live targeted web searches specific to this competitor

    LEARNING: Combining retrieval sources
    Neither source alone is sufficient:
      - Vector DB: may have context mentions but rarely detailed pricing/reviews
      - Live search: fresh and specific but limited by scraping time
    Together they give the LLM maximum evidence to produce a detailed profile.
    """
    # Source 1: vector DB retrieval
    db_chunks = retriever.retrieve_multi_query([
        f"{name} product features capabilities",
        f"{name} pricing customers target market",
        f"{name} weaknesses limitations reviews complaints",
        f"{name} market share revenue position",
    ], k_per_query=3)

    db_context = retriever.format_context(db_chunks, include_source=True) if db_chunks else ""

    # Source 2: live targeted web search
    console.print(f"       [dim]→ Live searching for {name}...[/dim]", end=" ")
    live_context = _live_search_competitor(name)
    console.print("[dim]done[/dim]")

    # Combine both sources
    combined_context_parts = []
    if db_context:
        combined_context_parts.append(f"=== FROM KNOWLEDGE BASE ===\n{db_context}")
    if live_context:
        combined_context_parts.append(f"=== FROM LIVE WEB SEARCH ===\n{live_context}")

    if not combined_context_parts:
        logger.warning(f"No context found for competitor: {name}")
        return None

    combined_context = "\n\n".join(combined_context_parts)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    structured_llm = llm.with_structured_output(CompetitorProfileSchema)

    try:
        result: CompetitorProfileSchema = structured_llm.invoke([
            SystemMessage(content=f"""You are a senior competitive analyst.
Profile the company/product "{name}" using the provided evidence.

IMPORTANT RULES:
- Use ONLY information found in the provided sources
- For pricing: quote any actual numbers found. If none, write "Not publicly disclosed"
- For customer reviews: summarize what users ACTUALLY say in the sources, both good and bad
- For market position/share: use any percentages or rankings mentioned in sources
- For notable customers: only name customers explicitly mentioned
- If information is genuinely absent from sources, be specific: "Not found in sources — likely quote-based" not just "unknown"
- Be precise and factual. This feeds an executive strategy document."""),
            HumanMessage(content=f"""Research Brief:
{brief}

Evidence about {name} (from knowledge base + live web search):
{combined_context[:12000]}

Generate a comprehensive competitor profile for {name}.""")
        ])

        return CompetitorProfile(
            name=name,
            website="",
            product_summary=result.product_summary,
            usp=result.usp,
            core_features=result.core_features,
            pricing_model=result.pricing_model,
            pricing_specifics=result.pricing_specifics,
            market_position=result.market_position,
            market_share_estimate=result.market_share_estimate,
            revenue_estimate=result.revenue_estimate,
            notable_customers=result.notable_customers,
            customer_reviews_summary=result.customer_reviews_summary,
            recent_developments=result.recent_developments,
            target_segment=result.target_segment,
            tech_stack_signals=result.tech_stack_signals,
            release_velocity=result.release_velocity,
            funding=result.funding_stage,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            differentiation_gaps=result.differentiation_gaps,
            source_chunks=[c.chunk_id for c in db_chunks],
        )

    except Exception as e:
        logger.error(f"Failed to profile {name}: {e}")
        return None


# ── Main node ────────────────────────────────────────────────────────────────

def competitive_intel_node(state: MICRAState) -> dict:
    """
    Profile all competitors: known from plan + auto-discovered from sources.
    """
    plan = state.get("research_plan", {})
    collection_name = state.get("vector_db_collection_name", "")
    brief = state.get("research_brief", "")

    if not collection_name:
        return {
            "errors": ["[competitive_intel] No knowledge base available."],
            "messages": ["[competitive_intel] Skipped."]
        }

    console.print("\n[bold cyan]Phase 3b: Competitive Intelligence[/bold cyan]")

    retriever = Retriever(collection_name)

    # Combine known (from plan) + auto-discovered competitors
    known = plan.get("competitor_names_to_research", [])
    console.print(f"  Known competitors from plan: {known or 'none — auto-discovering'}")

    discovered = _discover_competitors_from_sources(brief, retriever)
    console.print(f"  Auto-discovered: {discovered}")

    # Deduplicate (case-insensitive)
    seen = {n.lower() for n in known}
    all_competitors = list(known)
    for name in discovered:
        if name.lower() not in seen:
            seen.add(name.lower())
            all_competitors.append(name)

    all_competitors = all_competitors[:6]  # cap at 6 to manage cost/time

    if not all_competitors:
        return {
            "competitor_profiles": [],
            "errors": ["[competitive_intel] No competitors identified."],
            "messages": ["[competitive_intel] No competitors found to profile."]
        }

    console.print(f"\n  Profiling {len(all_competitors)} competitor(s): "
                  f"{', '.join(all_competitors)}\n")
    console.print("  [dim](Each competitor: vector DB retrieval + live web search)[/dim]\n")

    profiles: list[CompetitorProfile] = []
    errors: list[str] = []

    for name in all_competitors:
        console.print(f"  [cyan]→[/cyan] Profiling [bold]{name}[/bold]...")
        profile = _profile_competitor(name, brief, retriever)
        if profile:
            profiles.append(profile)
            console.print(f"     [green]✓[/green] {name} — "
                          f"{len(profile['core_features'])} features, "
                          f"pricing: {profile['pricing_specifics'][:60]}...")
        else:
            errors.append(f"[competitive_intel] Could not profile '{name}'")
            console.print(f"     [yellow]⚠[/yellow] {name} — no data found")

    console.print(f"\n[green]✓[/green] Profiled [bold]{len(profiles)}[/bold] competitors")

    return {
        "competitor_profiles": profiles,
        "messages": [f"[competitive_intel] Profiled {len(profiles)} competitors "
                     f"(live search + vector DB)."],
        "errors": errors,
    }
