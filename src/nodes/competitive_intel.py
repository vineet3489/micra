"""
MICRA — Competitive Intelligence Node
========================================

LEARNING CONCEPT: Per-entity analysis with filtered retrieval

The framework engine analyzes the whole market. The competitive intel
node drills down into EACH competitor individually.

Key technique: metadata-filtered retrieval.

When we retrieve for Porter's 5 Forces, we want broad context.
When we profile "AutoGrid", we want chunks specifically FROM or ABOUT AutoGrid.

ChromaDB metadata filtering:
    retriever.retrieve("AutoGrid features pricing", source_type="web_competitors")

This returns only chunks from pages with source_type="web_competitors",
which we set when scraping competitor sites.

LEARNING CONCEPT: Iterative structured output per entity

For N competitors, we run N LLM calls. Each produces one CompetitorProfile.
This is more reliable than asking the LLM to return profiles for all
competitors in one call (which often produces inconsistent formatting
and misattributes features across competitors).

"One LLM call per entity" is a common pattern for reliable extraction.

LEARNING CONCEPT: Competitor discovery vs. known competitors

The research_plan has two sources of competitors:
  1. competitor_names_to_research — known competitors from clarification
  2. Auto-discovered from scraped content (mentioned in articles, etc.)

For MVP, we profile known competitors + try to extract names from
scraped content using a quick LLM call.
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
from src.config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)
console = Console()


# ── Pydantic schema for competitor profile ─────────────────────────────────
class CompetitorProfileSchema(BaseModel):
    """
    LEARNING: This schema is used with with_structured_output().

    Notice the Field descriptions are very specific — they tell the LLM
    exactly what format each field should be in. Vague descriptions lead
    to inconsistent outputs across different competitors.
    """
    product_summary: str = Field(
        description="1-2 sentence description of what this company's product does"
    )
    core_features: list[str] = Field(
        description="5-8 specific product features, named precisely as the company describes them"
    )
    pricing_model: str = Field(
        description="Pricing structure: e.g., 'Enterprise SaaS, per-site licensing' or 'Usage-based API pricing'"
    )
    target_segment: str = Field(
        description="Primary customer segment: e.g., 'Tier 1 US utilities with >1GW capacity'"
    )
    tech_stack_signals: list[str] = Field(
        description="Technology indicators from job listings, docs, or marketing. "
                    "e.g., ['REST API', 'AWS', 'Python SDK', 'SCADA integration']"
    )
    release_velocity: str = Field(
        description="Estimated release cadence based on changelog/news. "
                    "e.g., 'Monthly releases, major version quarterly'"
    )
    funding_stage: str = Field(
        description="Funding information if available. e.g., '$45M Series C (2023)' or 'Unknown/Bootstrapped'"
    )
    strengths: list[str] = Field(
        description="3-5 genuine competitive strengths drawn from the context"
    )
    weaknesses: list[str] = Field(
        description="3-5 weaknesses or gaps visible from context (missing features, poor reviews, etc.)"
    )
    differentiation_gaps: list[str] = Field(
        description="3-5 specific opportunities where a new entrant could beat this competitor"
    )


def _extract_competitor_names_from_sources(
    brief: str,
    retriever: Retriever,
) -> list[str]:
    """
    Auto-discover competitor names from scraped content.

    LEARNING: This is a "classification" LLM call — asking the LLM to
    extract structured information (company names) from unstructured text.
    Much more reliable than regex or keyword matching.

    We limit to a quick retrieve + parse — not a full profile.
    """
    chunks = retriever.retrieve(
        "competitor company product vendor platform provider market leader",
        k=6
    )
    if not chunks:
        return []

    context = "\n\n".join(c.text for c in chunks[:4])

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke([
        SystemMessage(content="Extract company/product names that are competitors in this market. "
                              "Return a JSON array of name strings only. Max 5 names. "
                              "Only include companies that clearly have a competing product."),
        HumanMessage(content=f"Context:\n{context}\n\nReturn JSON array of competitor names.")
    ])

    try:
        names = json.loads(response.content)
        return [n for n in names if isinstance(n, str) and len(n) > 1][:5]
    except (json.JSONDecodeError, TypeError):
        return []


def _profile_competitor(
    name: str,
    brief: str,
    retriever: Retriever,
) -> CompetitorProfile | None:
    """
    Generate a structured profile for one competitor.

    LEARNING: Targeted retrieval for a specific entity.

    We combine competitor name + feature-type queries to retrieve
    chunks that are specifically about this competitor.
    Multiple queries cast a wider net — the competitor might be
    mentioned in different contexts (feature comparison, news, etc.)
    """
    queries = [
        f"{name} product features capabilities",
        f"{name} pricing model customers target segment",
        f"{name} technology platform architecture",
        f"{name} weaknesses limitations customer complaints",
    ]

    chunks = retriever.retrieve_multi_query(queries, k_per_query=4)

    if not chunks:
        logger.warning(f"No context found for competitor: {name}")
        return None

    context = retriever.format_context(chunks, include_source=True)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    structured_llm = llm.with_structured_output(CompetitorProfileSchema)

    try:
        result: CompetitorProfileSchema = structured_llm.invoke([
            SystemMessage(content=f"""You are a competitive analyst.
Profile the company "{name}" using ONLY the provided context.
If information is not in the context, say "Not found in sources".
Do not invent features, pricing, or capabilities."""),
            HumanMessage(content=f"""Research Brief: {brief}

Context about {name}:
{context}

Generate a complete competitor profile for {name}.""")
        ])

        return CompetitorProfile(
            name=name,
            website="",  # populated from search results if available
            product_summary=result.product_summary,
            core_features=result.core_features,
            pricing_model=result.pricing_model,
            target_segment=result.target_segment,
            tech_stack_signals=result.tech_stack_signals,
            release_velocity=result.release_velocity,
            funding=result.funding_stage,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            differentiation_gaps=result.differentiation_gaps,
            source_chunks=[c.chunk_id for c in chunks],
        )

    except Exception as e:
        logger.error(f"Failed to profile {name}: {e}")
        return None


def competitive_intel_node(state: MICRAState) -> dict:
    """
    Profile all competitors found in the plan and auto-discovered from sources.
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

    # Combine known + auto-discovered competitors
    known = plan.get("competitor_names_to_research", [])
    discovered = _extract_competitor_names_from_sources(brief, retriever)

    # Deduplicate (case-insensitive)
    seen = {n.lower() for n in known}
    all_competitors = list(known)
    for name in discovered:
        if name.lower() not in seen:
            seen.add(name.lower())
            all_competitors.append(name)

    all_competitors = all_competitors[:6]  # cap at 6 to manage cost/time

    console.print(f"  Profiling {len(all_competitors)} competitor(s): {', '.join(all_competitors)}\n")

    profiles: list[CompetitorProfile] = []
    errors: list[str] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        for name in all_competitors:
            task = progress.add_task(f"Profiling {name}...", total=None)
            profile = _profile_competitor(name, brief, retriever)
            if profile:
                profiles.append(profile)
                progress.update(task, description=f"[green]✓[/green] {name}", total=1, completed=1)
            else:
                errors.append(f"[competitive_intel] Could not profile '{name}' — no data found")
                progress.update(task, description=f"[yellow]⚠[/yellow] {name} (no data)", total=1, completed=1)

    console.print(f"\n[green]✓[/green] Profiled [bold]{len(profiles)}[/bold] competitors")

    return {
        "competitor_profiles": profiles,
        "messages": [f"[competitive_intel] Profiled {len(profiles)} competitors."],
        "errors": errors,
    }
