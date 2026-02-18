"""
MICRA — TAM / SAM / SOM Agent
================================

LEARNING CONCEPT: Triangulated estimation with uncertainty

TAM (Total Addressable Market), SAM (Serviceable Addressable Market),
and SOM (Serviceable Obtainable Market) are market sizing estimates.

The right way to think about these:
  TAM = If you captured 100% of the global market, what's the revenue?
  SAM = Of that, what's realistic given your geography/segment focus?
  SOM = Of SAM, what can you actually capture in years 1-3?

The WRONG approach (what most do): pick a number from one report and cite it.
The RIGHT approach: TRIANGULATE from multiple sources.

Triangulation example for DERMS TAM:
  Source 1 (Mordor Intelligence): "$3.2B by 2027, 18% CAGR"
  Source 2 (Grand View Research): "$2.8B by 2026, 15% CAGR"
  Source 3 (Academic paper): "~500 large utilities in USA × $3-5M avg deal = $1.5-2.5B SAM"
  → Triangulated estimate: "$2.5-3.5B TAM, $1.5-2B SAM for USA utilities"

LEARNING CONCEPT: Expressing uncertainty in structured output

Notice the `confidence` field on MarketSizing. This is important.
Market sizing is inherently uncertain. A good agent:
  1. States the estimate
  2. Lists the assumptions
  3. Expresses confidence clearly
  4. Notes data quality

An AI that outputs "$3.2B" without noting it's based on one analyst report
is more dangerous than one that says "$2-4B (Medium confidence, based on
2 reports — recommend validating with primary research)".

This is calibrated uncertainty — a key concept in AI reliability.
"""

from pydantic import BaseModel, Field

from src.nodes.frameworks.base import FrameworkAgent


class MarketSizing(BaseModel):
    """
    TAM / SAM / SOM analysis with explicit assumptions and confidence.

    LEARNING: Why include key_assumptions?

    Every market size number comes with hidden assumptions:
      - "Global DERMS market" — does this include residential or just utility-scale?
      - "18% CAGR" — from what base year? Under what scenario?

    Making assumptions explicit:
      1. Helps stakeholders evaluate the estimate
      2. Forces the LLM to reason carefully, not just cite numbers
      3. Creates auditability (you can trace each assumption to a source)
    """
    tam_value: str = Field(
        description="TAM value with range e.g. '$2.5B-$3.5B globally by 2027'"
    )
    tam_description: str = Field(
        description="What TAM covers: geography, segments, definition used"
    )
    sam_value: str = Field(
        description="SAM value — the portion addressable given the company's focus"
    )
    sam_description: str = Field(
        description="What limits SAM: geography filter, segment filter, channel"
    )
    som_value: str = Field(
        description="SOM value — realistic capture in years 1-3"
    )
    som_description: str = Field(
        description="Basis for SOM estimate: conversion rate, sales capacity, ramp time"
    )
    growth_rate: str = Field(
        description="Market CAGR with source e.g. '15-18% CAGR (2024-2029)'"
    )
    key_trends_driving_growth: list[str] = Field(
        description="3-5 macro/sector trends that are expanding this market. From context."
    )
    key_assumptions: list[str] = Field(
        description="5+ explicit assumptions underlying these estimates. Be precise."
    )
    data_sources_used: list[str] = Field(
        description="Which sources in the context provided the sizing data"
    )
    confidence: str = Field(
        description="'Low' (1 source, old data), 'Medium' (2-3 sources), or 'High' (3+ sources, recent)"
    )
    confidence_reasoning: str = Field(
        description="Why you assigned this confidence level"
    )


class TAMAgent(FrameworkAgent):

    name = "tam_sam_som"

    # LEARNING: TAM queries target sizing-specific content.
    # We specifically want analyst reports and academic papers
    # (more reliable than company marketing pages for market size).
    retrieval_queries = [
        "market size total addressable market revenue billion CAGR growth rate",
        "market forecast 2024 2025 2026 2027 industry analysis report",
        "number of customers addressable segment enterprise utility scale",
        "market growth drivers trends adoption rate expansion",
        "market revenue estimate analyst report research firm",
    ]

    output_schema = MarketSizing

    system_prompt = """You are a market research analyst performing TAM/SAM/SOM analysis.

Your goal: provide a grounded, triangulated market sizing based on the context.

TAM = Total global opportunity if 100% market share
SAM = Subset addressable given the company's geography and segment focus
SOM = Realistic capture in years 1-3 (typically 1-5% of SAM for a new entrant)

Rules:
- Use numbers from the context. If context has multiple estimates, triangulate.
- Express uncertainty — give ranges, not single point estimates.
- List every assumption explicitly.
- State your confidence level honestly based on source quality and count.
- If sizing data is thin, say so. Low confidence with honest reasoning is
  better than high confidence with manufactured data.
- Do not invent market size numbers. Only use data from the provided context."""

    def _build_user_prompt(self, research_brief: str, context: str) -> str:
        return f"""Research Brief:
{research_brief}

Market Research Context (sources with data):
{context}

Provide TAM/SAM/SOM analysis. Use ONLY the numbers and estimates from the
context above. Triangulate where multiple estimates exist. Be explicit about
every assumption. Express uncertainty honestly."""
