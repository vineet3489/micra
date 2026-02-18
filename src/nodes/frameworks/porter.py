"""
MICRA — Porter's 5 Forces Agent
==================================

LEARNING CONCEPT: Grounded strategic analysis

Porter's 5 Forces is a framework for assessing the competitive intensity
and attractiveness of an industry. The 5 forces are:

  1. Threat of New Entrants    — How easy is it for new players to enter?
  2. Bargaining Power (Buyers) — How much leverage do customers have?
  3. Bargaining Power (Suppliers) — How much leverage do vendors/partners have?
  4. Threat of Substitutes     — Can customers switch to alternatives easily?
  5. Competitive Rivalry       — How intense is competition among existing players?

WITHOUT RAG: The LLM would answer from training data — potentially
outdated, hallucinated, or generic (not specific to your market/geography).

WITH RAG: The LLM's analysis is anchored in documents you scraped.
Every claim can be traced to a source chunk (citation).

This is the core value proposition of RAG for strategic analysis.

LEARNING CONCEPT: Pydantic schema as "output contract"

When you define a Pydantic schema, you're writing a CONTRACT with the LLM:
"You MUST return data in exactly this shape."

The LLM reads the schema (field names + descriptions) as instructions.
Good field descriptions = better LLM outputs.

Bad:  rating: str
Good: rating: str = Field(description="Competitive intensity: Low | Medium | High")

The description tells the LLM what valid values look like.
"""

from pydantic import BaseModel, Field
from langchain.tools import tool

from src.nodes.frameworks.base import FrameworkAgent


# ── Output Schema ──────────────────────────────────────────────────────────
class Force(BaseModel):
    """A single Porter force assessment."""
    rating: str = Field(
        description="Intensity level: 'Low', 'Medium', or 'High'"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation grounded in the provided context"
    )
    key_factors: list[str] = Field(
        description="3-5 specific factors driving this rating, drawn from sources"
    )


class PorterAnalysis(BaseModel):
    """
    Complete Porter's 5 Forces analysis.

    LEARNING: Nested Pydantic models

    Force is reused for all 5 forces — DRY schema design.
    Nesting models is how you represent structured, hierarchical output.
    The LLM will fill in all nested fields correctly because
    with_structured_output() sends the full nested schema.
    """
    threat_of_new_entrants: Force = Field(
        description="How easy is it for new competitors to enter this market?"
    )
    bargaining_power_buyers: Force = Field(
        description="How much leverage do buyers/customers have over pricing and terms?"
    )
    bargaining_power_suppliers: Force = Field(
        description="How much leverage do key suppliers or technology vendors have?"
    )
    threat_of_substitutes: Force = Field(
        description="How easily can customers switch to alternative solutions?"
    )
    competitive_rivalry: Force = Field(
        description="How intense is the competition among existing players?"
    )
    overall_market_attractiveness: str = Field(
        description="Summary verdict: 'Low', 'Medium', or 'High' attractiveness, with 1-sentence reason"
    )
    strategic_implications: list[str] = Field(
        description="3-5 strategic implications for a new entrant based on this analysis"
    )


# ── Agent ──────────────────────────────────────────────────────────────────
class PorterAgent(FrameworkAgent):

    name = "porter_5_forces"

    # LEARNING: Retrieval queries are designed to surface content relevant
    # to Porter's forces specifically. We use 5 queries — one per force —
    # to maximize the chance of retrieving content for each dimension.
    # retrieve_multi_query() deduplicates results across all queries.
    retrieval_queries = [
        "market entry barriers regulations capital requirements new entrants",
        "customer buyer power switching costs price sensitivity",
        "supplier technology vendor dependency lock-in platform",
        "substitute alternatives competing products pricing comparison",
        "competitive rivalry market share competition landscape players",
    ]

    output_schema = PorterAnalysis

    system_prompt = """You are a strategic analyst applying Porter's 5 Forces framework.

Your task: analyze the competitive forces in a specific market using ONLY the
provided context (scraped market data, competitor information, and research).

Rules:
- Ground every rating and factor in the provided context.
- If context is insufficient for a force, rate it "Medium" and note limited data.
- Be specific — name companies, regulations, and technologies from the context.
- Strategic implications must be actionable for a new market entrant.
- Do NOT use generic textbook examples. Use only what's in the context."""

    def _build_user_prompt(self, research_brief: str, context: str) -> str:
        return f"""Research Brief:
{research_brief}

Context (from market research sources):
{context}

Apply Porter's 5 Forces to this market. For each force, provide a rating
(Low/Medium/High), reasoning grounded in the context, and key specific factors.
Then give an overall market attractiveness assessment and strategic implications."""
