"""
MICRA — SWOT Analysis Agent
==============================

LEARNING CONCEPT: Context-aware vs. generic analysis

SWOT (Strengths, Weaknesses, Opportunities, Threats) is one of the most
abused frameworks — it's easy to generate generic, useless SWOTs.

The difference between a useful and useless SWOT:

  USELESS (what a generic LLM produces):
    Strengths: "Strong technology", "Experienced team"
    Weaknesses: "Limited resources", "New entrant"

  USEFUL (what RAG-grounded analysis produces):
    Strengths: "L&T has existing relationships with 3 of the top 5 Indian
               utilities (Tata Power, NTPC, Adani) — validated by their
               $200M smart grid project portfolio [source: autogrid.com]"

The difference is grounding. The LLM is forced to justify every point
with evidence from the retrieved context.

LEARNING CONCEPT: Company-specific SWOT

A good SWOT is RELATIVE — it's about YOUR company entering THIS market.
The strengths are YOUR strengths vs. what the market requires.
The weaknesses are YOUR gaps vs. what competitors have.

The clarification answers (company context, capabilities) feed into this.
The research brief carries that context to every framework agent.
"""

from pydantic import BaseModel, Field

from src.nodes.frameworks.base import FrameworkAgent


class SWOTAnalysis(BaseModel):
    """
    SWOT analysis relative to the company described in the research brief.

    LEARNING: list[str] fields are a clean pattern for framework outputs.
    Each string should be a complete, specific, cited observation —
    not a vague label.
    """
    strengths: list[str] = Field(
        description="Internal advantages YOUR COMPANY has that are relevant to this market. "
                    "Each must be specific and grounded in the context or brief. 3-5 items."
    )
    weaknesses: list[str] = Field(
        description="Internal gaps or disadvantages YOUR COMPANY has relative to "
                    "what this market requires. Compare to competitors in context. 3-5 items."
    )
    opportunities: list[str] = Field(
        description="External market opportunities (gaps, trends, underserved segments) "
                    "that your company could exploit. Drawn from context. 3-5 items."
    )
    threats: list[str] = Field(
        description="External threats: competitor moves, regulatory risk, technology shifts, "
                    "market dynamics that could harm your position. 3-5 items."
    )
    most_critical_factor: str = Field(
        description="The single most important strategic insight from this SWOT — "
                    "the one thing that determines success or failure in this market."
    )
    recommended_strategic_posture: str = Field(
        description="One of: 'Aggressive (capitalize on strengths+opportunities)', "
                    "'Conservative (address weaknesses first)', "
                    "'Defensive (focus on threat mitigation)', or "
                    "'Turnaround (fix weaknesses before expanding)'"
    )


class SWOTAgent(FrameworkAgent):

    name = "swot"

    retrieval_queries = [
        "company strengths advantages capabilities market position",
        "market gaps underserved customer segments opportunities growth",
        "competitor weaknesses limitations missing features customer complaints",
        "market risks regulatory challenges technology disruption threats",
        "competitive differentiation unique value proposition positioning",
    ]

    output_schema = SWOTAnalysis

    system_prompt = """You are a strategic analyst performing a SWOT analysis.

This SWOT is COMPANY-SPECIFIC — it evaluates the company described in the
research brief (not a generic SWOT of the market).

Rules:
- Strengths and Weaknesses refer to the COMPANY in the research brief.
- Opportunities and Threats refer to the MARKET and external environment.
- Every point must be grounded in the provided context.
- Be specific: name competitors, cite market data, reference technologies.
- Avoid generic statements like "strong team" or "competitive market".
- Each bullet should be 1-2 sentences that could stand alone as an insight."""

    def _build_user_prompt(self, research_brief: str, context: str) -> str:
        return f"""Research Brief (describes the company and their context):
{research_brief}

Market Research Context:
{context}

Perform a SWOT analysis for the company in the brief entering the market
described. Ground every point in the context above."""
