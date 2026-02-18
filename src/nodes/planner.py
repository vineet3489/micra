"""
MICRA — Research Planner Node
================================

LEARNING CONCEPT: Agent Planning (ReAct pattern)

The planner demonstrates a key AI agent concept: reasoning before acting.

Before scraping anything, the agent thinks:
  "Given this research brief, what do I need to find?"
  "Which sources are most likely to have this information?"
  "Which frameworks should I apply?"
  "What specific sub-questions must I answer?"

This is the "Reason" step in the ReAct (Reason + Act) pattern.
ReAct is one of the most important agent patterns to understand:

  Reason → Act → Observe → Reason → Act → Observe → ...

In our pipeline it's simplified to a single planning pass, but the
concept is the same: the agent structures its own task before executing.

WHY PLAN?
---------
Without planning, agents either:
  A) Do too much (scrape everything, apply all frameworks = expensive + slow)
  B) Do too little (miss important source types or frameworks)

A planning step ensures the agent uses the minimum necessary resources
to answer the specific research question.

LEARNING CONCEPT: Structured output with Pydantic

Here we introduce Pydantic for the first time. Instead of asking the LLM
to return raw JSON and parsing it manually (like in clarification.py),
we use Pydantic to:
  1. Define what the output should look like (the schema)
  2. Ask the LLM to return structured output matching that schema
  3. Get back a typed Python object (not a raw dict)

This is the right pattern for production — you get validation for free.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table

from src.state import MICRAState

console = Console()


# ── Output schema (what the planner must produce) ──────────────────────────
class ResearchPlan(BaseModel):
    """
    LEARNING: Pydantic BaseModel as structured output schema.

    When you call `llm.with_structured_output(ResearchPlan)`, LangChain
    tells the LLM to return JSON matching this schema. The LLM is
    constrained to produce valid, typed output.

    Field(...) lets you add descriptions — these go into the schema
    sent to the LLM, which helps it understand what to fill in.
    """

    target_market: str = Field(description="The primary market being researched")
    geography: str = Field(description="Geographic scope of the research")
    company_context: str = Field(description="The company doing the analysis and their context")

    source_types_to_query: list[str] = Field(
        description="List of source types to query. Options: web_competitors, academic_papers, news, regulatory, funding"
    )
    competitor_names_to_research: list[str] = Field(
        description="Specific competitor names to research if known, or empty list if auto-discovery needed"
    )
    search_queries: list[str] = Field(
        description="5-8 specific search queries to use when fetching sources"
    )

    frameworks_to_apply: list[str] = Field(
        description="Strategic frameworks to apply. Options: tam_sam_som, porter_5_forces, swot, kano, bcg, ansoff, north_star, build_buy_partner"
    )
    sub_questions: list[str] = Field(
        description="Specific questions the research must answer to complete the brief"
    )

    estimated_complexity: str = Field(
        description="low | medium | high — based on scope of research needed"
    )


def planner_node(state: MICRAState) -> dict:
    """
    LEARNING: Note how this node is simpler than clarification_node.
    No user interaction — pure LLM reasoning.

    It reads the research_brief (set by the previous node) and produces
    a structured plan that all downstream nodes will follow.

    Key pattern: EACH NODE READS FROM STATE, DOES ONE JOB, WRITES BACK.
    The planner's one job: convert brief → actionable research plan.
    """

    console.print("\n[dim]Planning research strategy...[/dim]")

    # LEARNING: with_structured_output() is LangChain's way of enforcing
    # a Pydantic schema on LLM output. Under the hood it uses OpenAI's
    # function calling / tool use feature to guarantee valid JSON.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(ResearchPlan)

    system_prompt = """You are a strategic research planner for market intelligence.
Given a research brief, generate a precise research plan.

For source types, choose from: web_competitors, academic_papers, news, regulatory, funding
For frameworks, choose only what's relevant to the question. Don't apply BCG if there's
no market positioning question. Don't apply North Star if there's no product decision.

Generate search_queries that are specific enough to return useful results.
Bad: "DERMS software"
Good: "DERMS distributed energy resource management system USA market leaders 2024"

For sub_questions, make them answerable — each should be a question the research
will explicitly answer in the final report."""

    plan: ResearchPlan = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research brief:\n\n{state['research_brief']}")
    ])

    # Display the plan nicely
    _display_plan(plan)

    # LEARNING: We store the plan as a dict in state.
    # Downstream nodes (scraper, framework agents) will read specific fields.
    # We serialize the Pydantic model to dict with model_dump().
    return {
        "research_plan": plan.model_dump(),
        "messages": [
            f"[planner] Plan created. "
            f"Frameworks: {', '.join(plan.frameworks_to_apply)}. "
            f"Sources: {', '.join(plan.source_types_to_query)}. "
            f"Search queries: {len(plan.search_queries)}."
        ]
    }


def _display_plan(plan: ResearchPlan) -> None:
    """Display the research plan in a readable table."""

    table = Table(title="Research Plan", border_style="blue", show_header=True)
    table.add_column("Field", style="cyan", width=25)
    table.add_column("Value", style="white")

    table.add_row("Target Market", plan.target_market)
    table.add_row("Geography", plan.geography)
    table.add_row("Company Context", plan.company_context)
    table.add_row("Source Types", "\n".join(f"• {s}" for s in plan.source_types_to_query))
    table.add_row("Frameworks", "\n".join(f"• {f}" for f in plan.frameworks_to_apply))
    table.add_row("Known Competitors", "\n".join(f"• {c}" for c in plan.competitor_names_to_research) or "Auto-discover")
    table.add_row("Search Queries", "\n".join(f"• {q}" for q in plan.search_queries))
    table.add_row("Sub-Questions", "\n".join(f"• {q}" for q in plan.sub_questions))
    table.add_row("Complexity", plan.estimated_complexity.upper())

    console.print(table)
