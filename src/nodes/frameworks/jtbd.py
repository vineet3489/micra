"""
MICRA — Jobs-to-be-Done Framework Agent
=========================================

LEARNING CONCEPT: Jobs-to-be-Done (JTBD)

JTBD is a framework for understanding WHY customers buy products.
The core insight: customers don't buy products — they "hire" them
to do a job they need done.

Classic example: people don't buy a drill, they hire it to make a hole.
They don't want the hole — they want a shelf on the wall.

For market research, JTBD answers:
  - What are customers trying to accomplish?
  - What's frustrating about how they do it today?
  - What would "done perfectly" look like?

This is more actionable than demographic segmentation because it
tells you what to build, not just who might buy it.

THE JOB STATEMENT FORMAT:
  When [context/situation]
  I want to [motivation/goal]
  So I can [expected outcome]

Each job has:
  - Pain points with current solutions
  - Success metrics (how customers measure "done")
  - Current workarounds being used

WHY THIS BENEFITS FROM RAG:
JTBD requires evidence from customer voice — reviews, forums,
case studies, interviews. Our web scraper may have pulled:
  - Product reviews mentioning pain points
  - Forum discussions about workflow frustrations
  - Case studies describing customer goals
  - Competitor testimonials that reveal what customers valued

Grounding JTBD in real scraped evidence makes it far more credible
than asking the LLM to guess what customers want.
"""

from pydantic import BaseModel, Field

from src.nodes.frameworks.base import FrameworkAgent


# ── Output schema ───────────────────────────────────────────────────────────

class CustomerJob(BaseModel):
    job_name: str = Field(
        description="Short name for this job: e.g., 'Monitor Real-Time Operations'"
    )
    when_context: str = Field(
        description="The situation or trigger: 'When [context]...'"
    )
    desired_outcome: str = Field(
        description="What the customer wants to achieve: 'I want to [goal]...'"
    )
    so_that: str = Field(
        description="The ultimate outcome: 'So I can [outcome]...'"
    )
    current_solutions: list[str] = Field(
        description="2-3 existing tools/approaches customers use today (from sources)"
    )
    pain_points: list[str] = Field(
        description="3-4 specific frustrations with current solutions (from sources)"
    )
    success_metrics: list[str] = Field(
        description="2-3 measurable outcomes that define 'job done perfectly'"
    )


class JTBDOutput(BaseModel):
    primary_jobs: list[CustomerJob] = Field(
        description="The 4-6 most important jobs customers hire this product to do. "
                    "Order by frequency and importance. Ground each in source evidence."
    )
    primary_persona: dict = Field(
        description="The most important buyer persona as a dict with keys: "
                    "name, role, age_range, daily_challenges, goals, decision_criteria. "
                    "Should be specific and grounded in source evidence."
    )
    secondary_persona: dict = Field(
        description="The secondary buyer persona (e.g., IT manager vs. operator) with the "
                    "same keys as primary_persona."
    )
    insight_summary: str = Field(
        description="2-3 sentence synthesis: what is the #1 underserved job, and what "
                    "does that mean for product strategy?"
    )


# ── Framework agent ─────────────────────────────────────────────────────────

class JTBDAgent(FrameworkAgent):
    """
    Jobs-to-be-Done analysis agent.

    Retrieves customer-voice evidence (reviews, case studies, forum posts)
    and synthesizes the core jobs customers hire this type of product to do.
    """

    framework_name = "jtbd"
    output_schema = JTBDOutput

    retrieval_queries = [
        "customer pain points frustrations problems with current solution",
        "why customers buy switch to this type of product",
        "user reviews testimonials what they love hate",
        "customer workflows how do operators use historian daily",
        "jobs to be done customer needs use cases",
    ]

    system_prompt = """You are a product researcher specializing in Jobs-to-be-Done analysis.

Your task: analyze the scraped market research evidence to identify the core jobs
that customers hire this type of product to do.

JTBD principles to apply:
- A "job" is a problem a customer is trying to solve, not a feature request
- Jobs are stable over time; solutions change
- Focus on the functional job AND the emotional/social job
- Ground every job and pain point in evidence from the sources
- "Current solutions" should name specific tools mentioned in the sources

Output exactly 4-6 customer jobs, ordered from most to least critical.
Be specific — "monitor real-time grid data" not "get data"."""

    def _build_user_prompt(self, brief: str, context: str) -> str:
        return f"""Research Brief:
{brief}

Evidence from market research (scraped sources):
{context}

Identify the core jobs customers hire this product to do.
Ground each job in the evidence above — quote or paraphrase specific
pain points, frustrations, and goals you found in the sources."""
