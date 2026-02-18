"""
MICRA — Graph State Definition
================================

LEARNING CONCEPT: What is "state" in a LangGraph agent?

In LangGraph, your entire agent is a graph of nodes (functions) connected
by edges. The "state" is a shared data structure that flows through every
node. Each node reads from it and writes back to it.

Think of it like a baton in a relay race — every runner (node) receives
the baton, does their work, and passes it forward. The baton carries
everything accumulated so far.

WHY TypedDict?
--------------
Python's TypedDict gives us:
  1. Type hints — your IDE knows what's inside state at every step
  2. No overhead — it's just a dict at runtime, no class instantiation
  3. LangGraph expects this pattern (or Pydantic models)

HOW STATE UPDATES WORK:
-----------------------
In LangGraph, nodes return a *partial* state update — only the fields
they modified. LangGraph merges this back into the full state.

Example:
    # Node returns this:
    return {"research_brief": "Analyze DERMS market in USA"}

    # LangGraph merges it into state — all other fields stay unchanged.

For list fields (like `sources`, `messages`), you can use
`Annotated[list, operator.add]` — this means "append, don't replace".
We use this for messages and sources so nodes ADD to the list rather
than overwriting it.
"""

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class ClarificationAnswer(TypedDict):
    """A single question-answer pair from the clarification phase."""
    question: str
    answer: str


class CompetitorProfile(TypedDict):
    """Structured profile for one competitor."""
    name: str
    website: str
    product_summary: str
    usp: str                       # unique selling proposition — their #1 claim
    core_features: list[str]
    pricing_model: str
    pricing_specifics: str         # actual numbers found ("$200K license", "$40K/yr")
    market_position: str           # "Market Leader (40-45% share)", "Challenger", etc.
    market_share_estimate: str     # % or qualitative if no % found
    revenue_estimate: str          # ARR/revenue if available publicly
    notable_customers: list[str]   # customer names/logos mentioned in sources
    customer_reviews_summary: str  # what users say on G2/Capterra/forums
    recent_developments: list[str] # recent news, product launches, funding rounds
    target_segment: str
    tech_stack_signals: list[str]
    release_velocity: str
    funding: str
    strengths: list[str]
    weaknesses: list[str]
    differentiation_gaps: list[str]
    source_chunks: list[str]  # chunk IDs used — enables citation tracing


class SourceDocument(TypedDict):
    """A document ingested from any source."""
    url: str
    source_type: str        # "web", "academic", "news", "regulatory"
    title: str
    raw_text: str
    chunk_ids: list[str]    # IDs of chunks stored in vector DB


class FrameworkOutput(TypedDict):
    """
    Output from any strategic framework agent.
    Generic enough to hold any framework's result.
    """
    framework_name: str     # e.g. "porter_5_forces", "swot", "tam"
    output: dict[str, Any]  # The actual structured output (varies per framework)
    source_chunks: list[str]
    confidence: float       # 0.0 to 1.0


class EvaluationResult(TypedDict):
    """AI evaluation of the final output."""
    overall_score: float
    factual_grounding: float
    framework_correctness: float
    completeness: float
    strategic_coherence: float
    hallucinations_detected: int
    flagged_claims: list[str]
    passed: bool            # True if overall_score >= 0.75
    recommendation: str


class MICRAState(TypedDict):
    """
    The full graph state for MICRA.

    Every node receives this and returns a partial update.
    Fields are grouped by which phase populates them.

    PHASE 1 — Clarification & Planning:
        raw_query, clarification_answers, research_brief

    PHASE 2 — Data Ingestion:
        sources, vector_db_ready

    PHASE 3 — Framework Analysis:
        framework_outputs, competitor_profiles

    PHASE 4 — Synthesis:
        build_buy_partner_decision, mvp_recommendation

    PHASE 5 — Report:
        report_path

    PHASE 6 — Evaluation:
        evaluation

    ALWAYS PRESENT:
        messages — the running log of agent activity
        errors   — any errors encountered (nodes don't crash, they log)
    """

    # ── Input ──────────────────────────────────────────────────────────────
    raw_query: str
    # Example: "Analyze the DERMS market in USA. Should L&T build or partner?"

    # ── Phase 1: Clarification ─────────────────────────────────────────────
    clarification_answers: list[ClarificationAnswer]
    research_brief: str
    # The research_brief is a structured summary derived from
    # raw_query + clarification_answers. All downstream nodes use this.

    research_plan: dict[str, Any]
    # Produced by planner_node. All downstream nodes read from this.

    # ── Phase 2: Data Ingestion ────────────────────────────────────────────
    sources: Annotated[list[SourceDocument], operator.add]
    # Annotated with operator.add means nodes APPEND to this list.
    # Without this, each node would overwrite the whole list.
    vector_db_ready: bool
    vector_db_collection_name: str  # ChromaDB collection name for this run

    # ── Phase 3: Analysis ──────────────────────────────────────────────────
    framework_outputs: list[FrameworkOutput]
    competitor_profiles: list[CompetitorProfile]

    # ── Phase 4: Synthesis ─────────────────────────────────────────────────
    build_buy_partner_decision: dict[str, Any]
    mvp_recommendation: dict[str, Any]
    gtm_strategy: dict[str, Any]
    # {
    #   "primary_target_segments": [...],
    #   "go_to_market_by_region": [...],
    #   "pricing_options": [...],
    #   "revenue_projections": {...},
    #   "competitive_messaging": [...]
    # }
    team_requirements: dict[str, Any]
    # {
    #   "core_team_roles": [...],
    #   "total_headcount": "...",
    #   "critical_skills": [...],
    #   "extended_team": [...]
    # }
    phased_roadmap: dict[str, Any]
    # {
    #   "mvp_phase": {...},
    #   "phase_2": {...},
    #   "phase_3": {...},
    #   "phase_4": {...},
    #   "next_steps": {...}
    # }

    # ── Phase 5: Report ────────────────────────────────────────────────────
    report_path: str

    # ── Phase 6: Evaluation ────────────────────────────────────────────────
    evaluation: EvaluationResult

    # ── Always present ─────────────────────────────────────────────────────
    messages: Annotated[list[str], operator.add]
    # Running log: each node appends what it did.
    # e.g. ["[clarification] Asked 6 questions", "[scraper] Ingested 12 sources"]

    errors: Annotated[list[str], operator.add]
    # Non-fatal errors get logged here. The graph keeps running.
    # Fatal errors raise exceptions (graph stops).
