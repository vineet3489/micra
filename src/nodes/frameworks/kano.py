"""
MICRA — Kano Model Agent
===========================

LEARNING CONCEPT: Deriving user needs from market signals

The Kano Model classifies product features into 4 categories:

  MUST-HAVE (Basic needs):
    Features users expect. Their absence causes dissatisfaction.
    Their presence doesn't delight — it's just table stakes.
    Example for DERMS: "Real-time grid monitoring dashboard"

  PERFORMANCE (Linear satisfiers):
    More = better. Users notice and value improvements.
    Example: "Response time to grid events" — faster is always better.

  DELIGHTERS (Excitement features):
    Users don't know they want these. Their presence is a surprise.
    Their absence isn't noticed. These are differentiation opportunities.
    Example: "AI-predicted grid instability 2 hours before it happens"

  INDIFFERENT:
    Features users don't care about either way.
    Building these is waste.

WHY THIS MATTERS FOR MVP:
Must-haves define the minimum viable product (can't ship without them).
Performance features drive competitive positioning.
Delighters create "wow" moments and word of mouth.
Indifferent features should be cut from the roadmap.

LEARNING CONCEPT: Extracting user signals from scraped content

We can infer Kano categories from market signals:
  - "Users complain about X" → X is Must-Have (absence = dissatisfaction)
  - "Users praise Y" → Y is Performance or Delighter
  - "Competitor differentiates with Z" → Z might be a Delighter
  - "Industry standard requires W" → W is Must-Have (compliance)

This is the power of multi-source ingestion — forums, news, competitor
sites, and academic papers all contain signals about what users value.
"""

from pydantic import BaseModel, Field

from src.nodes.frameworks.base import FrameworkAgent


class KanoFeature(BaseModel):
    """A single feature with its Kano classification and evidence."""
    feature_name: str = Field(description="Short, specific feature name")
    description: str = Field(description="What this feature does, in user terms")
    kano_category: str = Field(
        description="One of: 'Must-Have', 'Performance', 'Delighter', 'Indifferent'"
    )
    evidence: str = Field(
        description="What in the context supports this classification? "
                    "Quote or paraphrase specific signals."
    )


class KanoAnalysis(BaseModel):
    """
    Kano model analysis for MVP feature prioritization.

    LEARNING: Why separate MVP from full feature list?

    The must_have list IS your MVP feature set.
    If you ship without any of these, users won't adopt the product.
    Performance features and delighters are v1.1 and v2.
    """
    must_have: list[KanoFeature] = Field(
        description="Features that are table stakes — product fails without them. 4-8 features."
    )
    performance: list[KanoFeature] = Field(
        description="Features where more = better. These drive competitive positioning. 3-6 features."
    )
    delighters: list[KanoFeature] = Field(
        description="Surprise features that create competitive moat. 2-4 features."
    )
    indifferent: list[KanoFeature] = Field(
        description="Features to deliberately cut from roadmap. 2-4 features."
    )
    mvp_recommendation: list[str] = Field(
        description="The 5-7 specific features that should be in v1. "
                    "All Must-Haves plus the highest-impact Performance feature."
    )
    north_star_candidate: str = Field(
        description="The single metric that best measures if users got value from the product. "
                    "Format: '[Metric] per [time period]'"
    )


class KanoAgent(FrameworkAgent):

    name = "kano"

    # LEARNING: Kano queries target user-signal content — forums,
    # reviews, complaints, feature requests, competitor differentiators.
    # We look for signals about what users DO and DON'T value.
    retrieval_queries = [
        "user complaints pain points missing features customer feedback",
        "product features capabilities differentiators competitor comparison",
        "must have requirements regulatory compliance standards mandatory",
        "innovative features AI automation intelligent new capabilities",
        "user satisfaction product adoption customer success case study",
    ]

    output_schema = KanoAnalysis

    system_prompt = """You are a product strategist applying the Kano Model framework.

Kano categories:
  Must-Have: Expected features. Absence = dissatisfaction. Presence = neutral.
  Performance: Linear value. More/better = more satisfaction.
  Delighter: Unexpected features. Presence = surprise delight. Absence = neutral.
  Indifferent: Users don't care either way.

Rules:
- Derive classifications from signals in the context:
  * Customer complaints → Must-Haves (absence = pain)
  * Competitor differentiators → Delighters or Performance
  * Industry standards/compliance → Must-Haves
  * Rarely-used features → Indifferent
- The MVP = all Must-Haves + top 1-2 Performance features
- Delighters are what create competitive moat — give them special attention
- North Star metric should measure the core value exchange, not vanity metrics"""

    def _build_user_prompt(self, research_brief: str, context: str) -> str:
        return f"""Research Brief:
{research_brief}

Context (competitor features, user signals, market data):
{context}

Apply the Kano Model. Classify features from the context into Must-Have,
Performance, Delighter, and Indifferent categories. Recommend a specific
MVP feature set and propose a North Star metric."""
