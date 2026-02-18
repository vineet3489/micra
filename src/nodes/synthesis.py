"""
MICRA — Synthesis Node
=========================

LEARNING CONCEPT: Agent synthesis — combining multiple agent outputs

This node is special: it doesn't retrieve from the knowledge base.
It works entirely from the ACCUMULATED STATE — taking outputs from
all previous nodes and synthesizing them into final recommendations.

This demonstrates a key architectural principle:
  EARLY NODES build the knowledge base and do atomic analysis.
  LATE NODES synthesize across all previous outputs.

The synthesis agent sees:
  - research_brief (what was asked)
  - framework_outputs (Porter, SWOT, TAM, Kano results)
  - competitor_profiles (structured competitor data)

And produces:
  - build_buy_partner_decision
  - mvp_recommendation

LEARNING CONCEPT: Chain-of-thought at scale

With all that context, we use a two-call pattern:
  1. First call: reason through Build/Buy/Partner
  2. Second call: derive MVP from framework outputs

Why two calls? One massive prompt for both decisions is less reliable
than two focused prompts. Splitting the reasoning into sequential steps
(chain-of-thought) produces better decisions.

This mirrors how a human strategist works:
  Step 1: "Should we enter at all, and how?"
  Step 2: "OK, we're building — what exactly do we build?"
"""

import json
import logging
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.state import MICRAState
from src.config import LLM_MODEL, LLM_TEMPERATURE_CREATIVE

logger = logging.getLogger(__name__)
console = Console()


# ── Output schemas ─────────────────────────────────────────────────────────
class BuildBuyPartnerSchema(BaseModel):
    recommendation: str = Field(
        description="Primary recommendation: 'Build', 'Buy', 'Partner', or 'Build+Partner'"
    )
    reasoning: str = Field(
        description="3-5 sentence strategic rationale referencing the framework outputs"
    )
    capability_gaps: list[str] = Field(
        description="What capabilities does the company lack that this market requires? 3-5 items."
    )
    time_to_market_assessment: str = Field(
        description="How long to reach first revenue with this approach, and why"
    )
    build_case: str = Field(description="Best argument FOR building in-house")
    buy_case: str = Field(description="Best argument FOR acquiring a player")
    partner_case: str = Field(description="Best argument FOR a strategic partnership")
    recommended_partners_or_targets: list[str] = Field(
        description="3-5 specific companies to partner with or acquire, from competitor profiles"
    )
    risk_factors: list[str] = Field(
        description="Top 4 risks with the recommended approach, drawn from analysis"
    )


class MVPSchema(BaseModel):
    target_customer: str = Field(
        description="Specific ICP: e.g., 'Grid operations managers at US utilities with 500MW-5GW capacity'"
    )
    core_features: list[str] = Field(
        description="The 5-7 specific features to ship in v1. Must align with Kano must-haves."
    )
    features_explicitly_excluded: list[str] = Field(
        description="3-5 features to deliberately NOT build in v1 and why"
    )
    north_star_metric: str = Field(
        description="The single metric that proves the product is delivering value. "
                    "Format: '[Measurable outcome] per [time unit]'"
    )
    go_to_market_entry_point: str = Field(
        description="Where to start: which customer segment, which geography, which channel"
    )
    pricing_strategy: str = Field(
        description="Recommended pricing model and rationale: e.g., 'Per-site SaaS, $X/month, "
                    "because competitors use usage-based and SMBs find it unpredictable'"
    )
    success_criteria_6_months: list[str] = Field(
        description="3-4 measurable milestones that would validate the MVP at 6 months"
    )
    differentiation_thesis: str = Field(
        description="1-2 sentences: the unique angle that justifies entering despite existing competition"
    )


def _serialize_framework_outputs(framework_outputs: list) -> str:
    """
    Convert framework outputs to a readable summary for the synthesis prompt.

    LEARNING: How to pass structured agent outputs to a downstream agent.

    We can't just dump raw JSON — it's noisy and hard for the LLM to reason about.
    Instead we format each output as a labeled section with the key insights.
    This is "context compression" — extracting the signal from structured data.
    """
    parts = []
    for output in framework_outputs:
        name = output["framework_name"].replace("_", " ").upper()
        data = output["output"]
        confidence = output["confidence"]
        parts.append(
            f"=== {name} (confidence: {confidence:.0%}) ===\n"
            f"{json.dumps(data, indent=2)}"
        )
    return "\n\n".join(parts)


def _serialize_competitor_profiles(profiles: list) -> str:
    """Format competitor profiles for synthesis prompt."""
    parts = []
    for p in profiles:
        parts.append(
            f"Competitor: {p['name']}\n"
            f"  Product: {p['product_summary']}\n"
            f"  Strengths: {', '.join(p['strengths'][:3])}\n"
            f"  Weaknesses: {', '.join(p['weaknesses'][:3])}\n"
            f"  Gaps: {', '.join(p['differentiation_gaps'][:3])}"
        )
    return "\n\n".join(parts)


def synthesis_node(state: MICRAState) -> dict:
    """
    Synthesize all framework outputs into final strategic recommendations.

    LEARNING: This node shows how a "meta-agent" works.
    It doesn't retrieve or scrape. It reasons over the outputs of
    all previous agents — combining perspectives into decisions.

    The synthesis agent is essentially doing what a senior strategist does
    after reading the analysis team's reports.
    """
    framework_outputs = state.get("framework_outputs", [])
    competitor_profiles = state.get("competitor_profiles", [])
    brief = state.get("research_brief", "")

    if not framework_outputs:
        return {
            "errors": ["[synthesis] No framework outputs to synthesize."],
            "messages": ["[synthesis] Skipped — no analysis available."]
        }

    console.print("\n[bold cyan]Phase 3c: Strategic Synthesis[/bold cyan]")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE_CREATIVE)

    framework_summary = _serialize_framework_outputs(framework_outputs)
    competitor_summary = _serialize_competitor_profiles(competitor_profiles)

    # ── Call 1: Build / Buy / Partner decision ────────────────────────────
    console.print("  [cyan]→[/cyan] Generating Build/Buy/Partner recommendation...", end=" ")

    try:
        bbp_llm = llm.with_structured_output(BuildBuyPartnerSchema)
        bbp_result: BuildBuyPartnerSchema = bbp_llm.invoke([
            SystemMessage(content="""You are a senior strategy consultant.
Your task: recommend whether the company in the brief should Build, Buy, Partner,
or some combination, to enter the described market.

Base your recommendation on:
- The framework outputs (Porter, SWOT, TAM, Kano analysis)
- The competitor profiles
- The company's described capabilities in the research brief

Be decisive. State a clear recommendation and defend it."""),
            HumanMessage(content=f"""Research Brief:
{brief}

Framework Analysis Results:
{framework_summary}

Competitor Landscape:
{competitor_summary}

Provide a Build/Buy/Partner recommendation.""")
        ])
        console.print("[green]✓[/green]")
        bbp_dict = bbp_result.model_dump()

    except Exception as e:
        logger.error(f"Build/Buy/Partner synthesis failed: {e}")
        bbp_dict = {"error": str(e)}
        console.print(f"[red]✗[/red]")

    # ── Call 2: MVP recommendation ─────────────────────────────────────────
    console.print("  [cyan]→[/cyan] Generating MVP recommendation...", end=" ")

    try:
        mvp_llm = llm.with_structured_output(MVPSchema)
        mvp_result: MVPSchema = mvp_llm.invoke([
            SystemMessage(content="""You are a senior product strategist.
Your task: define the MVP for this product based on:
- Kano analysis (must-haves define the MVP floor)
- Competitive gaps (differentiation opportunities)
- Market sizing (what customer segment offers fastest path to revenue)
- The Build/Buy/Partner decision (shapes the build scope)

Be specific. The MVP features should be concrete enough to put in a sprint."""),
            HumanMessage(content=f"""Research Brief:
{brief}

Framework Analysis:
{framework_summary}

Competitor Gaps:
{competitor_summary}

Build/Buy/Partner Decision:
{json.dumps(bbp_dict, indent=2)}

Define the MVP.""")
        ])
        console.print("[green]✓[/green]")
        mvp_dict = mvp_result.model_dump()

    except Exception as e:
        logger.error(f"MVP synthesis failed: {e}")
        mvp_dict = {"error": str(e)}
        console.print(f"[red]✗[/red]")

    # ── Display key outputs ────────────────────────────────────────────────
    if "recommendation" in bbp_dict:
        console.print(f"\n  [bold]Recommendation:[/bold] [green]{bbp_dict['recommendation']}[/green]")
    if "north_star_metric" in mvp_dict:
        console.print(f"  [bold]North Star:[/bold] {mvp_dict.get('north_star_metric', 'TBD')}")
    if "core_features" in mvp_dict:
        console.print(f"  [bold]MVP features:[/bold] {len(mvp_dict.get('core_features', []))} identified")

    return {
        "build_buy_partner_decision": bbp_dict,
        "mvp_recommendation": mvp_dict,
        "messages": [
            f"[synthesis] Build/Buy/Partner: {bbp_dict.get('recommendation', 'error')}. "
            f"MVP: {len(mvp_dict.get('core_features', []))} features defined."
        ],
        "errors": [],
    }
