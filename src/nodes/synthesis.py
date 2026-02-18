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


class GTMSchema(BaseModel):
    primary_target_segments: list[str] = Field(
        description="Top 3-4 customer segments to target first, each described in 1-2 sentences "
                    "including segment size hint and why they're the priority"
    )
    go_to_market_by_region: list[dict] = Field(
        description="List of region dicts. Each dict must have keys: region (str), "
                    "market_characteristics (list[str] of 3-4 traits), "
                    "go_to_market_approach (str), "
                    "key_channels (list[str]), "
                    "sales_cycle_months (str), "
                    "typical_deal_size (str)"
    )
    sales_strategy: str = Field(
        description="2-3 sentences on the overall sales motion: direct vs. channel, "
                    "inside sales vs. field, POC/trial approach"
    )
    marketing_channels: list[str] = Field(
        description="5-7 specific marketing channels with brief rationale each. "
                    "e.g. 'LinkedIn targeting SCADA engineers — primary awareness channel'"
    )
    competitive_messaging: list[dict] = Field(
        description="List of dicts, one per major competitor. Each dict: "
                    "competitor (str), messages (list[str] of 2-3 short positioning statements)"
    )
    pricing_options: list[dict] = Field(
        description="2-3 pricing model options. Each dict: model_name (str), "
                    "tiers (list[dict] with name/price/includes), "
                    "recommended (bool), rationale (str)"
    )
    revenue_year1: str = Field(description="Year 1 ARR estimate with assumptions")
    revenue_year3: str = Field(description="Year 3 ARR estimate with assumptions")
    revenue_year5: str = Field(description="Year 5 ARR estimate with assumptions")


class TeamSchema(BaseModel):
    core_team_roles: list[dict] = Field(
        description="Core team needed for MVP. Each dict: role (str), count (int), "
                    "key_skills (list[str]), focus (str one sentence)"
    )
    total_core_headcount: str = Field(
        description="Total headcount for MVP phase, e.g. '12-14 people'"
    )
    critical_skills: list[str] = Field(
        description="5-7 must-have technical or domain skills hardest to hire for"
    )
    extended_team_post_mvp: list[str] = Field(
        description="5-7 roles to add after MVP launch"
    )
    hiring_priorities: list[str] = Field(
        description="Top 3 hires to make in the first 30 days and why"
    )


class RoadmapSchema(BaseModel):
    mvp_phase: dict = Field(
        description="MVP phase dict with keys: timeline (str e.g. 'Months 1-6'), "
                    "vision (str 1 sentence product vision), "
                    "focus_areas (list[dict] with name/effort_percent/key_deliverables)"
    )
    phase_2: dict = Field(
        description="Phase 2 dict with keys: timeline, theme (str), key_deliverables (list[str])"
    )
    phase_3: dict = Field(
        description="Phase 3 dict with same structure as phase_2"
    )
    phase_4: dict = Field(
        description="Phase 4 dict with same structure as phase_2"
    )
    go_no_go_month3: list[str] = Field(
        description="3-5 measurable go/no-go criteria to evaluate at Month 3 checkpoint"
    )
    go_no_go_month6: list[str] = Field(
        description="3-5 measurable go/no-go criteria to evaluate at Month 6 (MVP) checkpoint"
    )
    week1_2_actions: list[str] = Field(
        description="5-7 concrete immediate actions for weeks 1-2"
    )
    week3_4_actions: list[str] = Field(
        description="5-7 concrete actions for weeks 3-4"
    )
    month1_milestones: list[str] = Field(
        description="4-5 milestones to hit by end of Month 1"
    )
    month3_milestones: list[str] = Field(
        description="4-5 milestones to hit by end of Month 3"
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

    # ── Call 3: GTM strategy + pricing + revenue projections ──────────────
    console.print("  [cyan]→[/cyan] Generating GTM strategy & pricing...", end=" ")
    gtm_dict: dict = {}

    try:
        gtm_llm = llm.with_structured_output(GTMSchema)
        gtm_result: GTMSchema = gtm_llm.invoke([
            SystemMessage(content="""You are a go-to-market strategist.
Your task: define the GTM strategy, pricing models, and revenue projections
for the product described in the research brief.

Base your recommendations on:
- Market sizing (TAM/SAM/SOM) from framework analysis
- Competitor pricing intelligence from competitor profiles
- Geographic market characteristics from the research brief
- Customer segments identified in the analysis

Be specific and grounded in the evidence. Name real pricing tiers with dollar figures.
Revenue projections should state their assumptions clearly."""),
            HumanMessage(content=f"""Research Brief:
{brief}

Framework Analysis:
{framework_summary}

Competitor Profiles:
{competitor_summary}

Build/Buy/Partner Decision:
{json.dumps(bbp_dict, indent=2)}

MVP Recommendation:
{json.dumps(mvp_dict, indent=2)}

Define the go-to-market strategy, pricing models, and revenue projections.""")
        ])
        console.print("[green]✓[/green]")
        gtm_dict = gtm_result.model_dump()
    except Exception as e:
        logger.error(f"GTM synthesis failed: {e}")
        console.print(f"[red]✗[/red]")

    # ── Call 4: Team requirements ──────────────────────────────────────────
    console.print("  [cyan]→[/cyan] Generating team requirements...", end=" ")
    team_dict: dict = {}

    try:
        team_llm = llm.with_structured_output(TeamSchema)
        team_result: TeamSchema = team_llm.invoke([
            SystemMessage(content="""You are a technology talent strategist.
Your task: define the team required to build and launch the MVP described.

Consider:
- The technical complexity of the product (from framework analysis)
- The competitive landscape (what level of engineering is needed to compete)
- The MVP scope and timeline
- The market geographies (local support needs)

Be specific about roles. Avoid generic 'software engineer' — specify
e.g. 'Time-Series Database Engineer (C++/Rust)' or 'OPC Protocol Engineer'."""),
            HumanMessage(content=f"""Research Brief:
{brief}

MVP Recommendation:
{json.dumps(mvp_dict, indent=2)}

GTM Strategy:
{json.dumps(gtm_dict, indent=2)}

Competitor Landscape:
{competitor_summary}

Define the team required to build and launch this MVP.""")
        ])
        console.print("[green]✓[/green]")
        team_dict = team_result.model_dump()
    except Exception as e:
        logger.error(f"Team synthesis failed: {e}")
        console.print(f"[red]✗[/red]")

    # ── Call 5: Phased roadmap + next steps ────────────────────────────────
    console.print("  [cyan]→[/cyan] Generating phased roadmap & next steps...", end=" ")
    roadmap_dict: dict = {}

    try:
        roadmap_llm = llm.with_structured_output(RoadmapSchema)
        roadmap_result: RoadmapSchema = roadmap_llm.invoke([
            SystemMessage(content="""You are a product roadmap strategist.
Your task: create a phased product roadmap (MVP through Phase 4) and
concrete next steps for the first 30-90 days.

The MVP phase should cover 6 months. Each subsequent phase is 6 months.
Focus areas in MVP should reflect the Kano must-haves and competitive gaps.
Next steps should be concrete and actionable — specific enough to assign
to a named person with a deadline."""),
            HumanMessage(content=f"""Research Brief:
{brief}

MVP Recommendation:
{json.dumps(mvp_dict, indent=2)}

Team Requirements:
{json.dumps(team_dict, indent=2)}

Build/Buy/Partner Decision:
{json.dumps(bbp_dict, indent=2)}

Framework Analysis Summary:
{framework_summary[:3000]}

Create a phased roadmap and 30-90 day action plan.""")
        ])
        console.print("[green]✓[/green]")
        roadmap_dict = roadmap_result.model_dump()
    except Exception as e:
        logger.error(f"Roadmap synthesis failed: {e}")
        console.print(f"[red]✗[/red]")

    # ── Display key outputs ────────────────────────────────────────────────
    if "recommendation" in bbp_dict:
        console.print(f"\n  [bold]Recommendation:[/bold] [green]{bbp_dict['recommendation']}[/green]")
    if "north_star_metric" in mvp_dict:
        console.print(f"  [bold]North Star:[/bold] {mvp_dict.get('north_star_metric', 'TBD')}")
    if "core_features" in mvp_dict:
        console.print(f"  [bold]MVP features:[/bold] {len(mvp_dict.get('core_features', []))} identified")
    if gtm_dict.get("revenue_year1"):
        console.print(f"  [bold]Year 1 Revenue:[/bold] {gtm_dict['revenue_year1']}")

    return {
        "build_buy_partner_decision": bbp_dict,
        "mvp_recommendation": mvp_dict,
        "gtm_strategy": gtm_dict,
        "team_requirements": team_dict,
        "phased_roadmap": roadmap_dict,
        "messages": [
            f"[synthesis] Build/Buy/Partner: {bbp_dict.get('recommendation', 'error')}. "
            f"MVP: {len(mvp_dict.get('core_features', []))} features. "
            f"GTM: {len(gtm_dict.get('go_to_market_by_region', []))} regions. "
            f"Roadmap: {bool(roadmap_dict)} generated."
        ],
        "errors": [],
    }
