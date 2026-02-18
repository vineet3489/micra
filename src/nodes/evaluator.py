"""
MICRA — AI Evaluator Node
===========================

LEARNING CONCEPT: LLM-as-Judge

One of the most important patterns in production AI systems.

The idea: use an LLM to evaluate another LLM's output.

WHY THIS WORKS:
  Evaluating quality is easier than producing quality.
  An LLM can spot "this claim about market size has no cited source"
  even if it couldn't have produced the correct market size itself.

  This is analogous to how a reviewer can catch errors in a paper
  they couldn't have written themselves.

THE PATTERN:

  Evaluator LLM receives:
    1. The original question / brief (what was asked)
    2. The source evidence (what the research found)
    3. The output to evaluate (what was produced)
    4. A detailed rubric (what "good" looks like)

  Evaluator LLM returns:
    - Scores per dimension (0-1)
    - Specific flagged claims ("this number has no source")
    - Overall pass/fail

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE-BASED vs. REFERENCE-FREE EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference-based: compare output against a "gold standard" answer.
  Used in: translation quality, QA benchmarks.
  Problem: we don't have gold standard market intelligence reports.

Reference-free: evaluate quality without a reference answer.
  Used in: chatbot quality, summarization, our use case.
  Judge criteria: Does it answer the question? Are claims supported?

We use reference-free + source-grounding evaluation:
  "Is every factual claim in the output supported by the scraped sources?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION DIMENSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Factual Grounding (0-1):
   Are specific claims (numbers, company names, market sizes) supported
   by the retrieved source chunks? Check each factual claim.
   0.9-1.0 = all major claims verifiable
   0.7-0.9 = minor extrapolations from sources
   Below 0.7 = significant unsupported claims

2. Framework Correctness (0-1):
   Were frameworks applied per their definitions?
   Porter = industry structure (not company strengths)
   SWOT = company-relative (not generic)
   TAM ≠ SAM ≠ SOM (did it distinguish them?)
   Kano = feature classification by user response (not arbitrary)

3. Completeness (0-1):
   Did the output answer all sub-questions in the research brief?
   Did every requested framework produce output?
   Is there a clear Build/Buy/Partner recommendation?
   Is there a concrete MVP feature set?

4. Strategic Coherence (0-1):
   Do the recommendations follow logically from the analysis?
   Does MVP align with Kano must-haves?
   Does Build/Buy/Partner follow from Porter + SWOT?
   Is the North Star metric connected to the core value proposition?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING: WHY A SEPARATE LLM CALL?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We use a SEPARATE LLM instance (not the same one that produced the output).
Why? The same LLM that generated an answer is biased toward validating it.
It "remembers" (in the context window) that it just produced that output
and is more likely to agree with itself.

A fresh LLM call, given the output as external input, is more objective.

In production, you'd use a different model family (e.g., evaluate
GPT-4o output with Claude, or vice versa) for even less bias.
"""

import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from src.state import MICRAState, EvaluationResult
from src.config import LLM_MODEL, EVAL_PASS_THRESHOLD

logger = logging.getLogger(__name__)
console = Console()


# ── Evaluation schema ──────────────────────────────────────────────────────
class EvaluationSchema(BaseModel):
    """
    LEARNING: The evaluation rubric as a Pydantic schema.

    Each field's description IS the rubric for that dimension.
    The evaluator LLM reads the description and uses it to score.

    Writing a good evaluation schema is itself a skill:
      - Descriptions must be unambiguous
      - Score ranges must be explicitly defined
      - What constitutes a "hallucination" must be crystal clear
    """
    factual_grounding: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Score 0.0-1.0: Are factual claims (numbers, company names, market sizes, "
            "product features) supported by the source chunks? "
            "1.0 = every claim has a source. 0.7 = most claims sourced, minor gaps. "
            "0.5 = some important claims unsupported. Below 0.5 = major hallucinations."
        )
    )
    framework_correctness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Score 0.0-1.0: Were frameworks applied correctly per their definitions? "
            "Porter should assess industry structure forces. SWOT should be company-specific. "
            "TAM/SAM/SOM should be clearly distinguished. Kano should classify by user response. "
            "1.0 = all frameworks applied correctly. 0.5 = partial misapplication."
        )
    )
    completeness: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Score 0.0-1.0: How completely were the research questions answered? "
            "Check: Is there market sizing? Competitive analysis? MVP recommendation? "
            "Build/Buy/Partner decision? Go-to-market strategy? "
            "1.0 = all sections complete. 0.7 = minor gaps. 0.5 = major sections missing."
        )
    )
    strategic_coherence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Score 0.0-1.0: Do recommendations follow logically from the analysis? "
            "Does the MVP align with Kano must-haves? Does Build/Buy/Partner follow from "
            "competitive analysis? Does the North Star reflect the core value proposition? "
            "1.0 = recommendations are tightly grounded in analysis. "
            "0.5 = recommendations seem disconnected from evidence."
        )
    )
    hallucinations_detected: int = Field(
        ge=0,
        description=(
            "Count of specific claims in the output that appear to be invented "
            "(not present in source chunks, not reasonable inferences). "
            "Examples: specific revenue figures not in sources, named companies not in context, "
            "market size numbers that contradict the sources."
        )
    )
    flagged_claims: list[str] = Field(
        description=(
            "Max 3 unsupported claims. Format each as: 'Claim: X — Issue: Y'. "
            "Empty list if none found. Keep each entry under 20 words."
        )
    )
    output_strengths: list[str] = Field(
        description="Exactly 2 strengths, max 15 words each."
    )
    output_improvements: list[str] = Field(
        description="Exactly 2 improvements, max 15 words each."
    )
    overall_recommendation: str = Field(
        description=(
            "One of exactly: 'PASS', 'CONDITIONAL PASS', 'FAIL — add sources', "
            "'FAIL — revise frameworks'. Add one short sentence reason."
        )
    )


def _build_evaluation_context(state: MICRAState) -> str:
    """
    Prepare a focused context package for the evaluator.

    LEARNING: Context management for evaluation

    We can't pass the full state to the evaluator — that would be
    100,000+ tokens. We extract the most evaluation-relevant pieces:
      - The research brief (what was asked)
      - Framework outputs (what analysis was done)
      - Key claims made (representative sample from outputs)
      - Source chunk samples (what evidence was available)

    This is a form of "context compression" — the evaluator gets
    enough information to judge quality without drowning in raw data.
    """
    parts = []

    # Research brief
    brief = state.get("research_brief", "")
    parts.append(f"=== RESEARCH BRIEF ===\n{brief}")

    # Framework outputs (summarized)
    framework_outputs = state.get("framework_outputs", [])
    if framework_outputs:
        parts.append("=== FRAMEWORK ANALYSIS OUTPUTS ===")
        for output in framework_outputs:
            name = output["framework_name"].upper()
            conf = output["confidence"]
            data = json.dumps(output["output"], indent=2)[:2000]  # cap length
            parts.append(
                f"[{name}] (confidence: {conf:.0%})\n{data}"
            )

    # Competitor profiles
    profiles = state.get("competitor_profiles", [])
    if profiles:
        parts.append("=== COMPETITOR PROFILES ===")
        for p in profiles[:3]:  # cap at 3 for context length
            parts.append(
                f"{p['name']}: {p['product_summary']}\n"
                f"Strengths: {', '.join(p.get('strengths', [])[:2])}\n"
                f"Gaps: {', '.join(p.get('differentiation_gaps', [])[:2])}"
            )

    # Build/Buy/Partner + MVP
    bbp = state.get("build_buy_partner_decision", {})
    mvp = state.get("mvp_recommendation", {})
    if bbp:
        parts.append(f"=== BUILD/BUY/PARTNER DECISION ===\n{json.dumps(bbp, indent=2)[:1500]}")
    if mvp:
        parts.append(f"=== MVP RECOMMENDATION ===\n{json.dumps(mvp, indent=2)[:1500]}")

    # Sample source evidence (for hallucination checking)
    sources = state.get("sources", [])
    if sources:
        parts.append("=== SAMPLE SOURCE EVIDENCE (for hallucination verification) ===")
        for source in sources[:5]:
            snippet = source.get("raw_text", "")[:500]
            parts.append(f"Source: {source['url']}\n{snippet}")

    return "\n\n".join(parts)


def evaluator_node(state: MICRAState) -> dict:
    """
    Evaluate the quality of the full pipeline output.

    LEARNING: The evaluator is the last node. It:
      1. Checks output quality before the report reaches stakeholders
      2. Surfaces specific problems (hallucinations, missing sections)
      3. Provides a pass/fail gate — low-quality reports are flagged

    The result is stored in state["evaluation"] which:
      - Gets printed in main.py after the run
      - Could trigger a re-run (conditional edge) in a future version
      - Gets embedded in the report as a quality disclosure
    """
    console.print("\n[bold cyan]Phase 4b: AI Evaluation[/bold cyan]")

    framework_outputs = state.get("framework_outputs", [])
    if not framework_outputs and not state.get("build_buy_partner_decision"):
        console.print("[yellow]⚠ Minimal output to evaluate — most nodes may have failed[/yellow]")

    console.print("  [cyan]→[/cyan] Running LLM-as-judge evaluation...", end=" ")

    context = _build_evaluation_context(state)

    try:
        # LEARNING: Fresh LLM instance — no shared context with analysis agents.
        # max_tokens caps the response length. Without it the structured output
        # schema generates very verbose JSON and hits the model's output limit.
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_tokens=1500)
        eval_llm = llm.with_structured_output(EvaluationSchema)

        result: EvaluationSchema = eval_llm.invoke([
            SystemMessage(content="""You are an expert quality reviewer for strategic market intelligence reports.
Your job: objectively evaluate the quality of the analysis output provided.

Be critical and specific. Your evaluation will be used to:
  1. Decide whether the report is ready to share with executives
  2. Identify specific claims to verify before sharing
  3. Guide improvements in future runs

Scoring rubric is defined in each field description. Apply it precisely.
Do not be lenient — a score of 0.85+ means the output is genuinely high quality,
not just technically present."""),
            HumanMessage(content=f"""Evaluate the following market intelligence analysis output.

{context}

Apply the scoring rubric for each dimension. Be specific about any hallucinations
or unsupported claims you find. Provide actionable improvements.""")
        ])

        console.print("[green]✓[/green]")

        # Calculate weighted overall score
        # Factual grounding and coherence weighted highest
        overall = (
            result.factual_grounding * 0.30 +
            result.framework_correctness * 0.20 +
            result.completeness * 0.25 +
            result.strategic_coherence * 0.25
        )
        overall = round(overall, 3)
        passed = overall >= EVAL_PASS_THRESHOLD

        eval_result = EvaluationResult(
            overall_score=overall,
            factual_grounding=result.factual_grounding,
            framework_correctness=result.framework_correctness,
            completeness=result.completeness,
            strategic_coherence=result.strategic_coherence,
            hallucinations_detected=result.hallucinations_detected,
            flagged_claims=result.flagged_claims,
            passed=passed,
            recommendation=result.overall_recommendation,
        )

        _display_evaluation(eval_result, result)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        eval_result = EvaluationResult(
            overall_score=0.0,
            factual_grounding=0.0,
            framework_correctness=0.0,
            completeness=0.0,
            strategic_coherence=0.0,
            hallucinations_detected=0,
            flagged_claims=[f"Evaluation failed: {e}"],
            passed=False,
            recommendation="FAIL — evaluation could not run",
        )

    return {
        "evaluation": eval_result,
        "messages": [
            f"[evaluator] Score: {eval_result['overall_score']:.0%} "
            f"({'PASS' if eval_result['passed'] else 'FAIL'}). "
            f"Hallucinations: {eval_result['hallucinations_detected']}."
        ],
    }


def _display_evaluation(result: EvaluationResult, schema: EvaluationSchema) -> None:
    """Display evaluation results in a formatted table."""

    table = Table(
        title=f"AI Evaluation — {'[green]PASS[/green]' if result['passed'] else '[red]FAIL[/red]'}",
        border_style="green" if result["passed"] else "red",
        show_header=True,
    )
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="white")
    table.add_column("Weight", style="dim")

    dimensions = [
        ("Factual Grounding", result["factual_grounding"], "30%"),
        ("Framework Correctness", result["framework_correctness"], "20%"),
        ("Completeness", result["completeness"], "25%"),
        ("Strategic Coherence", result["strategic_coherence"], "25%"),
    ]

    for name, score, weight in dimensions:
        color = "green" if score >= 0.75 else "yellow" if score >= 0.5 else "red"
        table.add_row(name, f"[{color}]{score:.0%}[/{color}]", weight)

    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{result['overall_score']:.0%}[/bold]",
        "—"
    )
    console.print(table)

    if result["hallucinations_detected"] > 0:
        console.print(f"\n[red]⚠ {result['hallucinations_detected']} potential hallucination(s) detected:[/red]")
        for claim in result["flagged_claims"][:3]:
            console.print(f"  [dim]• {claim}[/dim]")

    console.print(f"\n[dim]{result['recommendation']}[/dim]")

    if hasattr(schema, "output_strengths"):
        console.print("\n[bold]Strengths:[/bold]")
        for s in schema.output_strengths:
            console.print(f"  [green]✓[/green] {s}")

    if hasattr(schema, "output_improvements"):
        console.print("\n[bold]To improve:[/bold]")
        for s in schema.output_improvements:
            console.print(f"  [yellow]→[/yellow] {s}")
