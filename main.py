"""
MICRA — CLI Entry Point
========================

Run with:
    python main.py

Or with a query directly:
    python main.py "Analyze the DERMS market in USA. Should L&T build or partner?"

LEARNING: stream_mode="values"

The default stream mode ("updates") yields {node_name: partial_update} —
useful for seeing exactly what each node wrote, but you don't get the
full accumulated state.

stream_mode="values" yields the FULL state after every node completes.
The last yielded value is the final state — no second .invoke() needed.

This is the correct pattern for progress tracking without running twice.
"""

import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

load_dotenv()
console = Console()

NODE_LABELS = {
    "clarification":     "Phase 1 — Clarification",
    "planner":           "Phase 1 — Research Planning",
    "scraper":           "Phase 2 — Data Ingestion",
    "embedder":          "Phase 2 — Building Knowledge Base",
    "framework_engine":  "Phase 3 — Framework Analysis",
    "competitive_intel": "Phase 3 — Competitive Intelligence",
    "synthesis":         "Phase 3 — Strategic Synthesis",
    "report_generator":  "Phase 4 — Generating Report",
    "evaluator":         "Phase 4 — AI Evaluation",
}


def validate_env():
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set.")
        console.print("Add your key to .env:  OPENAI_API_KEY=sk-...")
        sys.exit(1)


def run_once():
    """Run one full research pipeline. Returns True to run again, False to exit."""
    from src.graph import build_graph, create_initial_state

    console.print("\n[bold cyan]MICRA — Market Intelligence & Competitive Research Agent[/bold cyan]")
    console.print("[dim]What market or product question do you want to research?[/dim]\n")
    query = Prompt.ask("[bold]Research query[/bold]")

    if not query.strip():
        console.print("[red]No query provided.[/red]")
        return False

    graph = build_graph()
    initial_state = create_initial_state(query)

    console.print(Rule("[bold cyan]MICRA Pipeline Starting[/bold cyan]"))

    final_state = None
    completed_nodes = []

    # stream_mode="values" → yields the full accumulated state after each node.
    # We capture each snapshot; the last one is the complete final state.
    # This runs the graph exactly once (no duplicate invoke call).
    for state_snapshot in graph.stream(initial_state, stream_mode="values"):
        final_state = state_snapshot

        # Figure out which node just completed by comparing message counts
        messages = state_snapshot.get("messages", [])
        if messages:
            last_msg = messages[-1]
            node = next((n for n in NODE_LABELS if f"[{n}]" in last_msg), None)
            if node and node not in completed_nodes:
                completed_nodes.append(node)
                console.print(f"  [dim]✓ {NODE_LABELS[node]}[/dim]")

    if not final_state:
        console.print("[red]Pipeline produced no output.[/red]")
        return _ask_again()

    # ── Summary ────────────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Complete[/bold cyan]"))

    report_path = final_state.get("report_path", "")
    if report_path:
        console.print(f"\n[bold green]Report:[/bold green] {report_path}")

    evaluation = final_state.get("evaluation", {})
    if evaluation:
        score = evaluation.get("overall_score", 0)
        passed = evaluation.get("passed", False)
        status = "[green]PASS[/green]" if passed else "[red]FAIL — review flagged claims[/red]"
        console.print(f"[bold]Quality Score:[/bold] {score:.0%} — {status}")
        flagged = evaluation.get("flagged_claims", [])
        if flagged:
            console.print(f"[yellow]{len(flagged)} claim(s) to verify before sharing[/yellow]")

    errors = [e for e in final_state.get("errors", []) if e]
    if errors:
        console.print(f"\n[yellow]{len(errors)} warning(s) during run[/yellow]")

    return _ask_again()


def _ask_again() -> bool:
    console.print()
    again = Prompt.ask("[dim]Research another market?[/dim]", choices=["y", "n"], default="n")
    return again == "y"


def main():
    validate_env()

    # Support passing a query as a CLI arg (non-interactive mode)
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        from src.graph import build_graph, create_initial_state
        graph = build_graph()
        final_state = None
        for state_snapshot in graph.stream(create_initial_state(query), stream_mode="values"):
            final_state = state_snapshot
        if final_state:
            console.print(f"\nReport: {final_state.get('report_path', 'N/A')}")
            eval_ = final_state.get("evaluation", {})
            if eval_:
                console.print(f"Score:  {eval_.get('overall_score', 0):.0%}")
        return

    # Interactive loop: ask what to analyze, run, offer to run again
    while run_once():
        pass

    console.print("\n[dim]Goodbye.[/dim]\n")


if __name__ == "__main__":
    main()
