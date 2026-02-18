"""
MICRA — Framework Engine Node
================================

LEARNING CONCEPT: Multi-agent coordination

The framework engine doesn't do analysis itself — it manages a fleet
of specialist agents. This is the multi-agent pattern:

  Orchestrator (framework_engine_node)
    ├─→ TAMAgent.run()
    ├─→ PorterAgent.run()
    ├─→ SWOTAgent.run()
    └─→ KanoAgent.run()

Each agent is independent — it has its own retrieval queries, prompt,
and output schema. They share only the knowledge base (via Retriever).

WHY NOT RUN THEM IN PARALLEL?
In theory you could use Python's concurrent.futures or asyncio to run
all agents simultaneously (cutting execution time by 4x).

In practice, for a learning project:
  1. Sequential is easier to debug (clear order in logs)
  2. OpenAI rate limits make parallel calls risky without backoff
  3. Each agent's output doesn't depend on others (no blocking dependency)

In Phase 2 of the roadmap (production version), we'd parallelize with:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(agent.run, brief, retriever) for agent in agents]
        results = [f.result() for f in futures]

LEARNING CONCEPT: Dynamic agent selection

The framework engine doesn't always run all 4 frameworks.
It reads `research_plan["frameworks_to_apply"]` and only runs
the frameworks the planner selected.

This is important: blindly applying all frameworks wastes tokens.
For a "should we acquire this company?" question, BCG and Ansoff
are relevant but Kano is not. The planner makes that call.
"""

import logging
from rich.console import Console
from rich.table import Table

from src.state import MICRAState, FrameworkOutput
from src.retriever import Retriever
from src.nodes.frameworks.porter import PorterAgent
from src.nodes.frameworks.swot import SWOTAgent
from src.nodes.frameworks.tam import TAMAgent
from src.nodes.frameworks.kano import KanoAgent

logger = logging.getLogger(__name__)
console = Console()

# Registry: framework name → agent class
# LEARNING: A registry (dict) is the clean alternative to a long if/elif chain.
# To add a new framework in the future, just add one entry here.
FRAMEWORK_REGISTRY: dict[str, type] = {
    "porter_5_forces": PorterAgent,
    "swot": SWOTAgent,
    "tam_sam_som": TAMAgent,
    "kano": KanoAgent,
}


def framework_engine_node(state: MICRAState) -> dict:
    """
    Run each requested framework agent against the knowledge base.

    LEARNING: Notice how this node:
      1. Reads what frameworks to run from the plan (dynamic)
      2. Checks if the knowledge base is ready
      3. Runs each framework agent, catching failures gracefully
      4. Collects all outputs into the state's framework_outputs list

    Partial failure tolerance is critical here. If Porter's 5 Forces
    fails (e.g., LLM retry exhausted), we still want SWOT and TAM
    to run. The report will note the missing framework.
    """
    plan = state.get("research_plan", {})
    collection_name = state.get("vector_db_collection_name", "")
    brief = state.get("research_brief", "")

    if not collection_name:
        return {
            "errors": ["[framework_engine] Vector DB not ready. Embedder may have failed."],
            "messages": ["[framework_engine] Skipped — no knowledge base available."]
        }

    requested_frameworks = plan.get("frameworks_to_apply", list(FRAMEWORK_REGISTRY.keys()))

    # Filter to only frameworks we have agents for
    runnable = [f for f in requested_frameworks if f in FRAMEWORK_REGISTRY]
    unknown = [f for f in requested_frameworks if f not in FRAMEWORK_REGISTRY]

    console.print(f"\n[bold cyan]Phase 3: Strategic Framework Analysis[/bold cyan]")
    console.print(f"Running {len(runnable)} framework(s): {', '.join(runnable)}\n")

    # Initialize retriever once — reused by all agents
    # LEARNING: Creating one Retriever instance and passing it to all agents
    # is more efficient than each agent creating its own ChromaDB connection.
    retriever = Retriever(collection_name)
    console.print(f"[dim]Knowledge base: {retriever.chunk_count} chunks available[/dim]\n")

    outputs: list[FrameworkOutput] = []
    errors: list[str] = []
    messages: list[str] = []

    for framework_name in runnable:
        agent_class = FRAMEWORK_REGISTRY[framework_name]
        agent = agent_class()

        console.print(f"  [cyan]→[/cyan] Running [bold]{framework_name}[/bold]...", end=" ")

        try:
            output = agent.run(
                research_brief=brief,
                retriever=retriever,
            )
            outputs.append(output)
            console.print(f"[green]✓[/green] (confidence: {output['confidence']:.0%})")

        except Exception as e:
            error_msg = f"[framework_engine] {framework_name} failed after retries: {e}"
            errors.append(error_msg)
            console.print(f"[red]✗[/red] {e}")
            logger.error(error_msg, exc_info=True)

    if unknown:
        errors.append(f"[framework_engine] Unknown frameworks (not implemented): {unknown}")

    _display_results_summary(outputs)

    messages.append(
        f"[framework_engine] Ran {len(outputs)}/{len(runnable)} frameworks. "
        f"Avg confidence: {sum(o['confidence'] for o in outputs) / max(len(outputs), 1):.0%}"
    )

    return {
        "framework_outputs": outputs,
        "messages": messages,
        "errors": errors,
    }


def _display_results_summary(outputs: list[FrameworkOutput]) -> None:
    if not outputs:
        return

    table = Table(title="Framework Results", border_style="green", show_header=True)
    table.add_column("Framework", style="cyan")
    table.add_column("Confidence", style="white")
    table.add_column("Sources Used", style="dim")

    for output in outputs:
        table.add_row(
            output["framework_name"].replace("_", " ").title(),
            f"{output['confidence']:.0%}",
            str(len(output["source_chunks"])),
        )

    console.print(table)
