"""
MICRA — Graph Assembly
========================

LEARNING CONCEPT: How LangGraph works

LangGraph builds on top of LangChain to let you define agents as
directed graphs. Here's what you need to understand:

1. NODES  — functions that do work and update state
2. EDGES  — connections between nodes (what runs after what)
3. STATE  — the shared dict that flows through every node
4. START  — the entry point of the graph
5. END    — the terminal node (graph stops here)

FULL GRAPH (all phases):

    START
      │
      ▼
  clarification_node          Phase 1: Ask user questions
      │
      ▼
  planner_node                Phase 1: Plan the research
      │
      ▼
  scraper_node                Phase 2: Ingest web + papers (TODO)
      │
      ▼
  embedder_node               Phase 2: Chunk + embed + store (TODO)
      │
      ▼
  framework_engine_node       Phase 4: Run all framework agents (TODO)
      │
      ▼
  competitive_intel_node      Phase 4: Profile competitors (TODO)
      │
      ▼
  synthesis_node              Phase 4: Build/Buy/Partner + MVP (TODO)
      │
      ▼
  report_generator_node       Phase 5: Generate .docx (TODO)
      │
      ▼
  evaluator_node              Phase 6: AI eval / LLM-as-judge (TODO)
      │
      ▼
    END

LEARNING CONCEPT: Conditional edges

In Phase 6, the evaluator might decide to re-run research if the score
is too low. This uses a "conditional edge" — a function that looks at
state and returns which node to go to next.

We haven't built that yet, but the architecture supports it:

    def route_after_evaluation(state: MICRAState) -> str:
        if state["evaluation"]["overall_score"] < 0.75:
            return "scraper_node"  # re-run research
        return END

    graph.add_conditional_edges("evaluator_node", route_after_evaluation)

This is how agents can loop, retry, and self-correct.

LEARNING CONCEPT: Compile vs. Run

LangGraph works in two steps:
1. DEFINE  — add nodes and edges to a StateGraph (no execution)
2. COMPILE — call .compile() to get a runnable CompiledGraph
3. RUN     — call .invoke() or .stream() on the compiled graph

You can compile once and invoke many times.
"""

from typing import Any
from langgraph.graph import StateGraph, START, END
from rich.console import Console

from src.state import MICRAState
from src.nodes.clarification import clarification_node
from src.nodes.planner import planner_node
from src.nodes.scraper import scraper_node
from src.nodes.embedder import embedder_node
from src.nodes.framework_engine import framework_engine_node
from src.nodes.competitive_intel import competitive_intel_node
from src.nodes.synthesis import synthesis_node
from src.nodes.report_generator import report_generator_node
from src.nodes.evaluator import evaluator_node

console = Console()


def _placeholder_node(name: str):
    """
    Factory that creates placeholder nodes for phases not yet built.

    LEARNING: This is a useful pattern during development.
    Instead of leaving gaps in the graph, we use placeholder nodes
    that log their position and pass state through unchanged.
    This lets you run the full graph end-to-end even before all
    phases are implemented.
    """
    def node(state: MICRAState) -> dict:
        console.print(f"[dim yellow]⏳ [{name}] — not yet implemented (coming in a future phase)[/dim yellow]")
        return {"messages": [f"[{name}] placeholder — not yet implemented"]}
    node.__name__ = name
    return node


def build_graph() -> Any:
    """
    Build and compile the full MICRA graph.

    LEARNING: StateGraph takes your state type as the argument.
    LangGraph uses this to know what fields are valid in state.

    Returns a CompiledGraph — the object you call .invoke() on.
    """

    # LEARNING: Create the graph builder.
    # StateGraph(MICRAState) tells LangGraph:
    # "Every node will receive and return dicts shaped like MICRAState"
    graph = StateGraph(MICRAState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    # LEARNING: graph.add_node(name, function)
    # The name is just a label used in edges and logs.
    # The function IS the node — it receives state, returns partial update.

    # Phase 1 (implemented)
    graph.add_node("clarification", clarification_node)
    graph.add_node("planner", planner_node)

    # Phase 2 (implemented)
    graph.add_node("scraper", scraper_node)
    graph.add_node("embedder", embedder_node)

    # Phase 3 (implemented)
    graph.add_node("framework_engine", framework_engine_node)
    graph.add_node("competitive_intel", competitive_intel_node)
    graph.add_node("synthesis", synthesis_node)

    # Phase 4 (implemented)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("evaluator", evaluator_node)

    # ── Add edges ──────────────────────────────────────────────────────────
    # LEARNING: Edges define the execution order.
    # add_edge(from, to) means "after `from` completes, run `to`"
    # START is a special constant — it's the entry point.
    # END is a special constant — the graph stops here.

    graph.add_edge(START, "clarification")
    graph.add_edge("clarification", "planner")
    graph.add_edge("planner", "scraper")
    graph.add_edge("scraper", "embedder")
    graph.add_edge("embedder", "framework_engine")
    graph.add_edge("framework_engine", "competitive_intel")
    graph.add_edge("competitive_intel", "synthesis")
    graph.add_edge("synthesis", "report_generator")
    graph.add_edge("report_generator", "evaluator")
    graph.add_edge("evaluator", END)

    # ── Compile ────────────────────────────────────────────────────────────
    # LEARNING: .compile() validates the graph structure and returns
    # a runnable object. It will error here (not at runtime) if:
    #   - A node is referenced in an edge but never added
    #   - There's no path from START to END
    #   - State fields don't match the TypedDict definition
    compiled = graph.compile()

    return compiled


# ── Initial state helper ───────────────────────────────────────────────────
def create_initial_state(query: str) -> MICRAState:
    """
    LEARNING: LangGraph needs the state to be fully initialized.
    Optional fields that haven't been set yet should be given sensible
    defaults so nodes don't crash trying to read undefined keys.

    We set empty lists / empty strings / False for everything.
    As the graph runs, nodes will fill in real values.
    """
    return MICRAState(
        # Input
        raw_query=query,

        # Phase 1
        clarification_answers=[],
        research_brief="",
        research_plan={},

        # Phase 2
        sources=[],
        vector_db_ready=False,
        vector_db_collection_name="",

        # Phase 3
        framework_outputs=[],
        competitor_profiles=[],

        # Phase 4
        build_buy_partner_decision={},
        mvp_recommendation={},
        gtm_strategy={},
        team_requirements={},
        phased_roadmap={},

        # Phase 5
        report_path="",

        # Phase 6
        evaluation={},

        # Always present
        messages=[],
        errors=[],
    )
