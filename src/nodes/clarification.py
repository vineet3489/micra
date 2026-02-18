"""
MICRA — Clarification Node
============================

LEARNING CONCEPT: What is a LangGraph "node"?

A node is just a Python function (or async function) that:
  1. Receives the current graph state
  2. Does some work
  3. Returns a PARTIAL state update (only the fields it changed)

That's it. No magic. The function signature is always:
    def my_node(state: MICRAState) -> dict:

LangGraph calls your function, takes the dict you return,
and merges it back into the state.

LEARNING CONCEPT: Human-in-the-loop

This node demonstrates "human-in-the-loop" — the agent PAUSES
to collect user input before continuing. There are two approaches:

Approach A (what we use here): The node itself asks questions
  via CLI input(). Simple, works for terminal-based tools.

Approach B (advanced, Phase 5): LangGraph's interrupt() mechanism
  — the graph truly pauses, saves state to disk, waits for an
  external event (API call, UI submission), then resumes.
  This is how you'd build a web UI version.

For now, Approach A is sufficient and teaches the concept cleanly.

LEARNING CONCEPT: LLM call inside a node

This node uses the LLM to GENERATE the right questions to ask.
This is better than hardcoded questions because:
  - Questions adapt to the domain (DERMS vs. HR software vs. fintech)
  - Questions adapt to what's already answered in raw_query
  - The agent is genuinely reasoning, not just filling a template
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.state import MICRAState, ClarificationAnswer

console = Console()


def _get_llm():
    """
    LEARNING: Lazy LLM instantiation.

    Never instantiate LLM clients at module level. If you do, importing
    the module crashes when no API key is set (e.g. during tests, CI, or
    when the .env hasn't been loaded yet).

    Instead, create the LLM inside the function that needs it. Python
    caches module imports, but function-level variables are cheap to create.

    Temperature note:
      0.0 = deterministic (same input → same output, good for structured tasks)
      0.7+ = creative (good for writing prose summaries)
    """
    return ChatOpenAI(model="gpt-4o", temperature=0)


def generate_clarification_questions(raw_query: str) -> list[str]:
    """
    Use the LLM to generate targeted clarifying questions for this query.

    LEARNING: This is a simple single-turn LLM call — no tools, no RAG,
    just prompt → response. The most basic form of LLM use.

    We use SystemMessage + HumanMessage — the standard chat format:
      - SystemMessage: sets the role/persona and instructions
      - HumanMessage: the actual input from the user
    """

    system_prompt = """You are a strategic research analyst.
Your job is to generate 5-7 targeted clarifying questions before beginning
market intelligence research. These questions help scope the research correctly.

Good questions ask about:
- Geography / market boundary
- The company's current capabilities in this space
- Time horizon (MVP in 6 months vs. 3-year strategy)
- Whether to evaluate build / buy / partner — or just one option
- Budget or resource constraints
- Specific competitors already known
- The primary decision-maker and what they need to decide

Return ONLY a JSON array of question strings. No explanation. No numbering.
Example format:
["Question 1?", "Question 2?", "Question 3?"]"""

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research query: {raw_query}")
    ])

    # LEARNING: response.content is always a string.
    # When you ask the LLM to return JSON, you must parse it yourself.
    # In Phase 4 we'll use JSON mode / structured outputs to avoid this.
    # For now, we parse manually with a try/except fallback.
    try:
        questions = json.loads(response.content)
        return questions
    except json.JSONDecodeError:
        # Fallback: return generic questions if LLM gave malformed JSON
        return [
            "What geography or market region are you targeting?",
            "What is your company's current capability in this domain?",
            "What is your time horizon — MVP in months, or multi-year strategy?",
            "Are you evaluating build, buy, or partner — or all three options?",
            "Who is the primary audience for this report and what decision will they make?",
        ]


def build_research_brief(
    raw_query: str,
    answers: list[ClarificationAnswer]
) -> str:
    """
    After collecting answers, synthesize a structured research brief.

    LEARNING: This shows how to pass structured context to an LLM.
    We format the Q&A pairs into a readable block and ask the LLM
    to synthesize them into a brief that all downstream nodes will use.

    The research brief is the "source of truth" for the entire pipeline.
    Every framework agent, every scraper, every synthesis node reads it.
    """

    qa_formatted = "\n".join([
        f"Q: {a['question']}\nA: {a['answer']}"
        for a in answers
    ])

    system_prompt = """You are a strategic research analyst.
Given an original research query and clarifying Q&A, write a structured
research brief in 150-200 words. The brief must clearly state:

1. The core research question
2. The market/geography scope
3. The company context (who is asking, their capabilities)
4. The decision to be made (build/buy/partner, enter/avoid, etc.)
5. The time horizon
6. Key constraints or priorities

Write in clear, professional prose. This brief will be used to guide
autonomous research agents — be precise and unambiguous."""

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Original query: {raw_query}

Clarifying answers:
{qa_formatted}
""")
    ])

    return response.content


# ── The Node ───────────────────────────────────────────────────────────────
def clarification_node(state: MICRAState) -> dict:
    """
    LEARNING: This is the node function LangGraph calls.

    Notice:
    1. It receives the full state
    2. It returns only the fields IT changed (partial update)
    3. It appends to `messages` — the log — but doesn't overwrite it

    The node does three things:
    A. Generates clarifying questions using the LLM
    B. Asks them interactively via CLI  (skipped if answers already in state)
    C. Synthesizes answers into a research brief

    The "skip if already answered" path is used by the Streamlit UI,
    which collects answers in a form BEFORE invoking the graph.
    This keeps the node reusable across CLI and web interfaces.
    """

    # ── Skip path: answers already provided (Streamlit / API) ──────────────
    if state.get("clarification_answers"):
        console.print("[dim]Clarification answers pre-loaded — skipping interactive prompt[/dim]")
        brief = build_research_brief(state["raw_query"], state["clarification_answers"])
        return {
            "research_brief": brief,
            "messages": [
                f"[clarification] Answers pre-provided ({len(state['clarification_answers'])} Q&As). "
                "Research brief generated."
            ]
        }

    # ── Interactive path: CLI ───────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]MICRA — Market Intelligence & Competitive Research Agent[/bold cyan]\n"
        "[dim]Phase 1: Clarification[/dim]",
        border_style="cyan"
    ))

    console.print(f"\n[bold]Research Query:[/bold] {state['raw_query']}\n")
    console.print("[dim]Generating clarifying questions...[/dim]\n")

    # Step A: Generate questions
    questions = generate_clarification_questions(state["raw_query"])

    # Step B: Ask questions interactively
    answers: list[ClarificationAnswer] = []

    console.print("[bold yellow]I need a few details to scope the research correctly.[/bold yellow]\n")

    for i, question in enumerate(questions, 1):
        answer = Prompt.ask(f"[cyan]{i}. {question}[/cyan]")
        answers.append({
            "question": question,
            "answer": answer
        })

    console.print("\n[dim]Building research brief from your answers...[/dim]")

    # Step C: Synthesize research brief
    brief = build_research_brief(state["raw_query"], answers)

    console.print(Panel(
        brief,
        title="[bold green]Research Brief[/bold green]",
        border_style="green"
    ))

    # LEARNING: We return ONLY the fields this node changed.
    # LangGraph merges this dict into the full state.
    # The `messages` list uses operator.add so our append won't overwrite.
    return {
        "clarification_answers": answers,
        "research_brief": brief,
        "messages": [f"[clarification] Asked {len(questions)} questions. Research brief generated."]
    }
