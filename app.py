"""
MICRA — Streamlit Web Interface
=================================

Run with:
    streamlit run app.py

LEARNING CONCEPT: Multi-interface design

The same LangGraph pipeline powers two UIs:
  - CLI (main.py)  — interactive terminal with Rich formatting
  - Web (app.py)   — Streamlit form-based interface

How is this possible without changing the agent?

The clarification_node has TWO paths:
  - Interactive path  → asks questions via CLI Prompt.ask()
  - Pre-loaded path   → reads answers already in state (used here)

When we call the graph from Streamlit, we pre-populate
`clarification_answers` in the initial state. The node detects
this and skips the interactive step entirely.

This is the right architecture: separate UI from agent logic.
The agent doesn't know or care what called it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STREAMLIT CONCEPTS USED HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.session_state — persistent state across reruns
  Streamlit reruns the ENTIRE script on every user interaction.
  (click a button → full re-execution from line 1)
  session_state is a dict that persists between reruns.
  This is how we remember "we're on the clarification step" after
  the user clicks "Generate Questions".

st.status — collapsible progress block for long-running tasks
  Shows a spinner while running. On completion, collapses with a
  summary. We update it inside the pipeline loop so users see live
  progress as each node completes.

st.download_button — file serving from Python
  Streamlit serves the .docx bytes directly from the Python process.
  No S3, no file server needed for a local or small-team deployment.

st.form — batch all inputs before submission
  Without a form, each text_input triggers a full rerun as the user
  types. With st.form, all inputs are batched until the submit
  button is clicked — giving us a proper "fill this out, then submit"
  UX for the clarification questions.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Environment setup ───────────────────────────────────────────────────────
# Load .env before any imports that check for API keys.
# sys.path insert lets Python find `src/` when running from project root.
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))


# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="MICRA — Market Intelligence Agent",
    page_icon="🔍",
    layout="wide",
)


# ── Lazy imports — only imported when API key is confirmed present ───────────
# We don't import graph/LLM modules at the top because they'll fail if
# OPENAI_API_KEY is missing. Instead, we import inside the stage handlers
# where we've already validated the key.

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


# ── Session state initialization ─────────────────────────────────────────────
# LEARNING: Always initialize session_state keys with defaults before reading.
# On the very first run, none of these keys exist yet.

def _init_state():
    defaults = {
        "stage": "query",       # "query" | "clarify" | "results"
        "query": "",
        "questions": [],
        "final_state": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── API key check ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not set. Add it to your `.env` file and restart.")
    st.code("OPENAI_API_KEY=sk-...", language="bash")
    st.stop()   # Halt rendering — nothing below runs


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 MICRA")
st.caption("Market Intelligence & Competitive Research Agent")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Query input
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == "query":
    st.subheader("What market do you want to research?")
    st.write("Describe your question in natural language. MICRA will ask clarifying questions, research the market, apply strategic frameworks, and generate a full report.")
    st.write("")

    query = st.text_area(
        "Research query",
        placeholder="e.g. Analyze the DERMS market in USA. Should L&T build or partner?",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("Generate Clarifying Questions →", type="primary"):
        if not query.strip():
            st.error("Please enter a research query.")
        else:
            st.session_state.query = query.strip()

            with st.spinner("Thinking about the right questions to ask..."):
                from src.nodes.clarification import generate_clarification_questions
                questions = generate_clarification_questions(query.strip())

            st.session_state.questions = questions
            st.session_state.stage = "clarify"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Clarification questions
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "clarify":
    st.subheader("A few details before we begin")

    st.info(f"**Query:** {st.session_state.query}")
    st.write("Answer as much as you know. Type `skip` or leave blank to move on.")
    st.write("")

    # LEARNING: st.form batches all widget interactions until submit.
    # Without a form, every keystroke triggers a full Streamlit rerun
    # (which would re-call the LLM on every character typed!).
    with st.form("clarification_form"):
        answer_inputs = {}
        for i, question in enumerate(st.session_state.questions, 1):
            answer_inputs[question] = st.text_input(
                f"{i}. {question}",
                key=f"q_{i}",
            )

        col1, col2 = st.columns([2, 1])
        with col1:
            start = st.form_submit_button("🚀  Start Research", type="primary", use_container_width=True)
        with col2:
            back = st.form_submit_button("← Change Query", use_container_width=True)

    if back:
        st.session_state.stage = "query"
        st.rerun()

    if start:
        # Build clarification_answers list (skip blanks gracefully)
        clarification_answers = [
            {"question": q, "answer": a.strip() if a.strip() else "Not specified"}
            for q, a in answer_inputs.items()
        ]

        unanswered = sum(1 for a in answer_inputs.values() if not a.strip())
        if unanswered:
            st.warning(f"{unanswered} question(s) left blank — marked as 'Not specified'.")

        # ── Build initial state with pre-loaded answers ───────────────────
        # LEARNING: This is the key to the dual-interface design.
        # We build state the same way main.py does, but pre-populate
        # clarification_answers so the clarification_node skips the CLI step.
        from src.graph import build_graph, create_initial_state

        initial_state = create_initial_state(st.session_state.query)
        initial_state["clarification_answers"] = clarification_answers

        # ── Run the pipeline with live progress updates ────────────────────
        # LEARNING: st.status is a context manager that:
        #   - Shows a spinner while inside the `with` block
        #   - Accepts .update() calls to change label/state
        #   - expanded=True shows live writes; collapses on completion
        final_state = None

        with st.status("Running MICRA pipeline...", expanded=True) as status:
            try:
                graph = build_graph()
                completed_nodes = []

                # LEARNING: stream_mode="values" yields the full accumulated
                # state after every node completes. The last snapshot is the
                # final state. This runs the graph exactly once.
                for state_snapshot in graph.stream(initial_state, stream_mode="values"):
                    final_state = state_snapshot

                    # Detect which node just finished by reading the last message
                    messages = state_snapshot.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        node = next(
                            (n for n in NODE_LABELS if f"[{n}]" in last_msg),
                            None,
                        )
                        if node and node not in completed_nodes:
                            completed_nodes.append(node)
                            st.write(f"✅ {NODE_LABELS[node]}")

                status.update(
                    label="Pipeline complete!",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                status.update(
                    label=f"Pipeline failed: {e}",
                    state="error",
                    expanded=True,
                )
                st.error(f"**Error:** {e}")
                st.stop()

        if final_state:
            st.session_state.final_state = final_state
            st.session_state.stage = "results"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Results
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == "results":
    final_state = st.session_state.final_state

    st.subheader("Research Complete")
    st.caption(f"**Query:** {st.session_state.query}")
    st.write("")

    # ── Evaluation scorecard ──────────────────────────────────────────────────
    evaluation = final_state.get("evaluation", {})
    if evaluation:
        score = evaluation.get("overall_score", 0)
        passed = evaluation.get("passed", False)

        # LEARNING: st.metric shows a number with an optional delta arrow.
        # Great for dashboards and quality scorecards.
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Overall Score",
                f"{score:.0%}",
                delta="PASS" if passed else "FAIL",
                delta_color="normal" if passed else "inverse",
            )
        with col2:
            st.metric("Factual Grounding", f"{evaluation.get('factual_grounding', 0):.0%}")
        with col3:
            st.metric("Framework Correctness", f"{evaluation.get('framework_correctness', 0):.0%}")
        with col4:
            st.metric("Completeness", f"{evaluation.get('completeness', 0):.0%}")
        with col5:
            st.metric("Strategic Coherence", f"{evaluation.get('strategic_coherence', 0):.0%}")

        if not passed:
            st.warning(
                f"Quality below threshold — review before sharing. "
                f"*{evaluation.get('recommendation', '')}*"
            )

        flagged = evaluation.get("flagged_claims", [])
        if flagged:
            with st.expander(f"⚠️ {len(flagged)} flagged claim(s) to verify before sharing"):
                for claim in flagged:
                    st.markdown(f"- {claim}")

    st.divider()

    # ── Report download ───────────────────────────────────────────────────────
    # LEARNING: st.download_button serves bytes directly from Python.
    # No external file server needed — perfect for internal tools.
    report_path = final_state.get("report_path", "")

    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            report_bytes = f.read()

        st.download_button(
            label="📥 Download Full Report (.docx)",
            data=report_bytes,
            file_name=os.path.basename(report_path),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )
        st.caption(f"Also saved locally at: `{report_path}`")
    else:
        st.error("Report file not found. Check terminal output for errors.")

    st.divider()

    # ── Research brief ────────────────────────────────────────────────────────
    with st.expander("📋 Research Brief (used to guide all analysis)"):
        st.write(final_state.get("research_brief", "Not available"))

    # ── Build/Buy/Partner decision ─────────────────────────────────────────────
    bbp = final_state.get("build_buy_partner_decision", {})
    if bbp:
        with st.expander("🏗️ Build / Buy / Partner Decision"):
            recommendation = bbp.get("recommendation", "")
            if recommendation:
                st.markdown(f"**Recommendation: {recommendation.upper()}**")
            reasoning = bbp.get("reasoning", "")
            if reasoning:
                st.write(reasoning)

    # ── MVP recommendation ────────────────────────────────────────────────────
    mvp = final_state.get("mvp_recommendation", {})
    if mvp:
        with st.expander("🎯 MVP Recommendation"):
            must_haves = mvp.get("must_have_features", [])
            if must_haves:
                st.markdown("**Must-have features:**")
                for f in must_haves:
                    st.markdown(f"- {f}")
            north_star = mvp.get("north_star_metric", "")
            if north_star:
                st.markdown(f"**North Star Metric:** {north_star}")

    # ── Sources ───────────────────────────────────────────────────────────────
    sources = final_state.get("sources", [])
    if sources:
        with st.expander(f"📚 Sources Ingested ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                title = src.get("title") or src["url"]
                source_type = src.get("source_type", "web")
                st.markdown(f"{i}. [{title}]({src['url']}) — `{source_type}`")

    # ── Pipeline messages ─────────────────────────────────────────────────────
    messages = final_state.get("messages", [])
    if messages:
        with st.expander("🔍 Pipeline Log"):
            for msg in messages:
                st.text(msg)

    # ── Warnings ─────────────────────────────────────────────────────────────
    errors = [e for e in final_state.get("errors", []) if e]
    if errors:
        with st.expander(f"⚠️ {len(errors)} warning(s) during run"):
            for e in errors:
                st.text(e)

    # ── Start over ────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Research Another Market", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
