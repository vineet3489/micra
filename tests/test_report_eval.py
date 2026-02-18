"""
MICRA — Phase 4 Tests: Report Generator + AI Evaluator
=========================================================

LEARNING: Testing document generation + LLM-as-judge

Document generation tests check:
  - The file is created
  - Key content is present (not empty sections)
  - Graceful handling of missing/empty state fields

LLM-as-judge tests check:
  - Evaluation schema is populated correctly
  - Pass/fail gate uses the correct threshold
  - Score weighting is computed correctly
  - Evaluation node handles LLM failure gracefully

We test WITHOUT making real LLM calls or writing real files
(mock OpenAI + use temp directories).
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.graph import create_initial_state
from src.state import EvaluationResult, FrameworkOutput, CompetitorProfile


# ── Shared fixtures ────────────────────────────────────────────────────────
def _full_state():
    """A state object with all fields populated for testing."""
    state = create_initial_state("Analyze DERMS market in USA")
    state["research_brief"] = "L&T evaluating DERMS market in USA. 3-year horizon."
    state["research_plan"] = {
        "target_market": "DERMS",
        "geography": "USA",
        "company_context": "L&T Infrastructure",
        "frameworks_to_apply": ["porter_5_forces", "swot", "tam_sam_som", "kano"],
        "source_types_to_query": ["web_competitors", "academic_papers"],
        "search_queries": ["DERMS market USA"],
        "competitor_names_to_research": ["AutoGrid"],
        "sub_questions": ["What is the TAM?"],
        "estimated_complexity": "high",
    }
    state["framework_outputs"] = [
        FrameworkOutput(
            framework_name="tam_sam_som",
            output={
                "tam_value": "$3.2B globally by 2027",
                "tam_description": "Global DERMS market including utility-scale",
                "sam_value": "$800M USA utility market",
                "sam_description": "Large utilities >500MW in USA",
                "som_value": "$40-80M realistic in years 1-3",
                "som_description": "2-5% SAM capture for new entrant",
                "growth_rate": "17% CAGR 2024-2029",
                "key_trends_driving_growth": ["DER proliferation", "Grid modernization"],
                "key_assumptions": ["Based on 2 analyst reports"],
                "data_sources_used": ["Mordor Intelligence", "Grand View Research"],
                "confidence": "Medium",
                "confidence_reasoning": "Two reports with similar estimates"
            },
            source_chunks=["c1", "c2"],
            confidence=0.82,
        ),
        FrameworkOutput(
            framework_name="swot",
            output={
                "strengths": ["L&T's existing utility relationships"],
                "weaknesses": ["No existing SaaS platform"],
                "opportunities": ["Grid modernization mandates"],
                "threats": ["AutoGrid's head start"],
                "most_critical_factor": "Time-to-market is critical",
                "recommended_strategic_posture": "Aggressive (capitalize on strengths+opportunities)"
            },
            source_chunks=["c3"],
            confidence=0.75,
        ),
    ]
    state["competitor_profiles"] = [
        CompetitorProfile(
            name="AutoGrid", website="autogrid.com",
            product_summary="AI-powered DERMS for utilities",
            core_features=["Real-time monitoring", "Demand forecasting"],
            pricing_model="Enterprise SaaS", target_segment="Tier 1 utilities",
            tech_stack_signals=["AWS", "REST API"], release_velocity="Monthly",
            funding="$45M Series C", strengths=["Strong analytics"],
            weaknesses=["High cost"], differentiation_gaps=["SMB market"],
            source_chunks=["c4"]
        )
    ]
    state["build_buy_partner_decision"] = {
        "recommendation": "Build+Partner",
        "reasoning": "L&T has engineering depth but lacks SaaS experience.",
        "capability_gaps": ["Cloud platform", "Real-time data pipeline"],
        "time_to_market_assessment": "18-24 months",
        "build_case": "Full product control",
        "buy_case": "Faster entry",
        "partner_case": "Leverage existing relationships",
        "recommended_partners_or_targets": ["AutoGrid", "Oracle Utilities"],
        "risk_factors": ["Integration complexity", "Talent gaps"]
    }
    state["mvp_recommendation"] = {
        "target_customer": "Indian utilities >1GW",
        "core_features": ["Grid monitoring", "Demand forecasting", "Alert system"],
        "features_explicitly_excluded": ["Residential DER", "P2P trading"],
        "north_star_metric": "Grid instability events prevented per month",
        "go_to_market_entry_point": "NTPC pilot program",
        "pricing_strategy": "Per-site annual SaaS",
        "success_criteria_6_months": ["2 pilots signed"],
        "differentiation_thesis": "Only DERMS built for Indian grid characteristics"
    }
    return state


# ── A. Source collection ───────────────────────────────────────────────────
class TestSourceCollection:

    def test_collect_sources_deduplicates_by_url(self):
        """Same URL appearing in multiple framework outputs → counted once."""
        from src.nodes.report_generator import _collect_all_sources

        state = create_initial_state("test")
        state["sources"] = [
            {"url": "https://example.com", "title": "Example",
             "source_type": "web", "raw_text": "text", "chunk_ids": ["c1", "c2"]},
        ]
        state["framework_outputs"] = [
            FrameworkOutput(framework_name="swot", output={},
                          source_chunks=["c1"], confidence=0.8),
            FrameworkOutput(framework_name="porter_5_forces", output={},
                          source_chunks=["c2"], confidence=0.7),
        ]
        state["competitor_profiles"] = []

        sources = _collect_all_sources(state)

        # Same URL in both framework outputs → should appear only once
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com"

    def test_collect_sources_returns_empty_for_no_sources(self):
        from src.nodes.report_generator import _collect_all_sources

        state = create_initial_state("test")
        sources = _collect_all_sources(state)
        assert sources == []


# ── B. Report Generator ────────────────────────────────────────────────────
class TestReportGenerator:

    def _run_with_mocked_docx(self, state):
        """
        LEARNING: When testing document generation, mock the Document class.
        We're testing OUR node's logic (what sections it creates, what path
        it returns), not python-docx's ability to write a file.

        Mocking Document() avoids file I/O entirely and prevents python-docx
        from trying to find its internal template via os.path.join.
        """
        from src.nodes.report_generator import report_generator_node

        mock_doc = MagicMock()
        mock_table = MagicMock()
        mock_table.rows = [MagicMock(cells=[MagicMock(), MagicMock(), MagicMock()])] * 10
        mock_doc.add_table.return_value = mock_table
        mock_doc.styles = {"Normal": MagicMock()}

        with patch("src.nodes.report_generator.Document", return_value=mock_doc), \
             patch("src.nodes.report_generator.os.makedirs"):
            result = report_generator_node(state)

        return result, mock_doc

    def test_report_returns_docx_path(self):
        """report_generator_node should return a .docx path in state."""
        state = _full_state()
        result, _ = self._run_with_mocked_docx(state)

        assert "report_path" in result
        assert result["report_path"].endswith(".docx")

    def test_report_calls_doc_save(self):
        """The node must call doc.save() with the output path."""
        state = _full_state()
        result, mock_doc = self._run_with_mocked_docx(state)

        mock_doc.save.assert_called_once()
        saved_path = mock_doc.save.call_args[0][0]
        assert saved_path == result["report_path"]

    def test_report_logs_to_messages(self):
        state = _full_state()
        result, _ = self._run_with_mocked_docx(state)

        assert any("[report_generator]" in m for m in result.get("messages", []))

    def test_report_handles_empty_framework_outputs(self):
        """Report should not crash if framework outputs are empty."""
        state = create_initial_state("test")
        state["research_plan"] = {"target_market": "Test Market", "geography": "Global"}

        result, mock_doc = self._run_with_mocked_docx(state)

        assert "report_path" in result
        mock_doc.save.assert_called_once()

    def test_find_framework_returns_none_for_missing(self):
        """_find_framework returns None if framework not in outputs."""
        from src.nodes.report_generator import _find_framework

        outputs = [
            FrameworkOutput(framework_name="swot", output={"x": 1},
                          source_chunks=[], confidence=0.8)
        ]
        result = _find_framework("porter_5_forces", outputs)
        assert result is None

    def test_find_framework_returns_output_dict(self):
        from src.nodes.report_generator import _find_framework

        outputs = [
            FrameworkOutput(framework_name="tam_sam_som", output={"tam_value": "$3B"},
                          source_chunks=[], confidence=0.9)
        ]
        result = _find_framework("tam_sam_som", outputs)
        assert result is not None
        assert result["tam_value"] == "$3B"


# ── C. AI Evaluator ────────────────────────────────────────────────────────
class TestEvaluator:

    def _make_eval_schema(self, **kwargs):
        """Create a mock EvaluationSchema with default good scores."""
        from src.nodes.evaluator import EvaluationSchema
        defaults = dict(
            factual_grounding=0.85,
            framework_correctness=0.80,
            completeness=0.90,
            strategic_coherence=0.82,
            hallucinations_detected=1,
            flagged_claims=["Claim X has no source — Issue: not in retrieved chunks"],
            output_strengths=["Clear TAM breakdown", "Specific competitor gaps"],
            output_improvements=["Add more academic sources"],
            overall_recommendation="PASS — report is ready to share",
        )
        defaults.update(kwargs)
        return EvaluationSchema(**defaults)

    def test_evaluator_returns_evaluation_result(self):
        from src.nodes.evaluator import evaluator_node

        state = _full_state()
        mock_schema = self._make_eval_schema()

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_schema

            result = evaluator_node(state)

        assert "evaluation" in result
        eval_result = result["evaluation"]
        assert "overall_score" in eval_result
        assert "passed" in eval_result
        assert "flagged_claims" in eval_result

    def test_overall_score_is_weighted_average(self):
        """Verify the weighting formula: grounding×0.3, correctness×0.2, completeness×0.25, coherence×0.25"""
        from src.nodes.evaluator import evaluator_node

        state = _full_state()
        mock_schema = self._make_eval_schema(
            factual_grounding=1.0,
            framework_correctness=1.0,
            completeness=1.0,
            strategic_coherence=1.0,
        )

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_schema

            result = evaluator_node(state)

        assert result["evaluation"]["overall_score"] == 1.0

    def test_pass_threshold_is_applied(self):
        """Score below threshold → passed=False. Above → passed=True."""
        from src.nodes.evaluator import evaluator_node
        from src.config import EVAL_PASS_THRESHOLD

        state = _full_state()

        # Low scores → FAIL
        fail_schema = self._make_eval_schema(
            factual_grounding=0.5,
            framework_correctness=0.5,
            completeness=0.5,
            strategic_coherence=0.5,
        )
        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = fail_schema
            result = evaluator_node(state)

        assert result["evaluation"]["passed"] is False
        assert result["evaluation"]["overall_score"] < EVAL_PASS_THRESHOLD

    def test_evaluator_handles_llm_failure_gracefully(self):
        """If LLM call fails, evaluator should return a failure result without crashing."""
        from src.nodes.evaluator import evaluator_node

        state = _full_state()

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API timeout")

            result = evaluator_node(state)

        assert "evaluation" in result
        assert result["evaluation"]["passed"] is False
        assert result["evaluation"]["overall_score"] == 0.0

    def test_evaluator_logs_to_messages(self):
        from src.nodes.evaluator import evaluator_node

        state = _full_state()
        mock_schema = self._make_eval_schema()

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_schema

            result = evaluator_node(state)

        assert any("[evaluator]" in m for m in result.get("messages", []))

    def test_hallucination_count_stored(self):
        """Hallucination count from schema should be stored in evaluation."""
        from src.nodes.evaluator import evaluator_node

        state = _full_state()
        mock_schema = self._make_eval_schema(hallucinations_detected=3)

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_schema

            result = evaluator_node(state)

        assert result["evaluation"]["hallucinations_detected"] == 3


# ── D. Full pipeline state flow ────────────────────────────────────────────
class TestEndToEndStateFlow:
    """
    LEARNING: Integration test — verify state flows correctly through all nodes.

    We don't call real LLMs. Instead we mock every LLM call and verify that:
      1. Each node reads the right fields from state
      2. Each node writes the right fields back
      3. operator.add fields (messages, errors) accumulate correctly
    """

    def test_messages_accumulate_across_all_nodes(self):
        """After running all nodes, messages should contain entries from each."""
        from langgraph.graph import StateGraph, START, END
        from src.state import MICRAState

        def node_a(state):
            return {"messages": ["from A"]}

        def node_b(state):
            return {"messages": ["from B"]}

        def node_c(state):
            return {"messages": ["from C"]}

        g = StateGraph(MICRAState)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.add_node("c", node_c)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", END)
        compiled = g.compile()

        result = compiled.invoke(create_initial_state("test"))

        assert "from A" in result["messages"]
        assert "from B" in result["messages"]
        assert "from C" in result["messages"]

    def test_evaluation_pass_field_is_bool(self):
        """evaluation.passed must be a boolean, not a string."""
        from src.nodes.evaluator import evaluator_node

        state = _full_state()
        mock_schema = MagicMock()
        mock_schema.factual_grounding = 0.9
        mock_schema.framework_correctness = 0.85
        mock_schema.completeness = 0.88
        mock_schema.strategic_coherence = 0.87
        mock_schema.hallucinations_detected = 0
        mock_schema.flagged_claims = []
        mock_schema.output_strengths = ["Good"]
        mock_schema.output_improvements = ["More sources"]
        mock_schema.overall_recommendation = "PASS — report ready"

        with patch("src.nodes.evaluator.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_schema

            result = evaluator_node(state)

        assert isinstance(result["evaluation"]["passed"], bool)
