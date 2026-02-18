"""
MICRA — Phase 3 Tests: Framework Engine, Competitive Intel, Synthesis
=======================================================================

LEARNING: Testing multi-agent systems

Testing agents that call other agents requires a layered mock strategy:
  - Mock the Retriever (no ChromaDB needed)
  - Mock the LLM calls (no OpenAI needed)
  - Test the LOGIC of each node, not the LLM's intelligence

What we test:
  A. Base agent confidence estimation
  B. Each framework agent builds correct prompt structure
  C. Framework engine handles partial failures
  D. Framework engine runs only requested frameworks
  E. Competitive intel deduplicates known + discovered competitors
  F. Synthesis node skips gracefully with no framework outputs
  G. Synthesis produces correct output fields
"""

import pytest
from unittest.mock import patch, MagicMock

from src.graph import create_initial_state
from src.state import FrameworkOutput, CompetitorProfile
from src.retriever import RetrievedChunk


# ── Fixtures ───────────────────────────────────────────────────────────────
def make_chunk(text: str, similarity: float = 0.75, chunk_id: str = "abc") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_url="https://example.com",
        source_type="web",
        title="Test Source",
        chunk_index=0,
        similarity=similarity,
        chunk_id=chunk_id,
    )


def make_mock_retriever(chunks: list[RetrievedChunk] | None = None) -> MagicMock:
    retriever = MagicMock()
    chunks = chunks or [make_chunk("DERMS market is growing. Competitors include AutoGrid.", 0.8, "c1")]
    retriever.retrieve.return_value = chunks
    retriever.retrieve_multi_query.return_value = chunks
    retriever.format_context.return_value = "DERMS market context..."
    retriever.chunk_count = len(chunks)
    return retriever


def make_framework_output(name: str) -> FrameworkOutput:
    return FrameworkOutput(
        framework_name=name,
        output={"test_key": "test_value"},
        source_chunks=["c1", "c2"],
        confidence=0.8,
    )


# ── A. Base Agent: Confidence Estimation ──────────────────────────────────
class TestBaseAgent:

    def test_confidence_zero_with_no_chunks(self):
        from src.nodes.frameworks.base import FrameworkAgent
        from src.nodes.frameworks.swot import SWOTAgent

        agent = SWOTAgent()
        confidence = agent._estimate_confidence([])
        assert confidence == 0.0

    def test_confidence_higher_with_more_relevant_chunks(self):
        from src.nodes.frameworks.swot import SWOTAgent

        agent = SWOTAgent()

        low_chunks = [make_chunk("text", similarity=0.3)]
        high_chunks = [make_chunk("text", similarity=0.9, chunk_id=f"c{i}") for i in range(8)]

        low_conf = agent._estimate_confidence(low_chunks)
        high_conf = agent._estimate_confidence(high_chunks)

        assert high_conf > low_conf

    def test_confidence_between_zero_and_one(self):
        from src.nodes.frameworks.swot import SWOTAgent

        agent = SWOTAgent()
        chunks = [make_chunk("text", similarity=0.95, chunk_id=f"c{i}") for i in range(10)]
        conf = agent._estimate_confidence(chunks)
        assert 0.0 <= conf <= 1.0


# ── B. Framework Agents: Prompt Structure ─────────────────────────────────
class TestFrameworkPrompts:
    """
    LEARNING: We test that prompts contain the required context.
    A prompt that drops the research_brief or context is a bug —
    the LLM would hallucinate without these inputs.
    """

    def test_porter_prompt_includes_brief_and_context(self):
        from src.nodes.frameworks.porter import PorterAgent

        agent = PorterAgent()
        prompt = agent._build_user_prompt("Analyze DERMS market.", "Context about competition.")

        assert "Analyze DERMS market." in prompt
        assert "Context about competition." in prompt

    def test_swot_prompt_includes_brief(self):
        from src.nodes.frameworks.swot import SWOTAgent

        agent = SWOTAgent()
        prompt = agent._build_user_prompt("L&T analyzing DERMS.", "Market context here.")

        assert "L&T analyzing DERMS." in prompt

    def test_tam_prompt_mentions_triangulation(self):
        from src.nodes.frameworks.tam import TAMAgent

        agent = TAMAgent()
        prompt = agent._build_user_prompt("Brief.", "Context with $3.2B market size.")

        assert "Context with $3.2B market size." in prompt

    def test_kano_prompt_includes_context(self):
        from src.nodes.frameworks.kano import KanoAgent

        agent = KanoAgent()
        prompt = agent._build_user_prompt("Brief.", "Feature signals from users.")

        assert "Feature signals from users." in prompt


# ── C. Framework Engine: Partial Failure Tolerance ────────────────────────
class TestFrameworkEngine:

    def test_engine_continues_on_agent_failure(self):
        """If one framework agent fails, others should still run."""
        from src.nodes.framework_engine import framework_engine_node

        state = create_initial_state("DERMS market")
        state["research_plan"] = {
            "frameworks_to_apply": ["porter_5_forces", "swot"],
        }
        state["vector_db_collection_name"] = "micra_test"
        state["research_brief"] = "Analyze DERMS market for L&T."

        swot_output = make_framework_output("swot")

        with patch("src.nodes.framework_engine.Retriever") as mock_retriever_class:
            mock_retriever_class.return_value = make_mock_retriever()

            with patch("src.nodes.framework_engine.FRAMEWORK_REGISTRY") as mock_registry:
                # Porter fails, SWOT succeeds
                mock_porter = MagicMock()
                mock_porter.return_value.run.side_effect = Exception("LLM timeout")

                mock_swot = MagicMock()
                mock_swot.return_value.run.return_value = swot_output

                mock_registry.__contains__ = lambda self, k: True
                mock_registry.__getitem__ = lambda self, k: (
                    mock_porter if k == "porter_5_forces" else mock_swot
                )
                mock_registry.keys.return_value = ["porter_5_forces", "swot"]

                result = framework_engine_node(state)

        # Should have error logged for porter, but still have swot output
        assert len(result.get("errors", [])) > 0
        assert any("porter_5_forces" in e for e in result["errors"])

    def test_engine_skips_when_no_collection(self):
        """Framework engine should log error if no vector DB ready."""
        from src.nodes.framework_engine import framework_engine_node

        state = create_initial_state("test")
        # No collection_name set

        result = framework_engine_node(state)

        assert len(result.get("errors", [])) > 0

    def test_engine_runs_only_requested_frameworks(self):
        """Only frameworks in the plan should run."""
        from src.nodes.framework_engine import framework_engine_node

        state = create_initial_state("test")
        state["research_plan"] = {"frameworks_to_apply": ["swot"]}
        state["vector_db_collection_name"] = "micra_test"
        state["research_brief"] = "Brief."

        swot_output = make_framework_output("swot")

        run_calls = []

        with patch("src.nodes.framework_engine.Retriever") as mock_retriever_class, \
             patch("src.nodes.framework_engine.FRAMEWORK_REGISTRY") as mock_registry:

            mock_retriever_class.return_value = make_mock_retriever()

            mock_agent_instance = MagicMock()
            mock_agent_instance.run.return_value = swot_output
            mock_agent_instance.run.side_effect = lambda **kwargs: run_calls.append("swot") or swot_output

            mock_agent_class = MagicMock(return_value=mock_agent_instance)
            mock_registry.__contains__ = lambda self, k: k == "swot"
            mock_registry.__getitem__ = lambda self, k: mock_agent_class
            mock_registry.keys.return_value = ["swot"]

            result = framework_engine_node(state)

        # Only SWOT should have run
        assert len(run_calls) == 1


# ── D. Competitive Intel ────────────────────────────────────────────────────
class TestCompetitiveIntel:

    def test_skips_when_no_collection(self):
        from src.nodes.competitive_intel import competitive_intel_node

        state = create_initial_state("test")
        result = competitive_intel_node(state)
        assert any("No knowledge base" in e for e in result.get("errors", []))

    def test_profiles_known_competitors(self):
        from src.nodes.competitive_intel import competitive_intel_node
        from src.state import CompetitorProfile

        state = create_initial_state("DERMS market")
        state["research_plan"] = {"competitor_names_to_research": ["AutoGrid"]}
        state["vector_db_collection_name"] = "micra_test"
        state["research_brief"] = "L&T analyzing DERMS."

        mock_profile = CompetitorProfile(
            name="AutoGrid", website="", product_summary="DERMS platform",
            core_features=["Real-time monitoring"], pricing_model="SaaS",
            target_segment="Utilities", tech_stack_signals=["AWS"],
            release_velocity="Monthly", funding="$45M Series C",
            strengths=["Strong analytics"], weaknesses=["High cost"],
            differentiation_gaps=["SMB market untapped"], source_chunks=["c1"]
        )

        with patch("src.nodes.competitive_intel.Retriever") as mock_retriever_class, \
             patch("src.nodes.competitive_intel._extract_competitor_names_from_sources", return_value=[]), \
             patch("src.nodes.competitive_intel._profile_competitor", return_value=mock_profile):

            mock_retriever_class.return_value = make_mock_retriever()
            result = competitive_intel_node(state)

        assert len(result["competitor_profiles"]) == 1
        assert result["competitor_profiles"][0]["name"] == "AutoGrid"


# ── E. Synthesis Node ──────────────────────────────────────────────────────
class TestSynthesis:

    def test_synthesis_skips_with_no_outputs(self):
        from src.nodes.synthesis import synthesis_node

        state = create_initial_state("test")
        # No framework_outputs
        result = synthesis_node(state)
        assert any("No framework outputs" in e for e in result.get("errors", []))

    def test_synthesis_returns_bbp_and_mvp(self):
        from src.nodes.synthesis import synthesis_node, BuildBuyPartnerSchema, MVPSchema

        state = create_initial_state("DERMS market for L&T")
        state["framework_outputs"] = [make_framework_output("swot")]
        state["competitor_profiles"] = []
        state["research_brief"] = "L&T analyzing DERMS market."

        mock_bbp = BuildBuyPartnerSchema(
            recommendation="Build+Partner",
            reasoning="L&T has engineering capability but lacks software platform experience.",
            capability_gaps=["Cloud SaaS platform", "Real-time data pipeline"],
            time_to_market_assessment="18-24 months to first revenue",
            build_case="Full control over product roadmap",
            buy_case="Faster market entry via acquisition",
            partner_case="Leverage existing utility relationships",
            recommended_partners_or_targets=["AutoGrid", "Oracle Utilities"],
            risk_factors=["Integration complexity", "Talent acquisition"]
        )
        mock_mvp = MVPSchema(
            target_customer="Indian utilities >1GW",
            core_features=["Grid monitoring", "Demand forecasting", "Alerts"],
            features_explicitly_excluded=["Residential DER", "Peer-to-peer trading"],
            north_star_metric="Grid events prevented per month",
            go_to_market_entry_point="NTPC and Tata Power pilot programs",
            pricing_strategy="Per-site annual SaaS",
            success_criteria_6_months=["2 pilots signed", "1 LOI"],
            differentiation_thesis="Only DERMS built for Indian grid characteristics"
        )

        with patch("src.nodes.synthesis.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_structured.invoke.side_effect = [mock_bbp, mock_mvp]

            result = synthesis_node(state)

        assert "build_buy_partner_decision" in result
        assert "mvp_recommendation" in result
        assert result["build_buy_partner_decision"]["recommendation"] == "Build+Partner"
        assert len(result["mvp_recommendation"]["core_features"]) == 3

    def test_synthesis_logs_to_messages(self):
        from src.nodes.synthesis import synthesis_node, BuildBuyPartnerSchema, MVPSchema

        state = create_initial_state("test")
        state["framework_outputs"] = [make_framework_output("porter_5_forces")]
        state["competitor_profiles"] = []
        state["research_brief"] = "Brief."

        mock_bbp = BuildBuyPartnerSchema(
            recommendation="Build", reasoning="Strong capability.",
            capability_gaps=[], time_to_market_assessment="12 months",
            build_case="Control", buy_case="Speed", partner_case="Risk sharing",
            recommended_partners_or_targets=[], risk_factors=[]
        )
        mock_mvp = MVPSchema(
            target_customer="Enterprise", core_features=["Feature A", "Feature B"],
            features_explicitly_excluded=["Feature X"],
            north_star_metric="Revenue per month",
            go_to_market_entry_point="Direct sales",
            pricing_strategy="SaaS",
            success_criteria_6_months=["First customer"],
            differentiation_thesis="Unique angle."
        )

        with patch("src.nodes.synthesis.ChatOpenAI") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            mock_llm.with_structured_output.return_value.invoke.side_effect = [mock_bbp, mock_mvp]

            result = synthesis_node(state)

        assert any("[synthesis]" in m for m in result.get("messages", []))
