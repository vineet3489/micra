"""
MICRA — Phase 1 Tests
========================

LEARNING CONCEPT: Testing LangGraph nodes

Testing agents has a key insight:
  TEST NODES IN ISOLATION, not the full graph.

Why? Because:
  1. The full graph makes LLM calls = expensive + slow + non-deterministic
  2. Individual nodes are just functions — easy to test with mock inputs
  3. You can test graph STRUCTURE separately from node LOGIC

We test three things:
  A. Graph structure (nodes and edges are wired correctly)
  B. State initialization (initial state has correct shape)
  C. Node logic with mocked LLM calls (fast, free, deterministic)

Run with:
    pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock

from src.state import MICRAState
from src.graph import build_graph, create_initial_state


class TestGraphStructure:
    """
    LEARNING: Test the graph structure without running it.
    LangGraph compiled graphs expose their node/edge structure.
    """

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_initial_state_has_required_fields(self):
        """Initial state should have all fields with correct types."""
        state = create_initial_state("Test query")

        assert state["raw_query"] == "Test query"
        assert isinstance(state["clarification_answers"], list)
        assert isinstance(state["sources"], list)
        assert isinstance(state["messages"], list)
        assert isinstance(state["errors"], list)
        assert state["vector_db_ready"] is False
        assert state["research_brief"] == ""

    def test_initial_state_lists_are_empty(self):
        """Lists should be empty, not None."""
        state = create_initial_state("Test query")
        list_fields = [
            "clarification_answers", "sources", "framework_outputs",
            "competitor_profiles", "messages", "errors"
        ]
        for field in list_fields:
            assert state[field] == [], f"Expected {field} to be [], got {state[field]}"


class TestClarificationNode:
    """
    LEARNING: Test a node by mocking the LLM.

    We don't want real OpenAI calls in tests — they're:
      - Slow (network round trip)
      - Expensive (costs money per call)
      - Non-deterministic (different output each run)

    Instead, we use unittest.mock.patch to replace the LLM
    with a mock that returns a controlled, predictable response.

    This is called "mocking" — a fundamental testing technique.
    """

    def test_clarification_node_returns_required_fields(self):
        """
        Clarification node must return clarification_answers,
        research_brief, and messages.
        """
        from src.nodes.clarification import clarification_node

        mock_state = create_initial_state("Analyze DERMS market in USA")

        # Mock the LLM calls AND the user input (Prompt.ask)
        with patch("src.nodes.clarification.ChatOpenAI") as mock_llm_class, \
             patch("src.nodes.clarification.Prompt.ask", return_value="India"):

            mock_instance = MagicMock()
            mock_llm_class.return_value = mock_instance
            mock_instance.invoke.side_effect = [
                MagicMock(content='["What geography?", "What company?"]'),
                MagicMock(content="This is the research brief about DERMS in India.")
            ]

            result = clarification_node(mock_state)

        assert "clarification_answers" in result
        assert "research_brief" in result
        assert "messages" in result
        assert len(result["messages"]) > 0
        assert isinstance(result["clarification_answers"], list)

    def test_clarification_node_logs_to_messages(self):
        """Messages should contain a log entry from clarification."""
        from src.nodes.clarification import clarification_node

        mock_state = create_initial_state("Test")

        with patch("src.nodes.clarification.ChatOpenAI") as mock_llm_class, \
             patch("src.nodes.clarification.Prompt.ask", return_value="test answer"):

            mock_instance = MagicMock()
            mock_llm_class.return_value = mock_instance
            mock_instance.invoke.side_effect = [
                MagicMock(content='["Question 1?", "Question 2?"]'),
                MagicMock(content="Research brief here.")
            ]

            result = clarification_node(mock_state)

        assert any("[clarification]" in msg for msg in result["messages"])


class TestPlannerNode:
    """Tests for the planner node."""

    def test_planner_node_returns_research_plan(self):
        """Planner must return a research_plan dict."""
        from src.nodes.planner import planner_node
        from src.nodes.planner import ResearchPlan

        state = create_initial_state("Test")
        state["research_brief"] = "Analyze DERMS market in USA for L&T."

        mock_plan = ResearchPlan(
            target_market="DERMS",
            geography="USA",
            company_context="L&T evaluating market entry",
            source_types_to_query=["web_competitors", "academic_papers"],
            competitor_names_to_research=["AutoGrid", "Oracle Utilities"],
            search_queries=["DERMS market leaders USA 2024"],
            frameworks_to_apply=["porter_5_forces", "swot", "tam_sam_som"],
            sub_questions=["What is the TAM?", "Who are the key players?"],
            estimated_complexity="high"
        )

        with patch("src.nodes.planner.ChatOpenAI") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            mock_llm_instance.with_structured_output.return_value.invoke.return_value = mock_plan

            result = planner_node(state)

        assert "research_plan" in result
        assert result["research_plan"]["target_market"] == "DERMS"
        assert "porter_5_forces" in result["research_plan"]["frameworks_to_apply"]

    def test_planner_logs_to_messages(self):
        """Planner must log to messages."""
        from src.nodes.planner import planner_node, ResearchPlan

        state = create_initial_state("Test")
        state["research_brief"] = "Test brief."

        mock_plan = ResearchPlan(
            target_market="Test Market",
            geography="Global",
            company_context="Test company",
            source_types_to_query=["web_competitors"],
            competitor_names_to_research=[],
            search_queries=["test query"],
            frameworks_to_apply=["swot"],
            sub_questions=["Test question?"],
            estimated_complexity="low"
        )

        with patch("src.nodes.planner.ChatOpenAI") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            mock_llm_instance.with_structured_output.return_value.invoke.return_value = mock_plan

            result = planner_node(state)

        assert any("[planner]" in msg for msg in result["messages"])


class TestStateAccumulation:
    """
    LEARNING: Test that list fields with operator.add accumulate correctly.

    When two nodes both append to `messages`, the final state should
    have messages from BOTH nodes, not just the last one.
    This verifies our Annotated[list, operator.add] type hint works.
    """

    def test_messages_accumulate_across_nodes(self):
        """Messages from multiple nodes should stack, not overwrite."""
        from langgraph.graph import StateGraph, START, END

        def node_a(state: MICRAState) -> dict:
            return {"messages": ["message from A"]}

        def node_b(state: MICRAState) -> dict:
            return {"messages": ["message from B"]}

        graph = StateGraph(MICRAState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        compiled = graph.compile()

        result = compiled.invoke(create_initial_state("test"))

        assert "message from A" in result["messages"]
        assert "message from B" in result["messages"]
        assert len(result["messages"]) == 2
