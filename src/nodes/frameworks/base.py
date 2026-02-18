"""
MICRA — Base Framework Agent
==============================

LEARNING CONCEPT: The Template Method pattern + Retry logic

Every framework agent (Porter, SWOT, TAM, Kano) does the same 4 steps:
  1. Retrieve relevant chunks from the knowledge base
  2. Format context for the LLM prompt
  3. Call LLM with structured output schema
  4. Return a FrameworkOutput

Only step 3 varies — each framework has a different schema and prompt.

The Template Method pattern solves this: the BASE CLASS defines the
shared skeleton (steps 1, 2, 4), and SUBCLASSES fill in step 3.

This keeps framework agents DRY — they only define what's unique.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING CONCEPT: Retry with tenacity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Even with JSON mode / structured outputs, LLMs occasionally fail:
  - Network timeout
  - Rate limit hit
  - Pydantic validation fails (LLM returned almost-valid JSON)

tenacity gives us retry logic with one decorator:

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValidationError, Exception))
    )
    def call_llm(...): ...

Reading this: "Try up to 3 times. Between retries, wait 2s, then 4s,
then 8s (exponential backoff). Retry on ValidationError or any exception."

Exponential backoff is important for rate limits — if the API is under
load, hammering it immediately makes things worse. Waiting progressively
longer gives the system time to recover.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING CONCEPT: @tool decorator (introduced here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain's @tool decorator converts a Python function into a Tool
object that LLMs can call via function calling / tool use.

    @tool
    def retrieve_market_data(query: str) -> str:
        \"\"\"Search the knowledge base for market data.\"\"\"
        chunks = retriever.retrieve(query)
        return retriever.format_context(chunks)

Why this matters: when you give an LLM a list of tools, it can DECIDE
which tool to call based on the task. This is how autonomous agents work.

In our framework agents, we DON'T use tool-calling — we call the
retriever directly because the retrieval queries are deterministic.
But in Phase 5 (clarification), you'll see tool-calling in action.

The @tool pattern is shown here so you understand it before Phase 5.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool  # ← the @tool decorator
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.state import FrameworkOutput
from src.retriever import Retriever, RetrievedChunk
from src.config import LLM_MODEL, LLM_TEMPERATURE, TOP_K


# ── Example: what @tool looks like (not used directly here, shown for learning)
# ─────────────────────────────────────────────────────────────────────────────
# @tool
# def search_knowledge_base(query: str) -> str:
#     """Search the research knowledge base for relevant information."""
#     # The docstring becomes the tool description — the LLM reads it to decide
#     # whether to call this tool. Write it clearly!
#     chunks = retriever.retrieve(query)
#     return retriever.format_context(chunks)
#
# Usage: llm.bind_tools([search_knowledge_base]).invoke(messages)
# The LLM will call search_knowledge_base when it decides it needs info.
# ─────────────────────────────────────────────────────────────────────────────


class FrameworkAgent(ABC):
    """
    Abstract base class for all strategic framework agents.

    Subclasses must define:
      - name: str — the framework identifier
      - retrieval_queries: list[str] — what to retrieve for this framework
      - output_schema: type[BaseModel] — Pydantic schema for LLM output
      - system_prompt: str — framework-specific instructions

    Subclasses must implement:
      - _build_user_prompt(brief, context) → str
    """

    name: str
    retrieval_queries: list[str]
    output_schema: type[BaseModel]
    system_prompt: str

    def run(
        self,
        research_brief: str,
        retriever: Retriever,
        source_type_filter: str | None = None,
    ) -> FrameworkOutput:
        """
        Execute the full framework analysis pipeline.

        LEARNING: This is the Template Method.
        The skeleton is defined here. Subclasses override _build_user_prompt.

        Steps:
          1. Retrieve relevant chunks (using multi-query for better coverage)
          2. Build the grounded user prompt
          3. Call LLM with structured output (with retry)
          4. Estimate confidence from source quality
          5. Return FrameworkOutput for state
        """
        # Step 1: Retrieve
        chunks = retriever.retrieve_multi_query(
            queries=self.retrieval_queries,
            k_per_query=TOP_K // len(self.retrieval_queries) + 2,
            source_type=source_type_filter,
        )

        context = retriever.format_context(chunks, include_source=True)

        # Step 2: Build prompt
        user_prompt = self._build_user_prompt(research_brief, context)

        # Step 3: Call LLM with retry
        result = self._call_llm_with_retry(user_prompt)

        # Step 4: Estimate confidence
        confidence = self._estimate_confidence(chunks)

        # Step 5: Return FrameworkOutput
        return FrameworkOutput(
            framework_name=self.name,
            output=result.model_dump(),
            source_chunks=[c.chunk_id for c in chunks],
            confidence=confidence,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((ValidationError, ValueError, Exception)),
        reraise=True,
    )
    def _call_llm_with_retry(self, user_prompt: str) -> BaseModel:
        """
        LEARNING: This is where retry logic lives.

        The @retry decorator from tenacity wraps this method.
        If the LLM returns invalid JSON or the call fails:
          - Attempt 1 fails → wait 2s → Attempt 2
          - Attempt 2 fails → wait 4s → Attempt 3
          - Attempt 3 fails → reraise=True means the exception propagates

        with_structured_output() uses OpenAI's function calling feature
        to guarantee JSON. But "guarantee" means "most of the time" —
        complex schemas can still occasionally produce validation errors.

        That's what retry handles.
        """
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        structured_llm = llm.with_structured_output(self.output_schema)

        return structured_llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ])

    @abstractmethod
    def _build_user_prompt(self, research_brief: str, context: str) -> str:
        """Build the framework-specific user prompt. Subclasses implement this."""
        pass

    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """
        Estimate how confident we are in the output based on source quality.

        LEARNING: Confidence estimation is a simple heuristic here,
        but it teaches an important concept: the quality of RAG output
        depends on the quality and quantity of retrieved context.

        Heuristic:
          - 0 chunks → 0.0 confidence (pure hallucination territory)
          - 1-3 chunks, low similarity → 0.3
          - 4+ chunks, avg similarity > 0.6 → 0.8+

        In Phase 6 (AI evaluation), we'll replace this simple heuristic
        with a proper LLM-based evaluation.
        """
        if not chunks:
            return 0.0

        avg_similarity = sum(c.similarity for c in chunks) / len(chunks)
        chunk_count_factor = min(len(chunks) / 8, 1.0)  # maxes out at 8 chunks

        return round(avg_similarity * 0.6 + chunk_count_factor * 0.4, 2)
