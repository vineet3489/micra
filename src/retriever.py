"""
MICRA — RAG Retriever
=======================

LEARNING CONCEPT: The "RA" in RAG — Retrieval-Augmented Generation

This is where RAG pays off. When a framework agent wants to reason about
"market competition", it doesn't guess from training data. Instead:

  1. Convert query to embedding ("what are the competitive threats?")
  2. Find the K most similar chunks in ChromaDB
  3. Pass those chunks as context to the LLM prompt

The LLM's answer is now GROUNDED IN YOUR DOCUMENTS — not in its
training data, which might be outdated or hallucinated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW SIMILARITY SEARCH WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you call collection.query(query_embeddings=[...], n_results=K):

  1. ChromaDB takes your query vector
  2. Computes cosine similarity against ALL stored vectors
  3. Returns the K vectors with highest similarity

Cosine similarity is a number between -1 and 1:
  1.0 = identical meaning
  0.8+ = very similar
  0.5  = loosely related
  0.0  = unrelated
  -1.0 = opposite meaning

In practice, relevant chunks score 0.7–0.9 for specific queries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METADATA FILTERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You can filter by metadata BEFORE similarity search:

  retrieve("market size", source_type="academic")
  → only searches academic paper chunks

This is useful for:
  - TAM: retrieve only from academic + regulatory sources (most reliable)
  - Competitor analysis: retrieve only from web_competitors
  - Recent trends: retrieve only from news

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE RETRIEVER IS NOT A NODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The retriever is a utility — a helper function. It's called BY nodes
(framework agents, synthesis agent), not registered as a graph node.
This is an important architectural distinction:

  Node = a graph participant that reads/writes shared state
  Tool/Utility = a stateless function called by nodes

The retriever is stateless — it takes a query and returns chunks.
It doesn't know about graph state.
"""

from dataclasses import dataclass
from typing import Optional

import chromadb
from openai import OpenAI

from src.config import TOP_K, EMBEDDING_MODEL, CHROMA_PERSIST_DIR


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store, with its similarity score."""
    text: str
    source_url: str
    source_type: str
    title: str
    chunk_index: int
    similarity: float    # cosine similarity (higher = more relevant)
    chunk_id: str


class Retriever:
    """
    RAG retriever backed by ChromaDB.

    LEARNING: Why a class instead of a function?

    The retriever needs to maintain connections to ChromaDB and OpenAI.
    A class lets us initialize these once and reuse them across many
    retrieve() calls — much more efficient than reconnecting each time.

    Each framework agent in Phase 4 will call retrieve() multiple times
    with different queries. One Retriever instance per graph run.
    """

    def __init__(self, collection_name: str):
        """
        Initialize retriever for a specific ChromaDB collection.

        Args:
            collection_name: The collection created by the embedder node
                             (stored in state["vector_db_collection_name"])
        """
        self._openai = OpenAI()
        self._chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma.get_collection(name=collection_name)

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        source_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the K most relevant chunks for a query.

        LEARNING: The retrieval process step by step:

        1. Embed the query — convert it to a vector using the same model
           that was used to embed the chunks. This is critical: query and
           chunks must use the SAME embedding model or similarity breaks.

        2. Query ChromaDB — find K chunks with highest cosine similarity
           to the query vector.

        3. Parse results — ChromaDB returns parallel lists (ids, documents,
           metadatas, distances). We zip them into RetrievedChunk objects.

        4. Convert distance to similarity — ChromaDB with cosine space
           returns distances (0 = identical, 2 = opposite). We convert
           to similarity (1 - distance/2) for intuitive interpretation.

        Args:
            query: Natural language query (e.g. "What is the market size?")
            k: How many chunks to retrieve
            source_type: Optional filter ("academic", "web", "news")

        Returns:
            List of RetrievedChunk, sorted by relevance (most relevant first)
        """
        # Step 1: Embed the query
        response = self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding

        # Step 2: Build optional metadata filter
        # LEARNING: ChromaDB uses a "where" clause for metadata filtering.
        # This runs BEFORE the similarity search — filters then ranks.
        where = {"source_type": source_type} if source_type else None

        # Step 3: Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count()),  # can't ask for more than exists
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        # Step 4: Parse into RetrievedChunk objects
        chunks = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Convert cosine distance (0–2) to similarity (0–1)
            # distance=0 → identical (similarity=1.0)
            # distance=2 → opposite  (similarity=0.0)
            similarity = 1 - (dist / 2)

            chunks.append(RetrievedChunk(
                text=doc,
                source_url=meta.get("source_url", ""),
                source_type=meta.get("source_type", ""),
                title=meta.get("title", ""),
                chunk_index=meta.get("chunk_index", 0),
                similarity=round(similarity, 3),
                chunk_id=chunk_id
            ))

        # Already sorted by ChromaDB (most similar first), but make explicit
        return sorted(chunks, key=lambda c: c.similarity, reverse=True)

    def retrieve_multi_query(
        self,
        queries: list[str],
        k_per_query: int = 5,
        source_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks for multiple queries, deduplicating results.

        LEARNING: Multi-query retrieval

        A single query might miss relevant chunks because the wording
        doesn't match how the content was phrased. Multiple queries
        cast a wider net:

            "competitive threats in DERMS market"
            "DERMS market competitors analysis"
            "distributed energy resource management companies"

        All three queries might retrieve different relevant chunks.
        We deduplicate by chunk_id and return the union, sorted by best score.

        This technique is called "query expansion" — a key RAG improvement.
        """
        seen_ids: set[str] = set()
        all_chunks: list[RetrievedChunk] = []

        for query in queries:
            chunks = self.retrieve(query, k=k_per_query, source_type=source_type)
            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        return sorted(all_chunks, key=lambda c: c.similarity, reverse=True)

    def format_context(
        self,
        chunks: list[RetrievedChunk],
        include_source: bool = True
    ) -> str:
        """
        Format retrieved chunks into a context string for LLM prompts.

        LEARNING: How retrieved context is injected into prompts

        The retrieved chunks are concatenated into a "context" block
        that gets injected into the framework agent's prompt:

            CONTEXT (from retrieved sources):
            [1] (Source: autogrid.com) AutoGrid's DERMS platform...
            [2] (Source: ieee.org) Grid instability research shows...

            Based on the context above, analyze Porter's 5 Forces...

        Including source information helps the LLM:
          1. Distinguish between sources (company website vs. academic paper)
          2. Weight information appropriately (cited research > marketing copy)
          3. Generate proper citations in the output

        Args:
            chunks: Retrieved chunks to format
            include_source: Whether to include source URL in each chunk

        Returns:
            Formatted context string ready for LLM prompt injection
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source_label = f"(Source: {chunk.source_url})" if include_source else ""
            parts.append(f"[{i}] {source_label}\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

    @property
    def chunk_count(self) -> int:
        """Total chunks in the collection."""
        return self._collection.count()
