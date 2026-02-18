"""
MICRA — Embedder Node
=======================

LEARNING CONCEPT: The core of RAG — Chunk → Embed → Store

This node is where RAG actually happens. Let's understand each step:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: CHUNKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: a scraped page might be 10,000 words. An LLM can only read
~8,000 words at once (128k tokens for GPT-4o, but we don't want to
pass the whole document — that defeats the purpose of RAG).

Solution: split documents into small chunks (512 tokens each).
At retrieval time, we fetch only the 5-10 most relevant chunks —
much cheaper and more focused than passing the whole document.

We use LangChain's RecursiveCharacterTextSplitter, which splits in
this order: paragraphs → sentences → words → characters.
This preserves semantic boundaries as much as possible.

Why "recursive"? It tries the biggest separator first (\n\n = paragraphs),
then falls back to smaller ones (\n = lines, then " " = words).
This means chunks respect paragraph and sentence boundaries when possible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: EMBEDDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An embedding is a list of numbers (a vector) that encodes the MEANING
of a piece of text. Example:

  "grid instability problem" → [0.12, -0.45, 0.83, 0.01, ...]  (1536 numbers)
  "power grid volatility"   → [0.11, -0.47, 0.81, 0.02, ...]  (very similar!)
  "recipe for pasta"        → [0.92, 0.15, -0.33, 0.77, ...]  (very different)

Why are similar meanings close together? Because the embedding model
was trained on billions of text pairs to pull similar meanings together
and push different meanings apart.

This is how we can search by meaning, not just keywords.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: STORING IN CHROMADB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ChromaDB stores:
  - The embedding vector (for similarity search)
  - The original text (returned with results)
  - Metadata: source_url, source_type, title, chunk_index

Each research run gets its own ChromaDB collection (named by run_id).
This keeps runs isolated — you can run MICRA multiple times without
old data polluting new results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenAI's embedding API has a limit of 2048 inputs per request.
We embed in batches of 100 to stay well within limits and to show
progress incrementally. Batching also makes retries more granular
(if one batch fails, we don't redo everything).
"""

import os
import uuid
import logging
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn

from src.state import MICRAState, SourceDocument
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, CHROMA_PERSIST_DIR
)

logger = logging.getLogger(__name__)
console = Console()

EMBED_BATCH_SIZE = 100   # chunks per OpenAI embedding API call


def _get_chroma_client() -> chromadb.PersistentClient:
    """
    LEARNING: ChromaDB persistence

    PersistentClient saves data to disk — your embeddings survive between
    Python sessions. If you restart the script, you don't need to re-embed.

    EphemeralClient() is the in-memory alternative (useful for testing).
    We use PersistentClient for the real pipeline.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _chunk_document(source: SourceDocument) -> list[dict]:
    """
    Split one SourceDocument into chunks with metadata.

    LEARNING: What is chunk metadata?

    Every chunk stores not just text but WHERE it came from:
      - source_url: the original URL
      - source_type: "web", "academic", "news"
      - title: page/paper title
      - chunk_index: position in the original document

    This metadata serves two purposes:
      1. Filtering: "retrieve only academic sources" for the TAM analysis
      2. Citation: "this claim comes from [url], chunk 3"

    Returns:
        List of dicts: {text, metadata, chunk_id}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,   # character count (not token count — simpler)
        separators=["\n\n", "\n", ". ", " ", ""]
        # LEARNING: separator priority — tries \n\n first (paragraph breaks),
        # falls back to \n (line breaks), then ". " (sentence ends), etc.
    )

    text_chunks = splitter.split_text(source["raw_text"])
    chunks = []

    for i, chunk_text in enumerate(text_chunks):
        if not chunk_text.strip():
            continue

        chunk_id = f"{uuid.uuid4().hex[:8]}_{i}"

        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source_url": source["url"],
                "source_type": source["source_type"],
                "title": source["title"][:200],  # ChromaDB metadata has length limits
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            }
        })

    return chunks


def _embed_batch(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    """
    Call OpenAI embeddings API for a batch of texts.

    LEARNING: The embedding API

    Input:  list of strings (up to 2048)
    Output: list of vectors, each of length EMBEDDING_DIMENSIONS (1536)

    Cost: text-embedding-3-small is $0.02 per million tokens.
    For a typical MICRA run (~50 chunks × 512 chars ≈ 150 tokens each),
    that's ~7,500 tokens = ~$0.00015. Essentially free for development.

    The response.data is a list of Embedding objects.
    Each .embedding is the vector (list of floats).
    """
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def embedder_node(state: MICRAState) -> dict:
    """
    Chunk all sources, embed them, store in ChromaDB.

    LEARNING: Notice the clear separation from the scraper node.
    The scraper fetches raw text. The embedder processes it.
    This separation lets you:
      - Re-embed with different chunk sizes without re-scraping
      - Test embedding logic without making HTTP requests
      - Scale scraping and embedding independently
    """
    sources = state.get("sources", [])

    if not sources:
        return {
            "errors": ["[embedder] No sources to embed. Scraper may have failed."],
            "messages": ["[embedder] Skipped — no sources available."]
        }

    console.print("\n[bold cyan]Building RAG Knowledge Base[/bold cyan]")

    # Create a unique collection name for this research run
    # Using a timestamp keeps runs isolated
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_name = f"micra_{run_id}"

    # ── Step 1: Chunk all documents ──────────────────────────────────────
    all_chunks = []
    for source in sources:
        chunks = _chunk_document(source)
        all_chunks.extend(chunks)

    console.print(
        f"  Chunked {len(sources)} sources → [bold]{len(all_chunks)} chunks[/bold] "
        f"(~{CHUNK_SIZE} chars each)"
    )

    # ── Step 2: Set up ChromaDB collection ───────────────────────────────
    chroma = _get_chroma_client()
    collection = chroma.create_collection(
        name=collection_name,
        # LEARNING: distance function for similarity.
        # "cosine" measures the angle between vectors (ignores magnitude).
        # It's the standard choice for text embeddings.
        # Alternative: "l2" (Euclidean distance) — works too but less common.
        metadata={"hnsw:space": "cosine"}
    )

    # ── Step 3: Embed and store in batches ───────────────────────────────
    openai_client = OpenAI()
    stored_count = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Embedding and storing chunks...",
            total=len(all_chunks)
        )

        # Process in batches
        for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch = all_chunks[i : i + EMBED_BATCH_SIZE]

            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            # Generate embeddings for this batch
            embeddings = _embed_batch(texts, openai_client)

            # Store in ChromaDB
            # LEARNING: ChromaDB's add() takes parallel lists:
            # ids, embeddings, documents (the text), and metadatas
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            stored_count += len(batch)
            progress.advance(task, len(batch))

    # ── Step 4: Update chunk_ids in source documents ──────────────────────
    # Build a map from source_url → chunk_ids for citation tracking
    url_to_chunks: dict[str, list[str]] = {}
    for chunk in all_chunks:
        url = chunk["metadata"]["source_url"]
        url_to_chunks.setdefault(url, []).append(chunk["id"])

    # Update sources with their chunk IDs
    updated_sources = []
    for source in sources:
        updated = dict(source)
        updated["chunk_ids"] = url_to_chunks.get(source["url"], [])
        updated_sources.append(updated)

    console.print(
        f"[green]✓[/green] Knowledge base ready: "
        f"[bold]{stored_count} chunks[/bold] stored in "
        f"collection '[cyan]{collection_name}[/cyan]'"
    )

    return {
        "sources": [],  # operator.add with empty list = no change (sources already in state)
        "vector_db_ready": True,
        "vector_db_collection_name": collection_name,
        "messages": [
            f"[embedder] {stored_count} chunks embedded and stored in ChromaDB "
            f"collection '{collection_name}'."
        ]
    }
