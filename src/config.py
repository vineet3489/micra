"""
MICRA — Configuration
======================

All tuneable constants live here.

LEARNING: Why centralize config?

When you're learning RAG, you'll want to experiment with these values:
  - Does chunk_size=256 give better retrieval than chunk_size=512?
  - Does top_k=10 give better framework analysis than top_k=5?

If these numbers are scattered across 10 files, changing them is a pain.
Central config = one place to tune, experiment, and understand trade-offs.
"""

# ── Chunking ───────────────────────────────────────────────────────────────
# LEARNING: Chunk size is one of the most impactful RAG parameters.
#
# Too small (e.g. 128 tokens):
#   + Very precise retrieval — matches narrow topics well
#   - Loses context — a sentence about pricing might be cut off from
#     the surrounding paragraph that explains what it's pricing
#
# Too large (e.g. 2048 tokens):
#   + Rich context per chunk
#   - Noisy retrieval — you retrieve a lot of irrelevant content alongside
#     the relevant part
#   - Hits embedding model limits faster
#
# 512 tokens with 64-token overlap is a good starting point.
# "Overlap" means adjacent chunks share 64 tokens — this prevents key
# sentences at chunk boundaries from being split across two chunks.

CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 64        # tokens shared between adjacent chunks

# ── Retrieval ──────────────────────────────────────────────────────────────
# LEARNING: top_k controls how many chunks are retrieved per query.
#
# Each framework agent retrieves TOP_K chunks from the vector DB before
# reasoning. More chunks = more context, but also more noise and higher
# token cost (all retrieved chunks go into the LLM prompt).
#
# For strategic frameworks (Porter's 5 Forces, SWOT), 8 chunks is enough
# to cover the topic without flooding the context window.

TOP_K = 8                 # chunks retrieved per query

# ── Embedding model ────────────────────────────────────────────────────────
# LEARNING: text-embedding-3-small vs text-embedding-3-large
#
# small: 1536 dimensions, ~$0.02 per million tokens — good enough for most
# large: 3072 dimensions, ~$0.13 per million tokens — marginal improvement
#
# For a dev/learning project, small is the right choice.
# In production with millions of documents, the cost difference matters.

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0       # deterministic for structured tasks
LLM_TEMPERATURE_CREATIVE = 0.3  # slightly creative for prose sections

# ── Scraping ───────────────────────────────────────────────────────────────
SCRAPER_TIMEOUT = 15       # seconds per URL
SCRAPER_MAX_URLS = 15      # cap to control cost and time
SCRAPER_DELAY = 1.0        # seconds between requests (be polite)

# Max characters of raw text to keep per scraped page.
# Beyond this, we're usually in boilerplate / footer territory.
SCRAPER_MAX_CHARS = 15_000

# ── Search ─────────────────────────────────────────────────────────────────
SEARCH_RESULTS_PER_QUERY = 5   # DuckDuckGo results per search query
PAPER_RESULTS_PER_QUERY = 5    # Semantic Scholar results per query

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "outputs/chroma"   # relative to project root

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_PASS_THRESHOLD = 0.75     # minimum overall score to pass
