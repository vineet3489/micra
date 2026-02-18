"""
MICRA — Phase 2 Tests: RAG Pipeline
======================================

LEARNING: Testing RAG components

Testing the RAG pipeline has a few unique challenges:
  1. Scrapers make HTTP requests (mock them)
  2. Embedder calls OpenAI API (mock it)
  3. ChromaDB writes to disk (use in-memory client for tests)

The strategy: mock all external I/O, test the logic.

We test:
  A. Chunking logic (no mocks needed — pure text transformation)
  B. Scraper tool (mock httpx)
  C. Paper fetcher (mock httpx)
  D. Scraper node (mock tools)
  E. Embedder node (mock OpenAI + use in-memory ChromaDB)
  F. Retriever (use in-memory ChromaDB with pre-loaded data)
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.graph import create_initial_state
from src.state import SourceDocument


# ── A. Chunking Logic ──────────────────────────────────────────────────────
class TestChunking:
    """
    LEARNING: Chunking tests don't need any mocks — it's pure logic.
    This is why it's good to isolate chunking into its own function.
    """

    def test_long_text_produces_multiple_chunks(self):
        """A document longer than CHUNK_SIZE should be split."""
        from src.nodes.embedder import _chunk_document
        from src.config import CHUNK_SIZE

        # Create text longer than one chunk
        long_text = "This is a sentence about market intelligence. " * 100
        source: SourceDocument = {
            "url": "https://example.com",
            "source_type": "web",
            "title": "Test page",
            "raw_text": long_text,
            "chunk_ids": []
        }

        chunks = _chunk_document(source)

        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_short_text_produces_one_chunk(self):
        """A short document should stay as one chunk."""
        from src.nodes.embedder import _chunk_document

        source: SourceDocument = {
            "url": "https://example.com",
            "source_type": "web",
            "title": "Short page",
            "raw_text": "This is a short document.",
            "chunk_ids": []
        }

        chunks = _chunk_document(source)

        assert len(chunks) == 1

    def test_chunks_have_required_metadata(self):
        """Every chunk must have source metadata for citation tracking."""
        from src.nodes.embedder import _chunk_document

        source: SourceDocument = {
            "url": "https://test.com/page",
            "source_type": "academic",
            "title": "Test paper",
            "raw_text": "Some academic content here. " * 20,
            "chunk_ids": []
        }

        chunks = _chunk_document(source)

        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            meta = chunk["metadata"]
            assert meta["source_url"] == "https://test.com/page"
            assert meta["source_type"] == "academic"
            assert meta["title"] == "Test paper"
            assert "chunk_index" in meta

    def test_chunk_ids_are_unique(self):
        """Each chunk must have a unique ID (required by ChromaDB)."""
        from src.nodes.embedder import _chunk_document

        source: SourceDocument = {
            "url": "https://example.com",
            "source_type": "web",
            "title": "Test",
            "raw_text": "Text content. " * 200,
            "chunk_ids": []
        }

        chunks = _chunk_document(source)
        ids = [c["id"] for c in chunks]

        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_text_produces_no_chunks(self):
        """Empty or whitespace-only text should produce no chunks."""
        from src.nodes.embedder import _chunk_document

        source: SourceDocument = {
            "url": "https://example.com",
            "source_type": "web",
            "title": "Empty page",
            "raw_text": "   \n\n   ",
            "chunk_ids": []
        }

        chunks = _chunk_document(source)
        assert len(chunks) == 0


# ── B. Web Scraper Tool ────────────────────────────────────────────────────
class TestWebScraper:

    def test_successful_scrape_returns_clean_text(self):
        """Scraper should extract text and strip HTML tags."""
        from src.tools.web_scraper import scrape_url

        html_content = """
        <html>
        <head><title>DERMS Market Analysis</title></head>
        <body>
            <nav>Skip this nav content</nav>
            <main>
                <h1>DERMS Overview</h1>
                <p>Distributed Energy Resource Management Systems are critical for grid stability.</p>
                <p>The market is growing at 15% CAGR annually.</p>
            </main>
            <footer>Skip this footer</footer>
            <script>skip this script content</script>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.raise_for_status = MagicMock()

        with patch("src.tools.web_scraper.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            result = scrape_url("https://example.com/derms")

        assert result.success is True
        assert "DERMS" in result.text
        assert "grid stability" in result.text
        assert "<nav>" not in result.text    # HTML tags stripped
        assert "<script>" not in result.text  # script tags removed
        assert result.title == "DERMS Market Analysis"

    def test_timeout_returns_failure(self):
        """Timeout should return ScrapedPage with success=False."""
        import httpx
        from src.tools.web_scraper import scrape_url

        with patch("src.tools.web_scraper.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Timed out")

            result = scrape_url("https://slow-site.com")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_non_html_content_returns_failure(self):
        """PDFs and other non-HTML responses should be skipped."""
        from src.tools.web_scraper import scrape_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with patch("src.tools.web_scraper.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            result = scrape_url("https://example.com/report.pdf")

        assert result.success is False
        assert "Non-HTML" in result.error


# ── C. Paper Fetcher ───────────────────────────────────────────────────────
class TestPaperFetcher:

    def test_fetch_papers_parses_response(self):
        """Should parse Semantic Scholar API response correctly."""
        from src.tools.paper_fetcher import fetch_papers

        mock_api_response = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "DERMS Integration in Smart Grids",
                    "abstract": "This paper examines distributed energy resource management...",
                    "year": 2023,
                    "authors": [{"name": "Jane Smith"}, {"name": "Bob Jones"}],
                    "citationCount": 47,
                    "url": "https://www.semanticscholar.org/paper/abc123"
                },
                {
                    "paperId": "def456",
                    "title": "Paper with no abstract",
                    "abstract": None,  # should be filtered out
                    "year": 2022,
                    "authors": [],
                    "citationCount": 0,
                    "url": None
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = MagicMock()

        with patch("src.tools.paper_fetcher.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            papers = fetch_papers("DERMS smart grid")

        # Only the paper WITH an abstract should be returned
        assert len(papers) == 1
        assert papers[0].title == "DERMS Integration in Smart Grids"
        assert papers[0].year == 2023
        assert "Jane Smith" in papers[0].authors

    def test_paper_to_text_includes_metadata(self):
        """paper_to_text should include title, year, and abstract."""
        from src.tools.paper_fetcher import paper_to_text, AcademicPaper

        paper = AcademicPaper(
            title="Test Paper",
            abstract="This is the abstract content.",
            year=2023,
            authors=["Alice"],
            citation_count=10,
            url="https://example.com"
        )

        text = paper_to_text(paper)

        assert "Test Paper" in text
        assert "2023" in text
        assert "This is the abstract content." in text
        assert "[ACADEMIC PAPER]" in text  # label for LLM context


# ── D. Scraper Node ────────────────────────────────────────────────────────
class TestScraperNode:

    def test_scraper_node_returns_sources(self):
        """Scraper node should return a list of SourceDocuments."""
        from src.nodes.scraper import scraper_node

        state = create_initial_state("DERMS market")
        state["research_plan"] = {
            "source_types_to_query": ["web_competitors"],
            "search_queries": ["DERMS market leaders"],
            "competitor_names_to_research": []
        }

        mock_search_results = [
            MagicMock(url="https://competitor.com", source_type="web")
        ]
        mock_scraped_page = MagicMock(
            success=True,
            url="https://competitor.com",
            title="Competitor Page",
            text="DERMS product features and pricing details.",
            source_type="web"
        )

        with patch("src.nodes.scraper.discover_urls_for_plan", return_value=mock_search_results), \
             patch("src.nodes.scraper.scrape_urls", return_value=[mock_scraped_page]), \
             patch("src.nodes.scraper.fetch_papers_for_queries", return_value=[]):

            result = scraper_node(state)

        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "https://competitor.com"

    def test_scraper_node_handles_no_plan(self):
        """Scraper should log an error if no research plan exists."""
        from src.nodes.scraper import scraper_node

        state = create_initial_state("test query")
        # Don't set research_plan

        result = scraper_node(state)

        assert len(result.get("errors", [])) > 0
        assert any("No research plan" in e for e in result["errors"])


# ── E. Embedder Node ───────────────────────────────────────────────────────
class TestEmbedderNode:

    def test_embedder_stores_chunks_in_chromadb(self):
        """Embedder should chunk documents and store in ChromaDB."""
        from src.nodes.embedder import embedder_node

        state = create_initial_state("test")
        state["sources"] = [
            {
                "url": "https://example.com",
                "source_type": "web",
                "title": "Test page",
                "raw_text": "Market intelligence content about DERMS systems. " * 30,
                "chunk_ids": []
            }
        ]

        # Mock OpenAI embeddings — return fake 1536-dim vectors
        fake_embedding = [0.1] * 1536
        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=fake_embedding)]

        # Use in-memory ChromaDB for tests (no disk I/O)
        mock_chroma_collection = MagicMock()
        mock_chroma_collection.count.return_value = 0

        with patch("src.nodes.embedder._get_chroma_client") as mock_chroma_fn, \
             patch("src.nodes.embedder.OpenAI") as mock_openai_class:

            # Set up ChromaDB mock
            mock_chroma = MagicMock()
            mock_chroma_fn.return_value = mock_chroma
            mock_chroma.create_collection.return_value = mock_chroma_collection

            # Set up OpenAI mock
            mock_openai_instance = MagicMock()
            mock_openai_class.return_value = mock_openai_instance
            mock_openai_instance.embeddings.create.return_value = mock_embed_response

            result = embedder_node(state)

        assert result["vector_db_ready"] is True
        assert result["vector_db_collection_name"].startswith("micra_")
        assert mock_chroma_collection.add.called  # chunks were stored

    def test_embedder_handles_no_sources(self):
        """Embedder should log error if no sources provided."""
        from src.nodes.embedder import embedder_node

        state = create_initial_state("test")
        # No sources

        result = embedder_node(state)

        assert any("No sources" in e for e in result.get("errors", []))


# ── F. Retriever ───────────────────────────────────────────────────────────
class TestRetriever:

    def test_retrieve_returns_chunks_sorted_by_similarity(self):
        """Retrieved chunks should be sorted highest similarity first."""
        from src.retriever import Retriever

        # Mock ChromaDB and OpenAI
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["text about DERMS", "text about pricing", "unrelated text"]],
            "metadatas": [[
                {"source_url": "https://a.com", "source_type": "web", "title": "A", "chunk_index": 0},
                {"source_url": "https://b.com", "source_type": "web", "title": "B", "chunk_index": 0},
                {"source_url": "https://c.com", "source_type": "web", "title": "C", "chunk_index": 0},
            ]],
            "distances": [[0.1, 0.3, 0.8]]  # lower distance = more similar
        }

        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("src.retriever.chromadb.PersistentClient") as mock_chroma, \
             patch("src.retriever.OpenAI") as mock_openai_class:

            mock_chroma.return_value.get_collection.return_value = mock_collection
            mock_openai_instance = MagicMock()
            mock_openai_class.return_value = mock_openai_instance
            mock_openai_instance.embeddings.create.return_value = mock_embed_response

            retriever = Retriever("test_collection")
            chunks = retriever.retrieve("DERMS market analysis")

        assert len(chunks) == 3
        # First chunk should have highest similarity (lowest distance = 0.1)
        assert chunks[0].similarity > chunks[1].similarity > chunks[2].similarity

    def test_format_context_includes_source_url(self):
        """format_context should include source URLs for citation."""
        from src.retriever import Retriever, RetrievedChunk

        with patch("src.retriever.chromadb.PersistentClient"), \
             patch("src.retriever.OpenAI"):
            retriever = Retriever.__new__(Retriever)

        chunks = [
            RetrievedChunk(
                text="DERMS market is growing rapidly.",
                source_url="https://research.com/paper",
                source_type="academic",
                title="Market Paper",
                chunk_index=0,
                similarity=0.85,
                chunk_id="abc123"
            )
        ]

        context = retriever.format_context(chunks, include_source=True)

        assert "https://research.com/paper" in context
        assert "DERMS market is growing rapidly." in context
