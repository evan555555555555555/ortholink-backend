"""
Tests for the regulatory text chunker.
"""


from app.ingestion.chunker import Chunk, chunk_regulatory_text


class TestChunkRegulatory:
    """Test hierarchy-preserving chunker."""

    def test_basic_article_splitting(self):
        text = """
Article 1: Scope
This regulation applies to medical devices.
All manufacturers must comply.

Article 2: Definitions
Medical device means any instrument.
Device class refers to risk classification.

Article 3: Requirements
All devices must meet safety standards.
Documentation must be provided.
"""
        chunks = chunk_regulatory_text(
            text=text,
            country="EU",
            regulation_name="Test Regulation",
        )
        assert len(chunks) > 0
        # Should find articles
        article_chunks = [c for c in chunks if c.clause is None]
        assert len(article_chunks) >= 2

    def test_country_metadata(self):
        text = "Article 1: Test\nSome regulatory text here for testing purposes."
        chunks = chunk_regulatory_text(
            text=text,
            country="UA",
            regulation_name="Resolution 753",
        )
        for chunk in chunks:
            assert chunk.country == "UA"
            assert chunk.regulation_name == "Resolution 753"

    def test_device_classes_propagated(self):
        text = "Article 1: Test\nSome regulatory text here for testing purposes."
        chunks = chunk_regulatory_text(
            text=text,
            country="US",
            regulation_name="21 CFR 820",
            device_classes=["II", "III"],
        )
        for chunk in chunks:
            assert chunk.device_classes == ["II", "III"]

    def test_empty_text(self):
        chunks = chunk_regulatory_text(
            text="",
            country="US",
            regulation_name="Test",
        )
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_no_articles_single_chunk(self):
        text = "This is plain text without any article or section markers. " * 10
        chunks = chunk_regulatory_text(
            text=text,
            country="US",
            regulation_name="Test",
        )
        # Should create at least one chunk for the full text
        assert len(chunks) >= 1
        assert chunks[0].article == "Full Text"

    def test_chunk_has_hash(self):
        text = "Article 1: Test\nContent here for the article."
        chunks = chunk_regulatory_text(
            text=text,
            country="US",
            regulation_name="Test",
        )
        for chunk in chunks:
            assert chunk.chunk_hash
            assert len(chunk.chunk_hash) == 64  # SHA-256 hex


class TestChunkDataclass:
    """Test the Chunk dataclass."""

    def test_chunk_creation(self):
        chunk = Chunk(
            chunk_id="test-1",
            text="Some text",
            parent_text="Full parent text",
            country="US",
            regulation_name="21 CFR 820",
            article="Article 1",
        )
        assert chunk.chunk_id == "test-1"
        assert chunk.country == "US"

    def test_chunk_auto_hash(self):
        chunk = Chunk(
            chunk_id="test-1",
            text="Some text",
            parent_text="Full parent text",
            country="US",
            regulation_name="Test",
            article="Art 1",
        )
        assert chunk.chunk_hash  # Auto-generated

    def test_chunk_default_fields(self):
        chunk = Chunk(
            chunk_id="test-1",
            text="Text",
            parent_text="Parent",
            country="US",
            regulation_name="Test",
            article="Art 1",
        )
        assert chunk.clause is None
        assert chunk.device_classes == []
        assert chunk.language == "en"
