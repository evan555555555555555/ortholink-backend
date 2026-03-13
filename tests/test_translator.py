"""
Tests for app.ingestion.translator (chunking and translation).
"""

import pytest


def test_translator_chunks_large_text():
    """Large text is split into chunks under 6,000 words each."""
    from app.ingestion.translator import _split_into_translation_chunks

    # Create fake text with ~20,000 words
    big_text = ("Стаття 1. Вимоги до виробника. " * 2000) + (
        "\n\nСтаття 2. Технічний файл. " * 2000
    )

    chunks = _split_into_translation_chunks(big_text, max_words=6000)

    assert len(chunks) > 1, "Large text must be split into multiple chunks"
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 6500, f"Chunk too large: {word_count} words"
