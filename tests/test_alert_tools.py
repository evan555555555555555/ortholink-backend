"""
RAA: soft_deactivate must set is_active=False, never DELETE chunks.
"""

import tempfile
from unittest.mock import patch

import pytest

from app.tools.alert_tools import soft_deactivate_chunks
from app.tools.vector_store import ChunkMetadata, VectorStore


def test_soft_deactivate_never_deletes():
    """RAA: soft_deactivate must set is_active=False, never DELETE chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(index_path=tmp)
        store._ensure_loaded()

        # Add two chunks to metadata (no need to add vectors for this test)
        c1 = ChunkMetadata(
            chunk_id="test-chunk-id-1",
            country="UA",
            regulation_name="Res 753",
            article="14",
            text="Requirement A",
            is_active=True,
        )
        c2 = ChunkMetadata(
            chunk_id="test-chunk-id-2",
            country="UA",
            regulation_name="Res 753",
            article="15",
            text="Requirement B",
            is_active=True,
        )
        store.metadata.append(c1)
        store.metadata.append(c2)
        initial_count = len(store.metadata)

        soft_deactivate_chunks(store, ["test-chunk-id-1"])

        # No deletion: metadata count unchanged
        assert len(store.metadata) == initial_count == 2

        # Targeted chunk is marked inactive, not removed
        m1 = next(m for m in store.metadata if m.chunk_id == "test-chunk-id-1")
        assert m1.is_active is False
        assert m1.valid_to is not None

        # Other chunk unchanged
        m2 = next(m for m in store.metadata if m.chunk_id == "test-chunk-id-2")
        assert m2.is_active is True
