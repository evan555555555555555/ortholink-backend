"""
M1: Country isolation — Ukraine query returns 0 FDA results and vice versa.
Tests actual FAISS metadata filter (country + is_active), not mocks.
"""

import json
import os
import tempfile
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from app.tools.vector_store import ChunkMetadata, VectorStore


@pytest.fixture
def temp_index_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def vector_store_ua_and_us(temp_index_path):
    """Build a minimal FAISS index with UA and US chunks; metadata filter is enforced in search."""
    dim = 3072
    index = faiss.IndexFlatIP(dim)

    # Two vectors (same norm so score doesn't matter; we care about metadata filter)
    v1 = np.random.randn(dim).astype(np.float32)
    v2 = np.random.randn(dim).astype(np.float32)
    faiss.normalize_L2(v1.reshape(1, -1))
    faiss.normalize_L2(v2.reshape(1, -1))
    index.add(np.vstack([v1, v2]))

    meta_ua = ChunkMetadata(
        chunk_id="ua-1",
        country="UA",
        regulation_name="Resolution 753",
        article="14",
        text="Ukraine requirement text.",
        device_classes=["IIb", "III"],
    )
    meta_us = ChunkMetadata(
        chunk_id="us-1",
        country="US",
        regulation_name="21 CFR 801",
        article="801",
        text="FDA requirement text.",
        device_classes=["II", "III"],
    )
    metadata = [meta_ua, meta_us]

    path = temp_index_path
    faiss.write_index(index, os.path.join(path, "faiss.index"))
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump([m.to_dict() for m in metadata], f)

    store = VectorStore(index_path=path)
    store._ensure_loaded()
    return store


def test_country_isolation_ukraine_returns_zero_fda_results(vector_store_ua_and_us):
    """Query with country=UA must return only UA chunks; 0 US/FDA chunks."""
    store = vector_store_ua_and_us
    # Query embedding can be anything; we mock to avoid OpenAI call
    query_vec = np.random.randn(3072).astype(np.float32)
    faiss.normalize_L2(query_vec.reshape(1, -1))

    with patch("app.tools.vector_store.embed_text", return_value=query_vec):
        results = store.search(
            query="registration requirements",
            country="UA",
            device_class="IIb",
            top_k=10,
            active_only=True,
        )

    assert len(results) <= 2
    for r in results:
        assert r["country"] == "UA"
    # No FDA/US result when querying Ukraine
    assert not any(r["country"] == "US" for r in results)


def test_country_isolation_us_returns_zero_ukraine_results(vector_store_ua_and_us):
    """Query with country=US must return only US chunks; 0 Ukraine chunks."""
    store = vector_store_ua_and_us
    query_vec = np.random.randn(3072).astype(np.float32)
    faiss.normalize_L2(query_vec.reshape(1, -1))

    with patch("app.tools.vector_store.embed_text", return_value=query_vec):
        results = store.search(
            query="device listing",
            country="US",
            device_class="II",
            top_k=10,
            active_only=True,
        )

    for r in results:
        assert r["country"] == "US"
    assert not any(r["country"] == "UA" for r in results)


def test_get_baseline_chunks_respects_country(vector_store_ua_and_us):
    """get_baseline_chunks returns only chunks for the requested country."""
    store = vector_store_ua_and_us
    ua_baseline = store.get_baseline_chunks(country="UA", active_only=True)
    us_baseline = store.get_baseline_chunks(country="US", active_only=True)
    assert all(c["country"] == "UA" for c in ua_baseline)
    assert all(c["country"] == "US" for c in us_baseline)
    assert len(ua_baseline) == 1 and ua_baseline[0]["regulation_name"] == "Resolution 753"
    assert len(us_baseline) == 1 and us_baseline[0]["regulation_name"] == "21 CFR 801"
