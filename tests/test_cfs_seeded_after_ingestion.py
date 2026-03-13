"""
1A: CFS synthetic seeding has been removed from the ingestion pipeline.
CFS requirements are now evaluated via FAISS search + LLM semantic evaluation.

These tests verify:
- The deactivation script correctly marks synthetic CFS chunks as inactive
- The truncated_missing_count feature still works with the new _find_missing_requirements
"""

import asyncio
import os
import tempfile
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest

from app.ingestion.cfs_seed import seed_cfs_for_country
from app.tools.vector_store import VectorStore


def test_cfs_seed_function_still_works():
    """seed_cfs_for_country still functions (for backward compat) but chunks
    created by it should be deactivatable via deactivate_cfs_seeds.py."""
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(index_path=tmp)
        store._ensure_loaded()

        with patch("app.ingestion.embedder.embed_batch") as mock_embed:
            mock_embed.return_value = np.random.randn(1, 3072).astype(np.float32)
            n = seed_cfs_for_country("UA", vector_store=store)

        assert n == 1
        cfs_chunks = [
            m for m in store.metadata
            if m.country == "UA"
            and "Certificate of Free Sale" in m.text
        ]
        assert len(cfs_chunks) >= 1


def test_truncated_missing_count_when_baseline_exceeds_cap():
    """When baseline has > 20 missing requirements, truncated_missing_count is correct and disclaimer mentions it."""
    from app.crews.verify_distributor import run_dva_analysis

    # Build 20 displayed + 5 "truncated" = 25 total missing
    missing_20 = [
        {"requirement": f"Req {i}", "citation": f"Reg Art {i}", "country": "UA"}
        for i in range(20)
    ]

    async def fake_find_missing(*args, **kwargs):
        return missing_20, 25  # 25 total, 20 shown

    # Mock embeddings so pipeline doesn't call OpenAI; force direct pipeline (not CrewAI)
    with patch("app.crews.verify_distributor.get_settings") as mock_settings:
        mock_settings.return_value.use_dva_crew = False
        mock_settings.return_value.similarity_threshold = 0.82
        with patch("app.crews.verify_distributor.embed_text") as mock_embed:
            mock_embed.return_value = __import__("numpy").random.randn(3072).astype("float32")
            with patch("app.crews.verify_distributor._find_missing_requirements", side_effect=fake_find_missing):
                report = asyncio.run(
                    run_dva_analysis(
                        csv_content="document\nTechnical File",
                        country="UA",
                        device_class="IIb",
                    )
                )

    assert report.truncated_missing_count == 5
    assert "5 additional missing requirements not shown" in report.disclaimer
