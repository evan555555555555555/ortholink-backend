"""
RAA — Regulatory Alert Agent tests.
test_change_detection: Modified regulation detected, old chunk archived with is_active=False,
new chunk embedded, alert generated.
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from app.agents.raa_agent import run_raa_for_document, AlertEvent, ChangeSummary, CONFIDENCE_THRESHOLD
from app.ingestion.chunker import Chunk
from app.services.alert_store import clear_for_tests, get_alerts
from app.tools.alert_tools import content_hash
from app.tools.vector_store import ChunkMetadata, VectorStore, get_vector_store


@pytest.fixture
def temp_store():
    """Vector store with one chunk for a document."""
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(index_path=tmp)
        store._ensure_loaded()
        dim = store.dimension
        # One chunk for document UA-RES-753
        chunk_id = "raa-test-chunk-1"
        text = "Article 14. Manufacturers must maintain technical documentation."
        chunk_hash = content_hash(text)
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            country="UA",
            regulation_name="Resolution 753",
            article="14",
            text=text,
            is_active=True,
            chunk_hash=chunk_hash,
            document_id="UA-RES-753",
            source_url="https://example.com/ua-753",
        )
        # Add one vector so index is valid
        emb = np.random.randn(1, dim).astype(np.float32)
        store.add_chunks(emb, [meta])
        store.save()
        yield tmp, store, chunk_id, chunk_hash


def test_change_detection(temp_store):
    """Modified regulation detected: old chunk archived (is_active=False), new chunk embedded, alert generated."""
    tmp_path, store, old_chunk_id, old_hash = temp_store
    clear_for_tests()

    new_content = "Article 14. Manufacturers must maintain technical documentation and submit annual updates."
    new_hash = content_hash(new_content)
    assert new_hash != old_hash

    def fake_scrape(url):
        return (new_content, new_hash, True)

    with patch("app.agents.raa_agent.get_vector_store") as m_get_store:
        m_get_store.return_value = VectorStore(index_path=tmp_path)

        import app.agents.raa_agent as raa_mod
        with patch.object(
            raa_mod,
            "_summarize_change",
            return_value=ChangeSummary(
                summary="Annual submission requirement added.",
                confidence=0.85,
            ),
        ):
            with patch("app.ingestion.embedder.embed_batch") as m_embed:
                def fake_embed(texts, batch_size=100):
                    n = len(texts)
                    dim = 3072
                    return np.random.randn(n, dim).astype(np.float32)

                m_embed.side_effect = fake_embed

                event = run_raa_for_document(
                    country="UA",
                    document_id="UA-RES-753",
                    source_url="https://example.com/ua-753",
                    regulation_name="Resolution 753",
                    _scrape_fn=fake_scrape,
                )

    assert event is not None
    assert isinstance(event, AlertEvent)
    assert event.country == "UA"
    assert event.document_id == "UA-RES-753"
    assert event.change_summary == "Annual submission requirement added."
    assert old_chunk_id in event.old_chunk_ids
    assert len(event.new_chunk_ids) >= 1
    assert "Reference tool only" in event.disclaimer

    # Reload store from disk and check old chunk is inactive
    store2 = VectorStore(index_path=tmp_path)
    store2._ensure_loaded()
    old_meta = next((m for m in store2.metadata if m.chunk_id == old_chunk_id), None)
    assert old_meta is not None
    assert old_meta.is_active is False

    # New chunks present and active
    new_ids = set(event.new_chunk_ids)
    active_new = [m for m in store2.metadata if m.chunk_id in new_ids]
    assert len(active_new) >= 1
    assert all(m.is_active for m in active_new)

    # Alert stored
    alerts = get_alerts(limit=10)
    assert len(alerts) >= 1
    assert any(a.get("document_id") == "UA-RES-753" for a in alerts)
