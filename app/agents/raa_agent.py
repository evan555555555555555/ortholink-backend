"""
RAA — Regulatory Alert Agent.
Monitors government/regulatory source URLs for content changes; soft-deactivates
old chunks, re-embeds new content, notifies subscribed orgs. Never deletes chunks.
HC: text-embedding-3-large for re-embedding; gpt-4o for change summary; confidence gate ≥0.7.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import re_embed_chunks
from app.tools.alert_tools import scrape_and_hash
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Reference tool only. Regulatory change alerts do not constitute legal advice. "
    "Verify with official sources and consult a qualified regulatory affairs professional."
)

CONFIDENCE_THRESHOLD = 0.7


class ChangeSummary(BaseModel):
    """LLM output: what changed and confidence."""

    summary: str = Field(..., description="Brief summary of what changed in the regulation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the summary (0-1)")


class AlertEvent(BaseModel):
    """RAA alert event — stored and returned by GET /api/v1/alerts."""

    country: str
    document_id: str
    change_summary: str
    old_chunk_ids: list[str] = Field(default_factory=list)
    new_chunk_ids: list[str] = Field(default_factory=list)
    notified_orgs: list[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer: str = Field(default=DISCLAIMER)
    # Semantic drift: cosine distance between old and new document embeddings
    # 0.0 = semantically identical, 1.0 = completely different meaning
    drift_score: float = Field(default=0.0, description="Semantic drift 0.0–1.0")
    drift_label: str = Field(default="", description="CRITICAL | MAJOR | MINOR | TRIVIAL")


def _document_hash_from_chunks(chunks: list[dict]) -> str:
    """Compute a single hash representing the document from stored chunk hashes."""
    hashes = sorted((c.get("chunk_hash") or "") for c in chunks if c.get("chunk_hash"))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def _compute_semantic_drift(old_text: str, new_text: str) -> tuple[float, str]:
    """
    Compute semantic drift score between old and new regulatory document text.

    Uses cosine distance between mean embeddings of old and new content.
    Returns (drift_score, drift_label):
      drift_score : 0.0 = identical meaning, 1.0 = completely different
      drift_label : TRIVIAL | MINOR | MAJOR | CRITICAL

    This is layer-2 change detection beyond cryptographic hashing:
    - Hash changed but drift_score < 0.05 → typo/formatting fix (TRIVIAL)
    - drift_score 0.05–0.20 → clause updates (MINOR)
    - drift_score 0.20–0.50 → significant requirement changes (MAJOR)
    - drift_score > 0.50 → document substantially rewritten (CRITICAL)
    """
    try:
        import numpy as np
        from app.tools.embeddings import embed_text

        # Sample first 2000 chars of each (embedding API cap)
        old_vec = np.array(embed_text(old_text[:2000]), dtype=np.float32)
        new_vec = np.array(embed_text(new_text[:2000]), dtype=np.float32)

        # Cosine similarity → distance
        norm_old = np.linalg.norm(old_vec)
        norm_new = np.linalg.norm(new_vec)
        if norm_old == 0 or norm_new == 0:
            return 0.0, ""

        cosine_sim = float(np.dot(old_vec, new_vec) / (norm_old * norm_new))
        drift = round(max(0.0, 1.0 - cosine_sim), 4)

        if drift < 0.05:
            label = "TRIVIAL"
        elif drift < 0.20:
            label = "MINOR"
        elif drift < 0.50:
            label = "MAJOR"
        else:
            label = "CRITICAL"

        logger.debug(f"SemanticDrift: cosine_sim={cosine_sim:.4f} drift={drift:.4f} ({label})")
        return drift, label

    except Exception as e:
        logger.debug(f"SemanticDrift computation failed: {e}")
        return 0.0, ""


def _summarize_change(old_text: str, new_text: str) -> ChangeSummary:
    """Use gpt-4o to summarize what changed; returns summary and confidence."""
    from app.tools.llm import structured_completion

    system = (
        "You are a regulatory affairs analyst. Given old and new regulatory text, "
        "output a brief change_summary and a confidence score between 0 and 1. "
        "Be concise; one or two sentences for the summary."
    )
    user = (
        f"Old text (excerpt, first 3000 chars):\n{old_text[:3000]}\n\n"
        f"New text (excerpt, first 3000 chars):\n{new_text[:3000]}\n\n"
        "Provide summary and confidence."
    )
    return structured_completion(system, user, ChangeSummary, temperature=0.0)


def run_raa_for_document(
    country: str,
    document_id: str,
    source_url: str,
    regulation_name: str,
    _scrape_fn=None,
) -> Optional[AlertEvent]:
    """
    Run RAA pipeline for one document: scrape → hash → compare → soft-deactivate
    → re-embed → summarize → notify (if confidence ≥ 0.7). Never deletes chunks.
    Returns AlertEvent if change detected and alert emitted; None otherwise.
    """
    store = get_vector_store()
    chunks = store.get_chunks_by_document(
        country=country,
        document_id=document_id,
        source_url=source_url,
        active_only=True,
    )
    if not chunks:
        logger.debug(f"RAA: no stored chunks for {country} {document_id}; skip")
        return None

    do_scrape = _scrape_fn if _scrape_fn is not None else scrape_and_hash
    new_text, new_hash, success = do_scrape(source_url)
    if not success:
        logger.warning(f"RAA: scrape failed for {source_url}")
        return None

    stored_hash = _document_hash_from_chunks(chunks)
    if new_hash == stored_hash:
        logger.debug(f"RAA: no change for {country} {document_id}")
        return None

    old_chunk_ids = [c["chunk_id"] for c in chunks]
    new_chunks = chunk_regulatory_text(
        new_text,
        country=country,
        regulation_name=regulation_name,
        document_id=document_id,
        source_url=source_url,
    )
    if not new_chunks:
        logger.warning(f"RAA: chunking produced no chunks for {document_id}")
        return None

    re_embed_chunks(old_chunk_ids, new_chunks, store)
    new_chunk_ids = [c.chunk_id for c in new_chunks]

    old_text = "\n\n".join(c.get("text", "") for c in chunks)

    # Semantic drift: embedding cosine distance between old and new content
    drift_score, drift_label = _compute_semantic_drift(old_text, new_text)

    try:
        change_result = _summarize_change(old_text, new_text)
    except Exception as e:
        logger.exception(f"RAA: LLM summarization failed: {e}")
        change_result = ChangeSummary(summary="Regulation content changed; automated summary unavailable.", confidence=0.5)

    if change_result.confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"RAA: confidence {change_result.confidence} < {CONFIDENCE_THRESHOLD}; skip alert for {document_id}"
        )
        # Still record the event for audit but do not notify
        from app.services.alert_store import add_alert, get_subscribed_orgs
        from app.services.audit_logger import get_audit_logger

        notified_orgs: list[str] = []
        event = AlertEvent(
            country=country,
            document_id=document_id,
            change_summary=change_result.summary,
            old_chunk_ids=old_chunk_ids,
            new_chunk_ids=new_chunk_ids,
            notified_orgs=notified_orgs,
            drift_score=drift_score,
            drift_label=drift_label,
        )
        add_alert(event.model_dump())
        get_audit_logger().log(
            action="raa_alert_skipped_low_confidence",
            org_id="system",
            details={"country": country, "document_id": document_id, "confidence": change_result.confidence},
        )
        return event

    from app.services.alert_store import add_alert, get_subscribed_orgs
    from app.services.audit_logger import get_audit_logger
    from app.tools.alert_tools import notify_subscribers

    notified_orgs = get_subscribed_orgs(country)
    n_sent = notify_subscribers(
        country=country,
        regulation_id=document_id,
        change_type="update",
        severity="Major",
        summary=change_result.summary,
    )
    logger.info(f"RAA: notified {len(notified_orgs)} orgs, {n_sent} emails for {country} {document_id}")

    event = AlertEvent(
        country=country,
        document_id=document_id,
        change_summary=change_result.summary,
        old_chunk_ids=old_chunk_ids,
        new_chunk_ids=new_chunk_ids,
        notified_orgs=notified_orgs,
        drift_score=drift_score,
        drift_label=drift_label,
    )
    add_alert(event.model_dump())
    get_audit_logger().log(
        action="raa_alert",
        org_id="system",
        resource_type="alert",
        resource_id=document_id,
        details={
            "country": country,
            "document_id": document_id,
            "change_summary": change_result.summary,
            "old_chunk_count": len(old_chunk_ids),
            "new_chunk_count": len(new_chunk_ids),
            "notified_orgs": notified_orgs,
            "drift_score": drift_score,
            "drift_label": drift_label,
        },
    )
    return event


def run_raa_for_country(
    country: str,
    documents: list[dict],
) -> list[AlertEvent]:
    """
    Run RAA for all monitored documents in a country.
    documents: list of {document_id, source_url, regulation_name}.
    Cron: run every 24h per country.
    """
    events: list[AlertEvent] = []
    for doc in documents:
        try:
            ev = run_raa_for_document(
                country=country,
                document_id=doc["document_id"],
                source_url=doc["source_url"],
                regulation_name=doc.get("regulation_name", doc["document_id"]),
            )
            if ev:
                events.append(ev)
        except Exception as e:
            logger.exception(f"RAA: failed for {country} {doc.get('document_id')}: {e}")
    return events
