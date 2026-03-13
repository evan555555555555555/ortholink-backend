"""
Integrity endpoints — tamper-proof audit trail for all agent responses.

POST /api/v1/integrity/verify-signature  — verify HMAC-SHA256 signature
GET  /api/v1/integrity/status            — check IntegrityGuard system status
POST /api/v1/integrity/check             — on-demand fact-check any text against FAISS
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integrity", tags=["Integrity"])


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class VerifySignatureBody(BaseModel):
    payload: dict[str, Any] = Field(
        ...,
        description="Complete agent response payload including the _signed block.",
    )


class FactCheckBody(BaseModel):
    text: str = Field(..., min_length=10, description="Regulatory text to fact-check.")
    country: Optional[str] = Field(default="", description="Country code (e.g. US, EU, UK).")
    device_class: Optional[str] = Field(default="", description="Device class (e.g. II, IIb, III).")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/verify-signature")
def verify_signature_endpoint(
    body: VerifySignatureBody,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Verify the cryptographic signature on a signed agent response.

    Pass the complete response payload (including the `_signed` block).
    Returns validity status, tamper-detection result, and payload age.

    OrthoLink signs every agent response with HMAC-SHA256 at completion time.
    Use this endpoint to prove that a regulatory analysis has not been altered
    since it was generated — critical for audit trails and regulatory submissions.
    """
    from app.services.crypto_signer import verify_signature
    return verify_signature(body.payload)


@router.get("/status")
def integrity_status(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Return IntegrityGuard system status and capabilities.

    IntegrityGuard runs automatically on every agent response:
    1. Cryptographic signing  (HMAC-SHA256, immediate)
    2. Background fact-check  (FAISS cosine similarity, ~1-3s after completion)
    """
    # Dynamic counts from vector store (avoid stale hardcoded values)
    chunk_count = 0
    country_count = 0
    try:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        store._ensure_loaded()
        chunk_count = store.get_chunk_count()
        country_count = len(store.get_countries())
    except Exception:
        pass  # graceful degradation — return zeros rather than crash

    return {
        "integrity_guard": "active",
        "crypto_signing": True,
        "auto_fact_check": True,
        "algorithm": "HMAC-SHA256",
        "faiss_backed": True,
        "regulatory_chunks": chunk_count,
        "countries_covered": country_count,
        "description": (
            f"Every OrthoLink agent response is automatically fact-checked against "
            f"{chunk_count:,} regulatory chunks from {country_count} countries, and "
            f"cryptographically signed with HMAC-SHA256. Tamper detection is available "
            f"via POST /verify-signature."
        ),
    }


@router.post("/check")
def on_demand_fact_check(
    body: FactCheckBody,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    On-demand fact-check any regulatory text against the FAISS vector store.

    Extracts regulatory claims from the provided text and verifies each one
    against official regulatory source documents. Returns an integrity report
    with claim-by-claim verdicts and source citations.

    No LLM calls — pure FAISS cosine similarity, sub-second response.
    """
    from app.services.integrity_guard import extract_regulatory_claims
    from app.crews.verify_claims import run_claim_verification

    claims = extract_regulatory_claims(body.text, max_claims=20)
    if not claims:
        return {
            "claims_checked": 0,
            "overall_verdict": "NO_CLAIMS",
            "overall_confidence": 1.0,
            "message": "No regulatory claims detected in the provided text.",
            "claim_results": [],
        }

    report = run_claim_verification(
        claims=claims,
        country=body.country or "US",
        device_class=body.device_class or "",
    )

    return {
        "claims_checked": report.total_claims,
        "overall_verdict": report.overall_verdict,
        "overall_confidence": report.overall_confidence,
        "verified": report.verified,
        "partially_verified": report.partially_verified,
        "unverified": report.unverified,
        "contradicted": report.contradicted,
        "claim_results": [
            {
                "claim": r.claim[:200],
                "verdict": r.verdict,
                "confidence": r.confidence,
                "citation": r.citation,
                "recommendation": r.recommendation,
            }
            for r in report.claim_results
        ],
        "disclaimer": report.disclaimer,
    }
