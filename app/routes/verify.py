"""
OrthoLink Verify Route — Truth Checker / Claim Verifier
POST /api/v1/verify-claims

Validates regulatory claims against the FAISS vector store.
No LLM calls — pure semantic similarity scoring for maximum reliability.
"""

import logging
import re

from fastapi import APIRouter, Depends, Form, HTTPException

from app.crews.verify_claims import VerificationReport, run_claim_verification
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verify-claims", tags=["Verify"])


@router.post("")
async def verify_claims(
    claims_text: str = Form(
        ...,
        description=(
            "Newline-separated list of regulatory claims to verify. "
            "Each line is one claim. Max 20 claims."
        ),
    ),
    country: str = Form(..., description="Country code for regulatory context"),
    device_class: str = Form(..., description="Device class"),
    user: AuthenticatedUser = Depends(require_reviewer),
) -> VerificationReport:
    """
    Claim Verifier — the reliable truth checker.

    Takes a list of regulatory claims (newline-separated) and verifies each
    against the FAISS regulatory database using semantic similarity scoring.

    Verdict scale:
    - VERIFIED (score ≥ 0.62): Strong regulatory backing found
    - PARTIALLY_VERIFIED (0.45-0.62): Related text found, exact claim may differ
    - UNVERIFIED (< 0.45): No regulatory backing found
    - CONTRADICTED: Found text that directly contradicts the claim

    This endpoint is synchronous (fast, no LLM) — results in ~2-5 seconds.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    # Split by newlines first; if only one chunk remains, try sentence-splitting
    raw_lines = [c.strip() for c in claims_text.splitlines() if c.strip()]
    claims: list[str] = []
    for line in raw_lines:
        # If the line contains multiple sentences (period+space+uppercase), split them
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        for sent in sentences:
            s = sent.strip().rstrip(".")
            if len(s) >= 10:  # Ignore fragments shorter than 10 chars
                claims.append(sent.strip())
    if not claims:
        raise HTTPException(status_code=422, detail="At least one claim is required.")

    country_code = country.strip().upper()

    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="verify")

    return run_claim_verification(
        claims=claims,
        country=country_code,
        device_class=device_class,
    )
