"""
OrthoLink CRA Routes
POST /api/v1/review-document — Compliance Review Agent (SSE streaming).
Returns job_id immediately; streams AIComment[] with severity + citations.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.crews.review_document import run_cra_review_stream
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/review-document", tags=["CRA"])


@router.post("")
async def review_document(
    file: UploadFile = File(..., description="Document to review (PDF, DOCX, TXT)"),
    standard: str = Form(..., description="Standard (e.g. FDA 21 CFR 820, ISO 13485)"),
    country: str = Form(..., description="Country code (e.g. UA, US)"),
    device_class: Optional[str] = Form(None),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Compliance Review Agent — RAG-grounded document review.
    Returns job_id; streams SSE events with AIComment (clause, severity, citation, suggestion).
    Confidence < 0.7 → structured refusal (no stream).
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    if len(content) > 10_485_760:  # 10 MB hard cap
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum upload size is 10 MB.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    stream = run_cra_review_stream(
        file_content=content,
        filename=file.filename or "document",
        standard=standard,
        country=country,
        device_class=device_class or "",
    )

    first_event = None
    async for event in stream:
        first_event = event
        break

    if first_event is None:
        raise HTTPException(status_code=500, detail="CRA stream produced no event.")

    if first_event.get("refused") and first_event.get("refusal_reason"):
        return {
            "job_id": first_event.get("job_id", ""),
            "refused": True,
            "refusal_reason": first_event["refusal_reason"],
            "disclaimer": first_event.get("disclaimer", ""),
        }

    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="cra")

    async def event_stream():
        yield f"data: {json.dumps(first_event)}\n\n"
        async for event in stream:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
