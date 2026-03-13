"""
OrthoLink DVA Routes
POST /api/v1/verify-distributor

Sync: returns GapAnalysisReport. Async: ?async=1 returns 202 + job_id; poll GET /api/v1/jobs/{job_id}.
"""

import logging
from typing import Optional

import json

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.crews.verify_distributor import run_dva_analysis, run_dva_analysis_stream
from app.middleware.auth import AuthenticatedUser, get_current_user
from app.middleware.rbac import require_reviewer
from app.services.audit_logger import get_audit_logger
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verify-distributor", tags=["DVA"])


async def _run_dva_background(
    job_id: str,
    csv_content: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    org_id: str,
    user_id: str,
):
    """Background task: run DVA and store result so client can poll."""
    set_running(job_id)
    try:
        report = await run_dva_analysis(
            csv_content=csv_content,
            country=country,
            device_class=device_class,
            device_type=device_type,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="dva")
        get_audit_logger().log_dva_analysis(
            org_id=org_id,
            user_id=user_id,
            analysis_id=report.analysis_id,
            country=country,
            device_class=device_class,
            item_count=report.summary.total_submitted,
            fraud_risk_score=report.summary.fraud_risk_score,
        )
        set_completed(job_id, report.model_dump())
    except Exception as e:
        logger.error(f"DVA background job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def verify_distributor(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file with distributor items (.csv, .pdf, .xlsx, .xls, .txt)"),
    country: str = Form(..., description="Target country code (e.g., 'UA', 'US', 'IN')"),
    device_class: Optional[str] = Form(None, description="Device class (e.g., 'IIb', 'CLASS_C'). Omit for India to match index."),
    device_type: Optional[str] = Form(None, description="Device type (optional)"),
    async_mode: bool = Form(False, description="If true, return 202 + job_id; poll GET /api/v1/jobs/{job_id}"),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Distributor Verification Agent (DVA) — "The Sentinel"

    Upload a CSV of distributor-claimed document requirements.
    DVA cross-references against actual regulatory requirements
    and classifies each as REQUIRED / EXTRA / MISSING / OPTIONAL.

    Sync (default): Returns GapAnalysisReport.
    Async (?async_mode=1): Returns 202 with job_id; poll GET /api/v1/jobs/{job_id} for result.
    """
    # Validate file format
    SUPPORTED_EXTENSIONS = (".csv", ".pdf", ".xlsx", ".xls", ".txt")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    file_ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if f".{file_ext}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Accepted formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Check usage limits
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Free trial limit reached ({usage['limit']} analyses). "
                "Please upgrade to continue using OrthoLink."
            ),
        )

    # Read file content
    try:
        content_bytes = await file.read()
        if len(content_bytes) > 10_485_760:  # 10 MB hard cap
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum upload size is 10 MB.",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Parse document: CSV passes through directly; other formats are converted to CSV text
    if file_ext == "csv":
        try:
            csv_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File encoding error. Please upload a UTF-8 encoded CSV file.",
            )
        if not csv_content.strip():
            raise HTTPException(status_code=400, detail="CSV file is empty.")
    else:
        from app.tools.document_parser import parse_document, content_to_csv_text
        try:
            items = parse_document(content_bytes, file.filename)
            csv_content = content_to_csv_text(items)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    device_class_val = device_class or ""  # Omit filter when not provided (e.g. India CLASS_C vs index I/II/IIb/III)

    if async_mode:
        job_id = create_job(agent="dva")
        background_tasks.add_task(
            _run_dva_background,
            job_id,
            csv_content,
            country,
            device_class_val,
            device_type,
            user.org_id or "",
            user.user_id,
        )
        return {"job_id": job_id, "status": "pending", "message": "Poll GET /api/v1/jobs/" + job_id}

    # Sync: run in request (for single-worker/low concurrency; use async_mode under load)
    try:
        report = await run_dva_analysis(
            csv_content=csv_content,
            country=country,
            device_class=device_class_val,
            device_type=device_type,
        )
    except ValueError as e:
        logger.warning("DVA validation error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid input. Check your CSV format and country code.")
    except Exception as e:
        logger.error(f"DVA analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again or contact support.",
        )

    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="dva")
    audit = get_audit_logger()
    audit.log_dva_analysis(
        org_id=user.org_id or "",
        user_id=user.user_id,
        analysis_id=report.analysis_id,
        country=country,
        device_class=device_class_val,
        item_count=report.summary.total_submitted,
        fraud_risk_score=report.summary.fraud_risk_score,
    )

    # ── Integrity: sign + inline FAISS fact-check ─────────────────────────────
    try:
        from app.services.integrity_guard import auto_verify_result
        from app.services.crypto_signer import sign_payload
        from fastapi.responses import JSONResponse

        from app.services.job_store import _DISCLAIMER
        report_dict = report.model_dump()
        report_dict["_disclaimer"] = _DISCLAIMER
        igr = auto_verify_result(report_dict, country=country, device_class=device_class_val)
        if igr:
            report_dict["_integrity"] = igr
        signed = sign_payload(report_dict)
        return JSONResponse(content=signed)
    except Exception as sign_exc:
        logger.warning("CRITICAL: DVA crypto-sign/integrity failed: %s — returning unsigned", sign_exc)

    return report


@router.post("/stream")
async def verify_distributor_stream(
    file: UploadFile = File(..., description="Document file (.csv, .pdf, .xlsx, .xls, .txt)"),
    country: str = Form(..., description="Target country code (e.g., 'UA', 'US')"),
    device_class: Optional[str] = Form(None, description="Device class (e.g., 'IIb')"),
    device_type: Optional[str] = Form(None),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    DVA Streaming — Server-Sent Events with live thinking status.

    Emits SSE events:
      data: {"status": "classifying", "detail": "DVA Agent analyzing 'Technical File'..."}
      data: {"item": {...classified item...}}
      data: {"missing": {...missing requirement...}}
      data: {"done": true, "report": {...full report...}}
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    try:
        content_bytes = await file.read()
        if len(content_bytes) > 10_485_760:
            raise HTTPException(status_code=400, detail="File too large. Maximum 10 MB.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read file.")

    file_ext = file.filename.lower().rsplit(".", 1)[-1] if file.filename and "." in file.filename else ""
    SUPPORTED_EXTENSIONS = (".csv", ".pdf", ".xlsx", ".xls", ".txt")
    if f".{file_ext}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}")

    if file_ext == "csv":
        try:
            csv_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding error. Please upload UTF-8 CSV.")
    else:
        from app.tools.document_parser import parse_document, content_to_csv_text
        try:
            items = parse_document(content_bytes, file.filename)
            csv_content = content_to_csv_text(items)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    device_class_val = device_class or ""

    stream = run_dva_analysis_stream(
        csv_content=csv_content,
        country=country,
        device_class=device_class_val,
        device_type=device_type,
    )

    async def event_stream():
        async for event in stream:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="dva")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/countries")
async def get_supported_countries(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get list of countries available in the vector store."""
    from app.tools.vector_store import get_vector_store

    store = get_vector_store()
    countries = store.get_countries()

    return {
        "countries": sorted(countries),
        "count": len(countries),
    }
