"""
OrthoLink ROA Routes
POST /api/v1/generate-checklist — Role-split checklist with legal citations.
Async with job_id; PDF export via separate endpoint (WeasyPrint) when implemented.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException

from app.crews.generate_checklist import RoleSplitChecklist, run_roa_checklist
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate-checklist", tags=["ROA"])


@router.post("")
async def generate_checklist(
    country: str = Form(..., description="Country code (e.g. US, UA, IN)"),
    device_class: str = Form(..., description="Device class (e.g. II, IIb)"),
    device_type: Optional[str] = Form(None),
    user: AuthenticatedUser = Depends(require_reviewer),
) -> RoleSplitChecklist:
    """
    Regulatory Operations Agent — role-split compliance checklist.
    Manufacturer ∩ Importer = ∅; includes QMSR for US, UDI per country.
    Returns job_id + items; PDF export via GET /api/v1/export-pdf when available.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    # run_roa_checklist is synchronous (blocks on LLM calls); offload to thread
    # so the FastAPI event loop is never blocked (PRD: async throughout).
    result = await asyncio.to_thread(
        run_roa_checklist,
        country=country,
        device_class=device_class,
        device_type=device_type,
    )
    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="roa")

    # ── Integrity: sign + inline FAISS fact-check ─────────────────────────────
    try:
        from app.services.integrity_guard import auto_verify_result
        from app.services.crypto_signer import sign_payload
        from fastapi.responses import JSONResponse

        from app.services.job_store import _DISCLAIMER
        result_dict = result.model_dump()
        result_dict["_disclaimer"] = _DISCLAIMER
        igr = auto_verify_result(result_dict, country=country, device_class=device_class)
        if igr:
            result_dict["_integrity"] = igr
        signed = sign_payload(result_dict)
        return JSONResponse(content=signed)
    except Exception as sign_exc:
        logger.warning("CRITICAL: ROA crypto-sign/integrity failed: %s — returning unsigned", sign_exc)

    return result
