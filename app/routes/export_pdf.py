"""
OrthoLink PDF Export (PRD G12)
POST /api/v1/export-pdf — WeasyPrint PDF for checklists (and future report types).
POST /api/v1/export-pdf/from-checklist — Render PDF from existing checklist JSON (no re-run).
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.crews.generate_checklist import run_roa_checklist
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.tools.pdf_generator import render_checklist_pdf, render_risk_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export-pdf", tags=["Export"])


class ChecklistExportBody(BaseModel):
    """Request body for from-checklist PDF export."""

    country: str = Field(..., min_length=1)
    device_class: str = Field(..., min_length=1)
    items: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = Field(default="Reference tool only. Verify with official sources.")


@router.post("")
async def export_pdf(
    report_type: str = Form(..., description="checklist | strategy"),
    country: Optional[str] = Form(None, description="For checklist: country code"),
    device_class: Optional[str] = Form(None, description="For checklist: device class"),
    analysis_id: Optional[str] = Form(None, description="ID of an existing analysis to export"),
    user: AuthenticatedUser = Depends(require_reviewer),
) -> Response:
    """
    Export report as PDF. PRD §6.2: POST /api/v1/export-pdf
    - report_type=checklist: requires country and device_class.
    - report_type=strategy: renders strategy report from analysis_id or inline data.
    """
    rt = report_type.lower()

    if rt == "checklist":
        if not country or not device_class:
            raise HTTPException(
                status_code=400,
                detail="For checklist PDF, country and device_class are required.",
            )
        checklist = run_roa_checklist(country=country, device_class=device_class)
        pdf_bytes = render_checklist_pdf(
            country=checklist.country,
            device_class=checklist.device_class,
            items=[i.model_dump() for i in checklist.items],
            disclaimer=checklist.disclaimer,
        )
        filename = f"ortholink-checklist-{checklist.country}-{checklist.device_class}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if rt == "strategy":
        from app.tools.pdf_generator import render_strategy_pdf
        if analysis_id:
            try:
                from app.services.supabase_client import get_supabase_client
                client = get_supabase_client()
                result = client.table("analysis_history").select("*").eq("id", analysis_id).single().execute()
                data = result.data or {}
            except Exception as e:
                logger.warning(f"Failed to load analysis {analysis_id}: {e}")
                raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
        else:
            data = {}
        pdf_bytes = render_strategy_pdf(data)
        filename = f"ortholink-strategy-{analysis_id or 'report'}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(status_code=400, detail=f"Unknown report_type: {report_type}. Use 'checklist' or 'strategy'.")


@router.post("/from-checklist")
async def export_pdf_from_checklist(
    body: ChecklistExportBody,
    user: AuthenticatedUser = Depends(require_reviewer),
) -> Response:
    """
    Render PDF from an existing checklist (e.g. from ROA panel). No agent re-run.
    """
    items = [i if isinstance(i, dict) else {} for i in body.items]
    normalized = []
    for i in items:
        normalized.append({
            "item": i.get("item", ""),
            "role": i.get("role", "BOTH"),
            "regulation_cite": i.get("regulation_cite", ""),
            "deadline_days": int(i.get("deadline_days", 0)),
            "apostille_required": bool(i.get("apostille_required", False)),
            "notes": i.get("notes", ""),
        })
    pdf_bytes = render_checklist_pdf(
        country=body.country,
        device_class=body.device_class,
        items=normalized,
        disclaimer=body.disclaimer,
    )
    filename = f"ortholink-checklist-{body.country}-{body.device_class}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class RiskReportExportBody(BaseModel):
    """Request body for from-risk-report PDF export."""

    device_description: str
    intended_use: str
    country: str
    overall_verdict: str
    hazard_analysis: list[dict[str, Any]] = Field(default_factory=list)
    counts: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = Field(
        default="Reference tool only. Not a substitute for qualified risk management review."
    )


@router.post("/from-risk-report")
async def export_pdf_from_risk_report(
    body: RiskReportExportBody,
    user: AuthenticatedUser = Depends(require_reviewer),
) -> Response:
    """
    Render ISO 14971 RMA risk report to PDF (e.g. from RMA panel). No agent re-run.
    """
    report_data = {
        "device_description": body.device_description,
        "intended_use": body.intended_use,
        "country": body.country,
        "overall_verdict": body.overall_verdict,
        "hazard_analysis": [h if isinstance(h, dict) else {} for h in body.hazard_analysis],
        "counts": body.counts if isinstance(body.counts, dict) else {},
        "disclaimer": body.disclaimer,
    }
    pdf_bytes = render_risk_pdf(report_data)
    filename = f"ortholink-rma-{body.country}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
