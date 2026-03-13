"""
OrthoLink CRA Crew — Compliance Review Agent
POST /api/v1/review-document

RAG-grounded document review: extract clauses, retrieve regulation, compare via LLM.
Returns AIComment[] with severity + citations. Confidence < 0.7 → structured refusal.
"""

import asyncio
import logging
import re
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.core.anti_hallucination import check_confidence
from app.ingestion.scraper import load_from_file
from app.tools.llm import compare_clause_to_standard
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = "Reference tool only. Verify with official sources."


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class AIComment(BaseModel):
    """Single compliance finding — PRD CRA output."""

    clause: str = Field(..., description="Document clause excerpt")
    severity: str = Field(..., description="CRITICAL | MAJOR | MINOR | INFO")
    citation: str = Field(..., description="Regulation reference")
    suggestion: str = Field(default="", description="Remediation suggestion")
    standard_reference: str = Field(..., description="Standard used (e.g. FDA 21 CFR 820)")


class CRAReviewResult(BaseModel):
    """CRA endpoint result."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    comments: list[AIComment] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)
    refused: bool = Field(default=False, description="True if confidence < 0.7")
    refusal_reason: Optional[str] = None


def _extract_clauses(document_text: str, max_clauses: int = 12) -> list[str]:
    """Split document into reviewable clauses (no hardcoded classification)."""
    if not document_text or not document_text.strip():
        return []
    text = document_text.strip()
    parts = re.split(r"\n\s*\n+|\bArticle\s+\d+|\bSection\s+\d+", text, flags=re.IGNORECASE)
    clauses = [p.strip() for p in parts if len(p.strip()) > 40]
    if not clauses:
        # Single large block: split by sentence groups
        clauses = [text[:3000]] if len(text) > 3000 else [text]
    return clauses[:max_clauses]


def _compare_clause_sync(
    clause: str,
    standard: str,
    country: str,
    regulation_snippets: str,
) -> tuple[AIComment, float]:
    """Synchronous wrapper for compare_clause_to_standard (run in thread)."""
    out = compare_clause_to_standard(
        clause_text=clause,
        standard_reference=standard,
        country=country,
        regulation_snippets=regulation_snippets,
    )
    severity = out.get("severity", "INFO")
    if severity not in ("CRITICAL", "MAJOR", "MINOR", "INFO"):
        severity = "INFO"
    comment = AIComment(
        clause=out.get("clause", clause[:200]),
        severity=severity,
        citation=out.get("citation", ""),
        suggestion=out.get("suggestion", ""),
        standard_reference=out.get("standard_reference", standard),
    )
    conf = float(out.get("confidence", 0.8))
    return comment, conf


async def run_cra_review_stream(
    file_content: bytes,
    filename: str,
    standard: str,
    country: str,
    device_class: str = "",
):
    """
    Async generator: true per-clause SSE. Yields dicts for SSE payloads.
    First yield: either {"refused": True, "job_id", "refusal_reason", "disclaimer"} (early refusal)
    or {"job_id": "...", "started": True}. Then zero or more {"comment": AIComment.model_dump()},
    then {"done": True, "disclaimer": "..."} or {"refused": True, "refusal_reason": "...", "disclaimer": "..."}.
    """
    import os
    import tempfile
    from pathlib import Path

    job_id = str(uuid.uuid4())

    suffix = Path(filename).suffix if filename else ".txt"
    if not suffix or suffix == ".":
        suffix = ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_content)
        tmp.flush()
        try:
            result = load_from_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    if not result.success or not result.text.strip():
        yield {"refused": True, "job_id": job_id, "refusal_reason": "Could not extract text from document.", "disclaimer": DISCLAIMER}
        return

    doc_text = result.text.strip()
    if len(doc_text) < 100:
        yield {"refused": True, "job_id": job_id, "refusal_reason": "Document text too short for review.", "disclaimer": DISCLAIMER}
        return

    clauses = _extract_clauses(doc_text)
    if not clauses:
        yield {"refused": True, "job_id": job_id, "refusal_reason": "No reviewable clauses found.", "disclaimer": DISCLAIMER}
        return

    yield {"job_id": job_id, "started": True}

    store = get_vector_store()

    # Process clauses sequentially to stream live-thinking status events
    confidence_sum = 0.0
    count = 0

    for i, clause in enumerate(clauses):
        clause = clause[:2500] + "..." if len(clause) > 2500 else clause
        clause_preview = clause[:80].replace("\n", " ")
        yield {
            "status": "searching",
            "detail": f"CRA Agent searching {country} regulations for clause: '{clause_preview}...'",
            "progress": f"{i + 1}/{len(clauses)}",
        }
        try:
            reg_results = store.search(clause, country=country, device_class=device_class or None, top_k=10)
            regulation_snippets = "\n\n".join(
                (r.get("text", "")[:800] + f" [Source: {r.get('section_path') or r.get('document_id')}]")
                for r in reg_results
            )
            if not regulation_snippets.strip():
                continue

            top_reg = reg_results[0].get("regulation_name", "") if reg_results else ""
            yield {
                "status": "comparing",
                "detail": f"Comparing clause against {top_reg}...",
            }

            comment, conf = await asyncio.to_thread(
                _compare_clause_sync, clause, standard, country, regulation_snippets
            )
            confidence_sum += conf
            count += 1
            yield {"comment": comment.model_dump()}
        except Exception as e:
            logger.error("CRITICAL: CRA clause comparison failed for clause %d: %s", i, e)
            yield {"comment": {"clause": clause[:80], "status": "ERROR", "comment": f"Comparison failed: {str(e)[:100]}", "confidence": 0.0}}
            continue

    avg_confidence = confidence_sum / count if count else 0.0
    if not check_confidence(avg_confidence).passed:
        yield {"refused": True, "refusal_reason": f"Review confidence {avg_confidence:.2f} below threshold. Verify document and standard.", "disclaimer": DISCLAIMER}
        return

    yield {"done": True, "disclaimer": DISCLAIMER}


async def run_cra_review(
    file_content: bytes,
    filename: str,
    standard: str,
    country: str,
    device_class: str = "",
) -> CRAReviewResult:
    """
    Run CRA (batch): extract text, clauses, compare each. Returns full result.
    Used when a single CRAReviewResult is needed; for SSE use run_cra_review_stream.
    """
    comments: list[AIComment] = []
    job_id = str(uuid.uuid4())
    async for event in run_cra_review_stream(file_content, filename, standard, country, device_class):
        if event.get("refused") and "refusal_reason" in event:
            return CRAReviewResult(
                job_id=event.get("job_id", job_id),
                refused=True,
                refusal_reason=event["refusal_reason"],
                disclaimer=event.get("disclaimer", DISCLAIMER),
            )
        if "comment" in event:
            comments.append(AIComment.model_validate(event["comment"]))
    return CRAReviewResult(job_id=job_id, comments=comments, disclaimer=DISCLAIMER)
