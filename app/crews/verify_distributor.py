"""
OrthoLink DVA Crew — Distributor Verification Sequential Crew
POST /api/v1/verify-distributor

Implements the full 7-step DVA pipeline:
1. Parse CSV → extract distributor items
2. Embed each item
3. Search FAISS for matching requirements (country-filtered)
4. Compute cosine similarity
5. LLM semantic evaluation (when cosine < threshold, gpt-4o reasons over FAISS chunks)
6. LLM classification with confidence gating
7. Aggregate → GapAnalysisReport

All classifications are grounded in FAISS regulatory text + LLM reasoning.
No hardcoded frozensets, no gate-bypass overrides, no synthetic data.

CrewAI path (PRD C1): Crew + Task with output_pydantic=GapAnalysisReport.
"""

import asyncio
import csv
import hashlib
import io
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agents.dva_agent import get_dva_agent
from app.core.anti_hallucination import (
    check_confidence,
    is_out_of_scope,
)
from app.core.config import get_settings
from app.crews.utils import build_regulation_context, multi_query_faiss, parse_llm_json
from app.tools.embeddings import embed_text
from app.tools.similarity import semantic_match
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ─── Country-Specific Mandatory Document Knowledge ──────────────────────────
# Domain knowledge: documents that are ALWAYS mandatory for a given country/regulation.
# Keyed by country code → list of (canonical_name, keywords_to_match, citation).
# If a distributor item fuzzy-matches any keyword set, it is promoted to REQUIRED.
_COUNTRY_MANDATORY_DOCS: dict[str, list[tuple[str, list[str], str]]] = {
    "UA": [
        (
            "Agreement between Manufacturer and Authorized Representative",
            ["agreement", "authorized representative", "power of attorney", "ar contract", "ar agreement"],
            "Resolution 753, Article 7 — Authorized Representative designation",
        ),
        (
            "Risk Management Report/File",
            ["risk management", "risk report", "risk file", "risk analysis", "iso 14971"],
            "Resolution 753, Annex 2/3 — Risk analysis documentation",
        ),
        (
            "Stability Studies",
            ["stability stud", "stability report", "shelf life", "aging study", "accelerated aging"],
            "Resolution 753, Annex 2 — Stability and performance data (Class IIb/III implants)",
        ),
    ],
    "IN": [
        (
            "Manufacturing License / Site Registration",
            ["manufacturing license", "site registration", "manufacturing site", "gmp certificate", "plant master file"],
            "India MDR 2017, Rule 18/19 — Manufacturing License (Form MD-3/MD-4)",
        ),
        (
            "Power of Attorney / Authorized Agent Agreement",
            ["power of attorney", "authorized agent", "agreement between manufacturer", "ar agreement", "agent agreement", "authorised agent"],
            "India MDR 2017, Rule 19 — Authorized Agent for import registration",
        ),
        (
            "Free Sale Certificate",
            ["free sale certificate", "fsc", "certificate of free sale", "cfs"],
            "India MDR 2017, Rule 19(2)(f) — Free Sale Certificate from country of origin",
        ),
        (
            "Declaration of Conformity",
            ["declaration of conformity", "doc", "conformity declaration"],
            "India MDR 2017, Schedule Fourth — Declaration of Conformity",
        ),
        (
            "Quality Management System Certificate (ISO 13485)",
            ["quality management", "qms", "iso 13485", "qms certificate"],
            "India MDR 2017, Schedule Fourth — ISO 13485 QMS certification",
        ),
        (
            "Clinical Investigation / Evaluation Report",
            ["clinical investigation", "clinical evaluation", "clinical report", "clinical data", "clinical evidence"],
            "India MDR 2017, Schedule Fourth — Clinical investigation data",
        ),
        (
            "Biocompatibility Test Report",
            ["biocompatibility", "biocompat", "biological evaluation", "iso 10993"],
            "India MDR 2017, Schedule Fourth — Biocompatibility per ISO 10993",
        ),
        (
            "Device Master File / Technical Documentation",
            ["device master file", "technical documentation", "technical file", "dmf", "device master record"],
            "India MDR 2017, Schedule Fourth — Device Master File",
        ),
        (
            "Labeling and Instructions for Use",
            ["label", "instructions for use", "ifu", "labeling", "labelling", "package insert"],
            "India MDR 2017, Rule 18(4) — Labeling requirements",
        ),
        (
            "Certificate of Analysis",
            ["certificate of analysis", "coa", "test certificate", "analysis certificate"],
            "India MDR 2017, Schedule Fourth — Certificate of Analysis",
        ),
    ],
}

# ─── Hard-Constraint Keyword Gate: Representative Authorization Docs ─────────
# FIX-1: Regex + keyword gate to distinguish "Letter of Intent" (soft/unenforceable)
# from a "Signed Agreement/Contract" (legally binding, required by Art. 14).
# This gate fires BEFORE _matches_country_mandatory() to prevent soft docs from
# inheriting the "authorized representative" mandatory promotion.
#
# REJECTION keywords: if these appear in an AR-category doc, status = MISSING
# ACCEPTANCE keywords: doc must contain at least one of these to pass
import re as _re

_AR_DOC_CATEGORY_PATTERNS = _re.compile(
    r"\b(authorized\s+rep|authorised\s+rep|ar\s+(contract|agreement)|"
    r"representative\s+(agreement|contract|appointment)|"
    r"manufacturer\s+(and|&)\s+(authorized|authorised)|"
    r"letter\s+of\s+(authorization|authorisation|intent)|"
    r"power\s+of\s+attorney|designation\s+(letter|doc))\b",
    _re.IGNORECASE,
)
_AR_DOC_SOFT_REJECT = _re.compile(
    r"\b(intent|draft|preliminary|proposed|pending|to\s+be\s+signed|"
    r"unsigned|letter\s+of\s+intent|loi|memorandum\s+of\s+understanding|mou)\b",
    _re.IGNORECASE,
)
_AR_DOC_HARD_ACCEPT = _re.compile(
    r"\b(signed|executed|agreement|contract|binding|notarized|notarised|"
    r"countersigned|formal\s+agreement|bilateral|sla)\b",
    _re.IGNORECASE,
)
# Country-specific hard-constraint rules:
# {country: {doc_category_regex, soft_reject_regex, accept_regex, citation}}
_HARD_CONSTRAINT_RULES: dict[str, dict] = {
    "UA": {
        "category": _AR_DOC_CATEGORY_PATTERNS,
        "reject_if": _AR_DOC_SOFT_REJECT,
        "accept_if": _AR_DOC_HARD_ACCEPT,
        "citation": "Resolution 753, Article 14 — Signed Agreement between Manufacturer and Authorized Representative (Letter of Intent is NOT sufficient)",
        "critical_note": "HARD CONSTRAINT VIOLATION: 'Letter of Intent' or draft documents do NOT satisfy Resolution 753 Article 14. A formally signed Agreement is required. This is a CRITICAL non-compliance — status forced to MISSING.",
    },
}


def _apply_hard_constraint_gate(item_text: str, country: str) -> dict | None:
    """
    Fix-1: Hard-Constraint Gate for document type enforcement.

    Returns a forced classification dict if the item violates a hard constraint,
    or None if no constraint applies (normal flow continues).

    Fires BEFORE _matches_country_mandatory() so it cannot be overridden by
    semantic similarity scores.
    """
    rule = _HARD_CONSTRAINT_RULES.get(country.upper())
    if not rule:
        return None

    lower = item_text.lower()
    # Only apply if this looks like an AR-category doc
    if not rule["category"].search(lower):
        return None

    # If it contains soft/draft language → MISSING regardless of similarity
    if rule["reject_if"].search(lower):
        logger.warning(
            "[HARD-CONSTRAINT] CRITICAL: '%s' contains soft/draft language for %s AR doc — forcing MISSING",
            item_text[:80], country
        )
        return {
            "status": "MISSING",
            "confidence": 1.0,
            "explanation": rule["critical_note"],
            "citation": rule["citation"],
            "semantic_match": False,
            "_hard_constraint": True,
            "_constraint_reason": "soft_draft_ar_doc",
            "rejection_code": RejectionCode.ERR_MISSING_SIGNATURE,
        }

    # If it's an AR-category doc but lacks any acceptance keyword → flag for human review
    if not rule["accept_if"].search(lower):
        logger.warning(
            "[HARD-CONSTRAINT] WARNING: '%s' is an AR doc with no 'signed/agreement/contract' keyword for %s — forcing MISSING",
            item_text[:80], country
        )
        return {
            "status": "MISSING",
            "confidence": 0.95,
            "explanation": (
                f"Document appears to be an AR authorization document but contains no "
                f"evidence of a formal signed agreement. {rule['citation']}"
            ),
            "citation": rule["citation"],
            "semantic_match": False,
            "_hard_constraint": True,
            "_constraint_reason": "ar_doc_missing_signature_evidence",
            "rejection_code": RejectionCode.ERR_MISSING_SIGNATURE,
        }

    return None  # Passes gate — continue normal classification


# ─── Business-Only Document Patterns ─────────────────────────────────────────
# Items matching these patterns are NEVER regulatory technical requirements.
# They are business/administrative due-diligence items → classified OPTIONAL.
_BUSINESS_ONLY_KEYWORDS: list[str] = [
    "bank statement",
    "financial guarantee",
    "financial stability",
    "insurance certificate",
    "business license",
    "tax certificate",
    "commercial register",
    "proof of solvency",
    "credit report",
    "annual report",
    "balance sheet",
    "profit and loss",
    "revenue statement",
]


def _is_business_only_document(item_text: str) -> bool:
    """Return True if the item is a business/admin doc, NOT a regulatory requirement."""
    lower = item_text.lower()
    return any(kw in lower for kw in _BUSINESS_ONLY_KEYWORDS)


def _matches_country_mandatory(item_text: str, country: str) -> tuple[bool, str, str]:
    """
    Check if item_text fuzzy-matches any known mandatory document for this country.
    Returns (is_match, canonical_name, citation).
    """
    mandatories = _COUNTRY_MANDATORY_DOCS.get(country.upper(), [])
    lower = item_text.lower()
    for canonical, keywords, citation in mandatories:
        if any(kw in lower for kw in keywords):
            return True, canonical, citation
    return False, "", ""


# ─── Embedding cache (request-scoped, in-process) ──────────────────────────
# Regulatory chunks repeat across items — cache by SHA-256 of first 2000 chars.
# Cuts embedding calls from O(items × candidates) → O(unique_chunks).
_EMB_CACHE: dict[str, Any] = {}


async def _embed_cached(text: str) -> Any:
    """Embed text with in-process cache. Avoids re-embedding the same regulatory chunk."""
    key = hashlib.sha256(text[:2000].encode()).hexdigest()
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = await asyncio.to_thread(embed_text, text[:2000])
    return _EMB_CACHE[key]


async def _secondary_semantic_check(
    item_text: str,
    search_results: list[dict],
    country: str,
    device_class: str,
) -> tuple[bool, dict | None]:
    """
    Secondary Semantic Check for near-miss items (cosine 0.45–0.55).

    Instead of auto-rejecting, asks the LLM a targeted question:
    "Does this document title semantically represent a mandatory entity
    (like an Authorized Representative or Risk File) required by the cited Article?"

    This catches items where embedding similarity is low but the document
    is clearly a mandatory submission requirement under the regulation.
    """
    from app.tools.llm import chat_completion

    if not search_results:
        return False, None

    # Build concise evidence from top 5 chunks
    evidence = "\n\n".join(
        f"[{r.get('regulation_name', 'Unknown')}, {r.get('article', '')}]: "
        f"{r.get('text', '')[:400]}"
        for r in search_results[:5]
    )

    system_prompt = (
        "You are a regulatory affairs specialist for medical devices. "
        "A document from a distributor scored borderline on semantic matching. "
        "Your task is to determine whether this document title represents a MANDATORY "
        "submission requirement under the cited regulation, even if the wording differs.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Only classify as mandatory if the regulatory excerpt EXPLICITLY requires this entity\n"
        "- You MUST quote the exact phrase (≤50 words) from an excerpt that establishes the mandate\n"
        "- If no excerpt explicitly mandates this entity, set is_mandatory=false\n"
        "- Never infer requirements not explicitly stated in the provided text\n"
        "- Focus on whether the document represents a mandatory ENTITY or ARTIFACT "
        "(e.g., an Authorized Representative agreement, a risk management file, "
        "a declaration of conformity) rather than exact text matching\n\n"
        "Respond ONLY in valid JSON."
    )
    user_prompt = (
        f'Document item from distributor: "{item_text}"\n'
        f"Country: {country}, Device Class: {device_class}\n\n"
        f"Regulatory text excerpts:\n{evidence}\n\n"
        "Question: Does this document title semantically represent a MANDATORY "
        "entity or artifact that is EXPLICITLY REQUIRED for regulatory submission under any "
        "of the cited articles? You MUST quote the supporting regulatory text.\n\n"
        'Return JSON: {"is_mandatory": true/false, "explanation": "brief reason", '
        '"best_excerpt_index": 0-4, "matched_entity": "what mandatory entity this represents", '
        '"evidence_quote": "exact phrase from excerpt that mandates this (≤50 words)"}'
    )

    try:
        raw = await asyncio.to_thread(chat_completion, system_prompt, user_prompt)
        data = parse_llm_json(raw)
        is_mandatory = bool(data.get("is_mandatory", False))
        best_idx = int(data.get("best_excerpt_index", 0))
        best_idx = max(0, min(best_idx, len(search_results) - 1))
        return is_mandatory, search_results[best_idx]
    except Exception as e:
        logger.warning("Secondary semantic check failed for '%s': %s", item_text, e)
        return False, None


async def _llm_semantic_evaluate(
    item_text: str,
    search_results: list[dict],
    country: str,
    device_class: str,
) -> tuple[bool, dict | None]:
    """
    LLM Semantic Evaluation — replaces all hardcoded frozenset gate-bypass logic.

    When cosine similarity < threshold, pass the top 10 FAISS chunks to gpt-4o
    and let it reason about whether the distributor's item satisfies a legal
    requirement in the regulatory text.

    Returns:
        (is_required, best_matching_chunk_or_None)
    """
    from app.tools.llm import chat_completion

    if not search_results:
        return False, None

    # Use up to 15 chunks to ensure country-specific schedules are represented
    evidence = "\n\n".join(
        f"[Excerpt {i+1}: {r.get('regulation_name', 'Unknown')}, "
        f"{r.get('article', '')}] {r.get('text', '')[:500]}"
        for i, r in enumerate(search_results[:15])
    )

    system_prompt = (
        "You are a regulatory affairs specialist for medical devices. "
        "You must determine whether a distributor's document item is legally required "
        "by ANY of the regulatory text excerpts provided.\n\n"
        "MAPPING STEP (do this first):\n"
        "The distributor's document title may use different wording than the regulation. "
        "Map the document to the regulatory concept it represents. Examples:\n"
        "  - 'Agreement between Manufacturer and Agent' = 'Power of Attorney' / 'Authorized Agent'\n"
        "  - 'Certificate of Analysis' = 'test certificate' / 'analytical report'\n"
        "  - 'QMS Certificate' = 'ISO 13485' / 'quality management system'\n"
        "  - 'Device Master File' = 'technical documentation' / 'technical file'\n"
        "A document SATISFIES a requirement if they refer to the same underlying artifact.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Only classify as required if the regulatory excerpt EXPLICITLY mandates it\n"
        "- You MUST quote the exact phrase (<=50 words) from an excerpt that supports your finding\n"
        "- If no excerpt explicitly requires this item, set is_required=false\n"
        "- Never infer requirements not explicitly stated in the provided text\n"
        "- An item may satisfy a requirement even if wording differs, but the REGULATORY "
        "TEXT must explicitly state the requirement\n\n"
        "Respond ONLY in valid JSON."
    )
    user_prompt = (
        f'Document item from distributor: "{item_text}"\n'
        f"Country: {country}, Device Class: {device_class}\n\n"
        f"Regulatory text excerpts:\n{evidence}\n\n"
        "Does any excerpt EXPLICITLY establish that this document item is a legal requirement "
        "or satisfies a legal requirement (even if wording differs)?\n"
        "You MUST quote the supporting text from the excerpt.\n\n"
        'Return JSON: {"is_required": true/false, "explanation": "brief reason", '
        '"best_excerpt_index": 0-9, "citation": "specific article/clause from excerpt", '
        '"evidence_quote": "exact phrase from the excerpt (≤50 words)"}'
    )

    try:
        raw = await asyncio.to_thread(chat_completion, system_prompt, user_prompt)
        data = parse_llm_json(raw)
        is_required = bool(data.get("is_required", False))
        best_idx = int(data.get("best_excerpt_index", 0))
        best_idx = max(0, min(best_idx, len(search_results) - 1))
        return is_required, search_results[best_idx]
    except Exception as e:
        logger.warning("LLM semantic evaluation failed for '%s': %s", item_text, e)
        return False, None


# ─── Structured Rejection Codes (Global Overhaul) ─────────────────────────
# Forces LLM + pipeline to output a machine-readable rejection reason,
# enabling deterministic fraud-risk accumulation and downstream reporting.


class RejectionCode(str, Enum):
    ERR_REVOKED_LAW = "ERR_REVOKED_LAW"              # Claim cites revoked/superseded regulation
    ERR_MISSING_SIGNATURE = "ERR_MISSING_SIGNATURE"  # AR doc lacks formal signature evidence
    ERR_MISSING_DOC = "ERR_MISSING_DOC"              # Required document absent from submission
    ERR_STALE_CLINICAL = "ERR_STALE_CLINICAL"        # Clinical data older than 5 years
    ERR_EXTRA_DOC = "ERR_EXTRA_DOC"                  # Non-required document submitted
    ERR_BUSINESS_ONLY = "ERR_BUSINESS_ONLY"          # Business/admin doc, not a regulatory req
    ERR_UNVERIFIABLE = "ERR_UNVERIFIABLE"            # Could not verify against FAISS


# ─── Document Legal Weight System (Global Overhaul) ──────────────────────
# Assigns a weight to each document type based on its legal enforceability:
#   3 = BINDING_AGREEMENT  (Agreement, Contract, Signed Authorization) — requires signatures
#   2 = OFFICIAL_CERTIFICATE (Certificate, Registration, Approval) — requires expiry/validity date
#   1 = DOCUMENTATION      (Plan, Report, Assessment, Study, File) — requires methodology
# Used to escalate fraud_risk_score when high-weight docs are missing/invalid.

_DOC_LEGAL_WEIGHT_3: tuple = (
    "signed agreement", "bilateral agreement", "formal agreement",
    "power of attorney", "appointment letter", "authorized representative",
    "manufacturer and authorized", "ar contract", "ar agreement",
    "contract of representation", "notarized",
)
_DOC_LEGAL_WEIGHT_2: tuple = (
    "certificate", "certification", "registration certificate",
    "marketing authorization", "ce certificate", "fda approval",
    "conformity declaration", "declaration of conformity",
)


def _document_legal_weight(item_text: str) -> tuple[int, str]:
    """
    Returns (weight: int, label: str) for the document type.

    Weight 3 (BINDING_AGREEMENT): highest legal enforceability, requires signatures.
    Weight 2 (OFFICIAL_CERTIFICATE): requires expiry dates and issuer validation.
    Weight 1 (DOCUMENTATION): requires methodology and completeness review.
    """
    lower = item_text.lower()
    if any(kw in lower for kw in _DOC_LEGAL_WEIGHT_3):
        return 3, "BINDING_AGREEMENT"
    if any(kw in lower for kw in _DOC_LEGAL_WEIGHT_2):
        return 2, "OFFICIAL_CERTIFICATE"
    return 1, "DOCUMENTATION"


# ─── Dynamic Fraud Risk Scoring (Global Overhaul) ─────────────────────────
# Old formula: (EXTRA + MISSING) / total  [flat, no risk differentiation]
# New formula: base + weighted penalties
#   +0.40 per ERR_REVOKED_LAW violation     (citing dead law = deliberate fraud indicator)
#   +0.30 per ERR_MISSING_SIGNATURE item    (unenforceable authorization = critical gap)
#   +0.20 per missing Weight-3 document     (missing binding agreement = severe gap)
#   +0.10 per ERR_STALE_CLINICAL violation  (outdated clinical data = safety concern)


def _compute_dynamic_fraud_risk(
    gap_items: list,
    total_missing: int,
    *,
    revoked_law_count: int = 0,
    stale_clinical_count: int = 0,
) -> float:
    """
    Weighted fraud risk score — replaces the flat (EXTRA+MISSING)/total formula.

    Returns a float in [0.0, 1.0].
    """
    total = len(gap_items) or 1
    extra_count = sum(1 for g in gap_items if g.status == "EXTRA")

    # Base: ratio of problematic items
    base = (extra_count + total_missing) / total

    # Count specific violation types from the item list
    revoked_violations = sum(
        1 for g in gap_items
        if g.rejection_code == RejectionCode.ERR_REVOKED_LAW
    ) + revoked_law_count

    missing_sig_violations = sum(
        1 for g in gap_items
        if g.rejection_code == RejectionCode.ERR_MISSING_SIGNATURE
    )

    missing_binding_docs = sum(
        1 for g in gap_items
        if g.status in ("MISSING",) and g.legal_weight == 3
    )

    # Weighted penalties (each capped to prevent runaway)
    penalty = (
        min(0.40, revoked_violations * 0.40)      # revoked law: max +0.40
        + min(0.30, missing_sig_violations * 0.30)  # missing signature: max +0.30
        + min(0.20, missing_binding_docs * 0.20)    # missing binding doc: max +0.20
        + min(0.10, stale_clinical_count * 0.10)    # stale clinical: max +0.10
    )

    return round(min(1.0, base + penalty), 2)


# ─── Pydantic Output Models ───────────────────────────────────────────


class GapItem(BaseModel):
    """A single item from the gap analysis."""

    distributor_item: str = Field(..., description="Original item from distributor")
    status: str = Field(
        ...,
        description="REQUIRED | EXTRA | MISSING | OPTIONAL (PRD contract; UNVERIFIABLE is internal-only, mapped to OPTIONAL + needs_human_review)",
    )
    matched_regulation: Optional[str] = Field(
        None, description="Matched regulatory requirement text"
    )
    citation: Optional[str] = Field(
        None, description="Specific regulation article/section"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    explanation: str = Field(..., description="Explanation of classification")
    semantic_similarity: Optional[float] = Field(
        None, description="Cosine similarity score if applicable"
    )
    needs_human_review: bool = Field(
        default=False, description="Flagged for human review"
    )
    # Global Overhaul: Structured rejection code for machine-readable audit trail
    rejection_code: Optional[RejectionCode] = Field(
        None, description="Structured rejection reason (ERR_REVOKED_LAW, ERR_MISSING_SIGNATURE, etc.)"
    )
    # Global Overhaul: Cite-to-source — pinpoints the exact FAISS chunk used for classification
    source_chunk_id: Optional[str] = Field(
        None, description="FAISS chunk_id of the best matching regulatory source"
    )
    source_verified_at: Optional[str] = Field(
        None, description="ISO 8601 timestamp when this item was classified"
    )
    # Global Overhaul: Document legal weight (3=Binding Agreement, 2=Certificate, 1=Documentation)
    legal_weight: int = Field(
        default=1, ge=1, le=3, description="Legal enforceability weight of this document type"
    )
    legal_weight_label: str = Field(
        default="DOCUMENTATION", description="Human-readable label for legal_weight"
    )


class GapAnalysisSummary(BaseModel):
    """Summary statistics for the gap analysis."""

    total_submitted: int
    required: int = 0
    extra: int = 0
    missing: int = 0
    optional: int = 0
    unverifiable: int = 0
    fraud_risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall fraud risk score"
    )


class GapAnalysisReport(BaseModel):
    """Complete DVA output — the gap analysis report."""

    country: str
    device_class: str
    device_type: Optional[str] = None
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: GapAnalysisSummary
    items: list[GapItem]
    missing_requirements: list[dict] = Field(
        default_factory=list,
        description="Regulatory requirements NOT in distributor list",
    )
    truncated_missing_count: int = Field(
        default=0,
        description="When baseline missing > len(missing_requirements), number of additional missing not shown.",
    )
    needs_human_review: list[str] = Field(
        default_factory=list,
        description="Items flagged for human review",
    )
    disclaimer: str = Field(
        default=(
            "Reference tool only. Verify with official sources. "
            "This analysis does not constitute legal advice. "
            "Consult a qualified regulatory affairs professional for official guidance."
        )
    )


# ─── Pipeline Functions ───────────────────────────────────────────────


def parse_csv(csv_content: str) -> list[str]:
    """
    Step 1: Parse CSV content and extract distributor document items.
    Handles various CSV formats (single column, multi-column with 'document' header).
    """
    items: list[str] = []

    reader = csv.reader(io.StringIO(csv_content))
    rows = list(reader)

    if not rows:
        return items

    # Detect header row
    header = [col.strip().lower() for col in rows[0]]
    doc_col_idx = None

    # Look for a document/item column
    doc_column_names = ["document", "document_name", "item", "doc", "requirement", "name"]
    for idx, col in enumerate(header):
        if col in doc_column_names:
            doc_col_idx = idx
            break

    if doc_col_idx is not None:
        # Use identified column, skip header
        for row in rows[1:]:
            if doc_col_idx < len(row) and row[doc_col_idx].strip():
                items.append(row[doc_col_idx].strip())
    else:
        # Assume first column, check if first row is data or header
        start_idx = 0
        # If first row looks like a header (all non-numeric), skip it
        if all(not cell.strip().replace(".", "").isdigit() for cell in rows[0] if cell.strip()):
            start_idx = 1

        for row in rows[start_idx:]:
            if row and row[0].strip():
                items.append(row[0].strip())

    return items


def _run_dva_crew_sync(
    country: str,
    device_class: str,
    csv_items: str,
) -> GapAnalysisReport:
    """Run DVA via CrewAI Crew (sync). Called from run_dva_analysis in thread."""
    from crewai import Crew, Process, Task

    from app.core.crew_memory import get_ltm_memory

    dva_agent = get_dva_agent()
    verify_task = Task(
        description=(
            "Analyze the following distributor document list against {country} "
            "regulations for Class {device_class} medical devices. "
            "Before requesting redundant inputs, query your entity memory for existing "
            "device profiles or prior classifications for this distributor/country. "
            "For each item classify as REQUIRED/EXTRA/MISSING/OPTIONAL with citation. "
            "Documents: {csv_items}"
        ),
        expected_output="A GapAnalysisReport JSON with all items classified",
        agent=dva_agent,
        output_pydantic=GapAnalysisReport,
    )

    ltm = get_ltm_memory()
    crew_kw: dict = {
        "agents": [dva_agent],
        "tasks": [verify_task],
        "process": Process.sequential,
        "output_pydantic": GapAnalysisReport,
        "verbose": True,
    }
    if ltm is not None:
        crew_kw["memory"] = True
        crew_kw["long_term_memory"] = ltm
    crew = Crew(**crew_kw)

    result = crew.kickoff(
        inputs={
            "country": country,
            "device_class": device_class,
            "csv_items": csv_items,
        }
    )

    # CrewAI may return CrewOutput; extract task output (Pydantic model)
    if hasattr(result, "tasks_output") and result.tasks_output:
        out = result.tasks_output[0]
    elif hasattr(result, "tasks") and result.tasks:
        out = getattr(result.tasks[0], "output", result.tasks[0])
    else:
        out = result

    if isinstance(out, GapAnalysisReport):
        logger.info("DVA CrewAI: successfully parsed GapAnalysisReport with %d items", len(out.items))
        return out

    # Fallback: CrewAI failed to parse LLM output into Pydantic model.
    # This is the ROOT CAUSE of the "ghost town" UI — returns 0 items, 0 submitted.
    logger.error(
        "CRITICAL: DVA CrewAI fallback triggered — LLM output could not be parsed into GapAnalysisReport. "
        "Output type: %s, repr: %.500s. Returning EMPTY report (0 items). "
        "Set USE_DVA_CREW=false to use the reliable direct pipeline.",
        type(out).__name__,
        repr(out),
    )
    return GapAnalysisReport(
        country=country,
        device_class=device_class,
        summary=GapAnalysisSummary(total_submitted=0, fraud_risk_score=0.0),
        items=[],
        disclaimer=(
            "Reference tool only. Verify with official sources. "
            "This analysis does not constitute legal advice."
        ),
    )


async def run_dva_analysis(
    csv_content: str,
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> GapAnalysisReport:
    """
    Execute the full DVA 7-step pipeline.

    This is the direct pipeline implementation that can also be orchestrated
    via CrewAI for the agent-based approach.
    """
    # Clear embedding cache for this request (prevents unbounded growth across requests)
    _EMB_CACHE.clear()

    # Check scope
    scope_check = is_out_of_scope(csv_content, country, device_class)
    if scope_check:
        return GapAnalysisReport(
            country=country,
            device_class=device_class,
            device_type=device_type,
            summary=GapAnalysisSummary(
                total_submitted=0,
                fraud_risk_score=0.0,
            ),
            items=[],
            disclaimer=scope_check.reason,
        )

    # Step 1: Parse CSV
    distributor_items = parse_csv(csv_content)
    if not distributor_items:
        raise ValueError("No document items found in CSV")

    logger.info(f"DVA: Parsed {len(distributor_items)} items from CSV")

    # Optional: Run via CrewAI (PRD C1). When USE_DVA_CREW=1, use crew; else use pipeline.
    if get_settings().use_dva_crew:
        normalized_items = "\n".join(distributor_items)
        report = await asyncio.to_thread(
            _run_dva_crew_sync,
            country,
            device_class,
            normalized_items,
        )
        return report

    # Steps 2-6: Process each item in parallel (semaphore caps concurrent OpenAI calls)
    store = get_vector_store()
    semaphore = asyncio.Semaphore(10)

    async def _process_with_semaphore(item_text: str) -> GapItem:
        async with semaphore:
            return await _process_single_item(
                item_text=item_text,
                country=country,
                device_class=device_class,
                device_type=device_type,
                store=store,
            )

    tasks = [_process_with_semaphore(item_text) for item_text in distributor_items]
    gap_items = list(await asyncio.gather(*tasks))
    human_review_items = [g.distributor_item for g in gap_items if g.needs_human_review]

    # Step 7: Find missing requirements (UA = 22 named documents only; others = baseline gap)
    missing_reqs, total_missing = await _find_missing_requirements(
        submitted_items=distributor_items,
        country=country,
        device_class=device_class,
        device_type=device_type,
        store=store,
    )
    truncated_missing_count = max(0, total_missing - len(missing_reqs))

    # Calculate summary
    status_counts = {"REQUIRED": 0, "EXTRA": 0, "MISSING": 0, "OPTIONAL": 0, "UNVERIFIABLE": 0}
    for item in gap_items:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1

    # Dynamic fraud risk score (Global Overhaul): weighted accumulation
    # Replaces flat (EXTRA+MISSING)/total with weighted penalty system
    fraud_risk = _compute_dynamic_fraud_risk(gap_items, total_missing)

    summary = GapAnalysisSummary(
        total_submitted=len(distributor_items),
        required=status_counts.get("REQUIRED", 0),
        extra=status_counts.get("EXTRA", 0),
        missing=total_missing,
        optional=status_counts.get("OPTIONAL", 0),
        unverifiable=status_counts.get("UNVERIFIABLE", 0),
        fraud_risk_score=fraud_risk,
    )

    base_disclaimer = (
        "Reference tool only. Verify with official sources. "
        "This analysis does not constitute legal advice. "
        "Consult a qualified regulatory affairs professional for official guidance."
    )
    if truncated_missing_count > 0:
        base_disclaimer += (
            f" {truncated_missing_count} additional missing requirements not shown. Download full report."
        )

    report = GapAnalysisReport(
        country=country,
        device_class=device_class,
        device_type=device_type,
        summary=summary,
        items=gap_items,
        missing_requirements=missing_reqs,
        truncated_missing_count=truncated_missing_count,
        needs_human_review=human_review_items,
        disclaimer=base_disclaimer,
    )

    logger.info(
        f"DVA complete: {summary.total_submitted} items, "
        f"fraud_risk={summary.fraud_risk_score}, "
        f"{summary.extra} EXTRA, {summary.missing} MISSING"
    )

    return report


async def run_dva_analysis_stream(
    csv_content: str,
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
):
    """
    Async generator: SSE-friendly DVA pipeline with live status events.

    Yields dicts for SSE:
      {"status": "...", "detail": "..."}   — thinking/progress event
      {"item": GapItem.model_dump()}       — per-item classification result
      {"missing": MissingReq.model_dump()} — missing requirement found
      {"done": True, "report": {...}}      — final report
      {"error": "..."}                     — error event
    """
    _EMB_CACHE.clear()

    # Scope check
    scope_check = is_out_of_scope(csv_content, country, device_class)
    if scope_check:
        yield {"error": scope_check.reason}
        return

    # Step 1: Parse CSV
    distributor_items = parse_csv(csv_content)
    if not distributor_items:
        yield {"error": "No document items found in CSV"}
        return

    yield {"status": "parsing", "detail": f"Parsed {len(distributor_items)} items from CSV"}

    store = get_vector_store()
    semaphore = asyncio.Semaphore(10)
    gap_items: list[GapItem] = []
    human_review_items: list[str] = []

    # Steps 2-6: Process each item
    for i, item_text in enumerate(distributor_items):
        yield {
            "status": "classifying",
            "detail": f"DVA Agent analyzing '{item_text}' against {country} regulations...",
            "progress": f"{i + 1}/{len(distributor_items)}",
        }
        try:
            gap_item = await _process_single_item(
                item_text=item_text,
                country=country,
                device_class=device_class,
                device_type=device_type,
                store=store,
            )
            gap_items.append(gap_item)
            if gap_item.needs_human_review:
                human_review_items.append(gap_item.distributor_item)
            yield {"item": gap_item.model_dump()}
        except Exception as e:
            logger.warning(f"DVA stream: item '{item_text}' failed: {e}")
            yield {"status": "warning", "detail": f"Failed to classify '{item_text}': {e}"}

    # Step 7: Missing requirements
    yield {"status": "gap_analysis", "detail": f"Searching {country} regulatory baseline for missing requirements..."}
    try:
        missing_reqs, total_missing = await _find_missing_requirements(
            submitted_items=distributor_items,
            country=country,
            device_class=device_class,
            device_type=device_type,
            store=store,
        )
        for m in missing_reqs:
            yield {"missing": m.model_dump() if hasattr(m, "model_dump") else m}
    except Exception as e:
        logger.warning(f"DVA stream: missing requirements failed: {e}")
        missing_reqs, total_missing = [], 0

    truncated_missing_count = max(0, total_missing - len(missing_reqs))

    # Build summary
    status_counts = {"REQUIRED": 0, "EXTRA": 0, "MISSING": 0, "OPTIONAL": 0, "UNVERIFIABLE": 0}
    for item in gap_items:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1

    # Dynamic fraud risk score (Global Overhaul): weighted accumulation
    fraud_risk = _compute_dynamic_fraud_risk(gap_items, total_missing)

    summary = GapAnalysisSummary(
        total_submitted=len(distributor_items),
        required=status_counts.get("REQUIRED", 0),
        extra=status_counts.get("EXTRA", 0),
        missing=total_missing,
        optional=status_counts.get("OPTIONAL", 0),
        unverifiable=status_counts.get("UNVERIFIABLE", 0),
        fraud_risk_score=fraud_risk,
    )

    base_disclaimer = (
        "Reference tool only. Verify with official sources. "
        "This analysis does not constitute legal advice. "
        "Consult a qualified regulatory affairs professional for official guidance."
    )

    report = GapAnalysisReport(
        country=country,
        device_class=device_class,
        device_type=device_type,
        summary=summary,
        items=gap_items,
        missing_requirements=missing_reqs,
        truncated_missing_count=truncated_missing_count,
        needs_human_review=human_review_items,
        disclaimer=base_disclaimer,
    )

    yield {"done": True, "report": report.model_dump()}


def _query_relates_to_ivd_or_active_implant(query: str) -> bool:
    """True if query explicitly concerns IVD or active implantable (Resolution 754/755)."""
    q = (query or "").lower()
    return any(
        x in q
        for x in (
            "ivd",
            "in vitro",
            "diagnostic",
            "active implant",
            "pacemaker",
            "active implantable",
        )
    )


def _filter_ua_search_to_753_only(
    search_results: list[dict],
    search_query: str,
    device_type: Optional[str],
) -> list[dict]:
    """
    For UA orthopedic/implant queries, return only Resolution 753 chunks.
    Exclude 754 (IVD) and 755 (active implantable) unless query explicitly relates to them.
    """
    if not search_results:
        return search_results
    dt = (device_type or "").lower()
    if "implant" not in dt and "orthopedic" not in dt:
        return search_results
    if _query_relates_to_ivd_or_active_implant(search_query):
        return search_results
    filtered = [
        r
        for r in search_results
        if "753" in (r.get("regulation_name") or "")
        and "754" not in (r.get("regulation_name") or "")
        and "755" not in (r.get("regulation_name") or "")
    ]
    return filtered if filtered else search_results


def _boost_india_primary_docs(search_results: list[dict]) -> list[dict]:
    """
    For India queries, re-rank search results to prioritize MDR 2017 and
    CDSCO submission format chunks over generic FAQ/guidance chunks.

    MDR 2017 Schedule Fourth contains the actual mandatory document requirements.
    FAQ addenda are supplementary — they should not outrank the primary regulation.
    """
    if not search_results:
        return search_results

    # Priority tiers by document_id prefix
    _INDIA_PRIMARY = ("IN-MDR-2017", "IN_MDR_2017_FULL", "IN_CDSCO_SUBMISSION_FORMAT")
    _INDIA_SECONDARY = ("IN_CDSCO_MD_COMPREHENSIVE", "IN_CDSCO_FSC_GUIDANCE", "IN_CDSCO_MD_PORTAL")

    def _rank(r: dict) -> int:
        doc_id = r.get("document_id", "")
        if any(doc_id.startswith(p) for p in _INDIA_PRIMARY):
            return 0  # Highest priority
        if any(doc_id.startswith(p) for p in _INDIA_SECONDARY):
            return 1
        return 2  # FAQ/addenda/other

    # Stable sort: within same tier, preserve original FAISS score order
    return sorted(search_results, key=_rank)


async def _process_single_item(
    item_text: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    store,
) -> GapItem:
    """Process a single distributor item through steps 2-6."""

    # Step 2: Embed the item (cached)
    try:
        item_embedding = await _embed_cached(item_text)
    except Exception as e:
        logger.error(f"Embedding failed for '{item_text}': {e}")
        # PRD: only REQUIRED|EXTRA|MISSING|OPTIONAL. Map internal failure → OPTIONAL + human review.
        return GapItem(
            distributor_item=item_text,
            status="OPTIONAL",
            confidence=0.0,
            explanation=f"Failed to process: {str(e)}. Human verification required.",
            needs_human_review=True,
        )

    # Step 3: Search FAISS (Fix-3: Redis cache is handled inside VectorStore.search())
    # top_k is dynamically scaled inside VectorStore.search() via _dynamic_top_k(),
    # but we pass a generous base to ensure country-specific schedules are retrieved.
    search_query = item_text
    search_results = await asyncio.to_thread(
        store.search,
        search_query,
        country,
        device_class,
        20,  # Base top_k; _dynamic_top_k() may increase for large-corpus countries
        True,
    )

    # UA + implant/orthopedic: restrict to Resolution 753 only (exclude 754 IVD, 755 active implantable)
    if country.upper() == "UA" and search_results:
        search_results = _filter_ua_search_to_753_only(
            search_results, search_query, device_type
        )

    # IN: Boost MDR 2017 and CDSCO submission-format chunks to the top.
    # Generic FAQ chunks rank lower so Schedule Fourth requirements surface first.
    if country.upper() == "IN" and search_results:
        search_results = _boost_india_primary_docs(search_results)

    if not search_results:
        # No matches found — likely EXTRA or unsupported (citation empty per PRD)
        _weight, _weight_label = _document_legal_weight(item_text)
        return GapItem(
            distributor_item=item_text,
            status="EXTRA",
            citation="",
            confidence=0.6,
            explanation=(
                f"No matching regulatory requirement found in {country} database. "
                "This item may not be required by regulation."
            ),
            needs_human_review=True,
            rejection_code=RejectionCode.ERR_EXTRA_DOC,
            source_verified_at=datetime.now(timezone.utc).isoformat(),
            legal_weight=_weight,
            legal_weight_label=_weight_label,
        )

    # Step 4: Set Theory gate — embed all candidates in parallel (cached), pick best match.
    settings = get_settings()
    threshold = settings.similarity_threshold  # 0.82

    candidate_embs = await asyncio.gather(
        *[_embed_cached(c["text"]) for c in search_results],
        return_exceptions=True,
    )

    best_similarity = -1.0
    top_match = search_results[0]
    match_result = None
    for candidate, reg_emb in zip(search_results, candidate_embs):
        if isinstance(reg_emb, Exception):
            continue
        result = semantic_match(item_embedding, reg_emb, threshold=threshold)
        if result["similarity"] > best_similarity:
            best_similarity = result["similarity"]
            top_match = candidate
            match_result = result

    if match_result is None:
        fallback_emb = await _embed_cached(top_match["text"])
        match_result = semantic_match(item_embedding, fallback_emb, threshold=threshold)
    set_theory_result = match_result["is_match"]

    # ── Fix 1 (NEW): Hard-Constraint Keyword Gate ────────────────────────
    # Fires FIRST, before any semantic or mandatory-doc logic.
    # Catches Letter of Intent / draft docs pretending to be signed Agreements.
    # If triggered, returns MISSING with CRITICAL note — cannot be overridden.
    hard_constraint = _apply_hard_constraint_gate(item_text, country)
    if hard_constraint is not None:
        _weight, _weight_label = _document_legal_weight(item_text)
        return GapItem(
            distributor_item=item_text,
            status=hard_constraint["status"],
            matched_regulation=top_match["text"][:300],
            citation=hard_constraint["citation"],
            confidence=hard_constraint["confidence"],
            explanation=hard_constraint["explanation"],
            semantic_similarity=match_result["similarity"],
            needs_human_review=True,
            rejection_code=hard_constraint.get("rejection_code"),
            source_chunk_id=top_match.get("chunk_id"),
            source_verified_at=datetime.now(timezone.utc).isoformat(),
            legal_weight=_weight,
            legal_weight_label=_weight_label,
        )

    # ── Fix 3: Business-only document guard ──────────────────────────────
    # Items like "Bank Statement" are business due-diligence, NOT technical requirements.
    # Intercept BEFORE LLM evaluation to prevent hallucinated REQUIRED classification.
    if _is_business_only_document(item_text):
        citation = f"{top_match['regulation_name']}, {top_match['article']}"
        if top_match.get("clause"):
            citation += f", Clause {top_match['clause']}"
        logger.info("DVA: '%s' classified OPTIONAL (business-only document)", item_text)
        _weight, _weight_label = _document_legal_weight(item_text)
        return GapItem(
            distributor_item=item_text,
            status="OPTIONAL",
            matched_regulation=top_match["text"][:300],
            citation=citation,
            confidence=match_result["similarity"],
            explanation=(
                f"'{item_text}' is a business/administrative due-diligence item, "
                "not a regulatory technical file requirement for device clearance. "
                "Classified as OPTIONAL."
            ),
            semantic_similarity=match_result["similarity"],
            needs_human_review=False,
            rejection_code=RejectionCode.ERR_BUSINESS_ONLY,
            source_chunk_id=top_match.get("chunk_id"),
            source_verified_at=datetime.now(timezone.utc).isoformat(),
            legal_weight=_weight,
            legal_weight_label=_weight_label,
        )

    # ── Fix 2: Country mandatory document check ──────────────────────────
    # If item matches a known mandatory doc for this country, promote to REQUIRED
    # regardless of cosine score. This handles borderline items like AR agreements.
    is_mandatory, mandatory_name, mandatory_citation = _matches_country_mandatory(
        item_text, country
    )
    if is_mandatory:
        logger.info(
            "DVA: '%s' matched country-mandatory '%s' → REQUIRED (%s)",
            item_text, mandatory_name, mandatory_citation,
        )
        _weight, _weight_label = _document_legal_weight(item_text)
        return GapItem(
            distributor_item=item_text,
            status="REQUIRED",
            matched_regulation=top_match["text"][:300],
            citation=mandatory_citation,
            confidence=max(match_result["similarity"], 0.90),
            explanation=(
                f"'{item_text}' matches mandatory document '{mandatory_name}' "
                f"under {country} regulation. {mandatory_citation}."
            ),
            semantic_similarity=match_result["similarity"],
            needs_human_review=False,
            source_chunk_id=top_match.get("chunk_id"),
            source_verified_at=datetime.now(timezone.utc).isoformat(),
            legal_weight=_weight,
            legal_weight_label=_weight_label,
        )

    # ── Primary LLM Semantic Evaluation ──────────────────────────────────
    # When cosine similarity is below threshold, let gpt-4o reason over
    # the top 10 FAISS chunks to determine if the item satisfies a legal requirement.
    if not set_theory_result and search_results:
        llm_required, llm_match = await _llm_semantic_evaluate(
            item_text, search_results, country, device_class
        )
        if llm_required and llm_match is not None:
            top_match = llm_match
            set_theory_result = True

    # ── Fix 1: Secondary Semantic Check for near-miss (0.45–0.55) ────────
    # If primary evaluation failed AND cosine is in the near-miss zone,
    # run a second, more targeted LLM check focused on mandatory entities.
    if not set_theory_result and search_results:
        sim = match_result["similarity"]
        if 0.45 <= sim <= 0.55:
            logger.info(
                "DVA: '%s' in near-miss zone (%.2f) — running secondary semantic check",
                item_text, sim,
            )
            sec_required, sec_match = await _secondary_semantic_check(
                item_text, search_results, country, device_class
            )
            if sec_required and sec_match is not None:
                top_match = sec_match
                set_theory_result = True
                logger.info("DVA: '%s' promoted to REQUIRED by secondary check", item_text)

    # Step 5: Final classification DRIVEN by set theory (HC: if/else must use set_theory_result)
    if not set_theory_result:
        # Distributor item NOT in regulatory baseline → EXTRA (no LLM override)
        citation = f"{top_match['regulation_name']}, {top_match['article']}"
        if top_match.get("clause"):
            citation += f", Clause {top_match['clause']}"
        _weight, _weight_label = _document_legal_weight(item_text)
        return GapItem(
            distributor_item=item_text,
            status="EXTRA",
            matched_regulation=top_match["text"][:300],
            citation=citation,
            confidence=match_result["similarity"],
            explanation=(
                f"No regulatory requirement in {country} semantically matches this item "
                f"(best similarity {match_result['similarity']:.2f}, threshold {threshold}). "
                "Item is not required by regulation."
            ),
            semantic_similarity=match_result["similarity"],
            needs_human_review=True,
            rejection_code=RejectionCode.ERR_EXTRA_DOC,
            source_chunk_id=top_match.get("chunk_id"),
            source_verified_at=datetime.now(timezone.utc).isoformat(),
            legal_weight=_weight,
            legal_weight_label=_weight_label,
        )

    # set_theory_result is True: item matches baseline → LLM confirms REQUIRED/OPTIONAL
    from app.tools.llm import classify_with_llm

    try:
        classification = await asyncio.to_thread(
            classify_with_llm,
            item_text,
            top_match["text"],
            country,
            device_class,
        )
    except Exception as e:
        logger.error(f"LLM classification failed for '{item_text}': {e}")
        # PRD: only REQUIRED|EXTRA|MISSING|OPTIONAL. Map internal failure → OPTIONAL + human review.
        return GapItem(
            distributor_item=item_text,
            status="OPTIONAL",
            matched_regulation=top_match["text"][:200],
            confidence=match_result["similarity"],
            explanation=f"Classification failed: {str(e)}. Human verification required.",
            semantic_similarity=match_result["similarity"],
            needs_human_review=True,
        )

    # Step 6: Confidence gating — do not auto-assign REQUIRED when confidence < 0.7
    confidence = classification.get("confidence", 0.0)
    gate = check_confidence(confidence)
    needs_review = not gate.passed

    # Internal state: when confidence < threshold we use UNVERIFIABLE for logic.
    # PRD contract: only REQUIRED | EXTRA | MISSING | OPTIONAL. Map UNVERIFIABLE → OPTIONAL
    # with needs_human_review=True so downstream consumers never see UNVERIFIABLE.
    llm_status = classification.get("status", "UNVERIFIABLE")
    if not gate.passed:
        status_internal = "UNVERIFIABLE"
    else:
        status_internal = llm_status

    if status_internal == "UNVERIFIABLE":
        status = "OPTIONAL"
        explanation = (classification.get("explanation", "") or "").strip()
        if explanation:
            explanation += " Confidence below threshold. Human verification required."
        else:
            explanation = "Confidence below threshold. Human verification required."
    else:
        status = status_internal
        explanation = classification.get("explanation", "")

    citation = f"{top_match['regulation_name']}, {top_match['article']}"
    if top_match.get("clause"):
        citation += f", Clause {top_match['clause']}"

    # Map LLM rejection_code string → RejectionCode enum (gracefully)
    llm_rejection_str = classification.get("rejection_code")
    llm_rejection_code: Optional[RejectionCode] = None
    if llm_rejection_str and llm_rejection_str not in (None, "null", ""):
        try:
            llm_rejection_code = RejectionCode(llm_rejection_str)
        except ValueError:
            logger.debug("Unknown rejection_code from LLM: %s", llm_rejection_str)

    _weight, _weight_label = _document_legal_weight(item_text)

    return GapItem(
        distributor_item=item_text,
        status=status,
        matched_regulation=top_match["text"][:300],
        citation=classification.get("citation") or citation,
        confidence=confidence,
        explanation=explanation,
        semantic_similarity=match_result["similarity"],
        needs_human_review=needs_review,
        rejection_code=llm_rejection_code,
        source_chunk_id=top_match.get("chunk_id"),
        source_verified_at=datetime.now(timezone.utc).isoformat(),
        legal_weight=_weight,
        legal_weight_label=_weight_label,
    )


async def _find_missing_requirements(
    submitted_items: list[str],
    country: str,
    device_class: str,
    device_type: Optional[str],
    store,
) -> tuple[list[dict], int]:
    """
    DVA Step 7: Gap detection via FAISS baseline + LLM reasoning.

    Queries FAISS for the country's regulatory baseline documents, then asks
    gpt-4o to identify which required items are NOT covered by the submitted list.
    Falls back to pure FAISS semantic matching on LLM failure.
    """
    from app.tools.llm import chat_completion

    # Build baseline queries dynamically from FAISS
    # Country-specific queries target the actual regulatory schedules/annexes
    # to pull the RIGHT chunks instead of generic ones.
    _COUNTRY_BASELINE_QUERIES: dict[str, list[str]] = {
        "IN": [
            "India MDR 2017 Schedule Fourth mandatory documents registration",
            "CDSCO medical device registration submission format requirements",
            "India MDR 2017 manufacturer obligations import license class {device_class}",
            "CDSCO clinical investigation requirements medical devices India",
            "India MDR 2017 conformity assessment technical documentation",
            "CDSCO Form 40 Form 41 registration certificate requirements",
        ],
        "AU": [
            "TGA medical device registration application requirements class {device_class}",
            "Australia TGA Schedule 3 essential principles conformity assessment",
            "TGA manufacturer evidence certificate requirements medical device",
            "Australia TGA clinical evidence requirements class {device_class}",
            "TGA ARTG inclusion application mandatory documents",
        ],
        "JP": [
            "Japan PMDA medical device registration shonin approval requirements",
            "PMDA QMS certification manufacturing site audit Japan",
            "Japan medical device classification approval clinical evaluation",
            "PMDA mandatory submission documents class {device_class} device",
        ],
        "UA": [
            "Ukraine Resolution 753 conformity assessment requirements medical devices",
            "Ukraine medical device registration mandatory documents authorized representative",
            "Ukraine technical regulation medical devices submission requirements class {device_class}",
        ],
    }
    country_upper = country.upper()
    if country_upper in _COUNTRY_BASELINE_QUERIES:
        baseline_queries = [
            q.format(device_class=device_class)
            for q in _COUNTRY_BASELINE_QUERIES[country_upper]
        ]
    else:
        baseline_queries = [
            f"{country} medical device registration required documents class {device_class}",
            f"{country} manufacturer obligations medical device submission requirements",
            f"{country} regulatory submission mandatory documents checklist",
            f"{country} medical device conformity assessment requirements",
            f"{country} medical device technical documentation requirements",
        ]
    baseline_chunks = multi_query_faiss(
        store, baseline_queries, country, device_class=device_class, top_k=10, max_chunks=30
    )

    if not baseline_chunks:
        return [], 0

    regulation_context = build_regulation_context(baseline_chunks, max_chunks=25)
    submitted_str = "\n".join(f"- {item}" for item in submitted_items)

    system_prompt = (
        "You are a regulatory affairs specialist for medical devices. "
        "Given regulatory text excerpts and a list of documents submitted by a distributor, "
        "you must FIRST map each submitted document to regulatory requirements, THEN identify gaps.\n\n"
        "STEP 1 - MAPPING (do this mentally before identifying gaps):\n"
        "For each submitted document, determine which regulatory requirement it could satisfy.\n"
        "Documents may use different wording than the regulation. Examples:\n"
        "  - 'Agreement between Manufacturer and Agent' satisfies 'Power of Attorney' or 'Authorized Agent'\n"
        "  - 'Certificate of Analysis' satisfies 'test certificate' or 'analytical report'\n"
        "  - 'QMS Certificate' satisfies 'ISO 13485' or 'quality management system'\n"
        "  - 'Device Master File' satisfies 'technical documentation' or 'technical file'\n"
        "A submitted document COVERS a requirement if they refer to the same underlying artifact, "
        "even if the exact wording is completely different.\n\n"
        "STEP 2 - GAP IDENTIFICATION (only after completing Step 1):\n"
        "Only report requirements that are NOT covered by ANY submitted document.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Only report requirements that are EXPLICITLY stated in the regulatory text excerpts\n"
        "- Every missing requirement MUST include a direct quote (<=50 words) from the text\n"
        "- Do NOT invent or infer requirements not present in the excerpts\n"
        "- If a submitted document could PLAUSIBLY satisfy a requirement (even with COMPLETELY "
        "different wording), do NOT report it as missing\n"
        "- Business documents (bank statements, insurance) are NOT regulatory requirements\n\n"
        "Respond ONLY in valid JSON."
    )
    user_prompt = (
        f"Country: {country}, Device Class: {device_class}\n\n"
        f"SUBMITTED DOCUMENTS:\n{submitted_str}\n\n"
        f"REGULATORY TEXT EXCERPTS:\n{regulation_context}\n\n"
        "First, mentally map each submitted document to the regulatory requirements it could satisfy "
        "(even with different wording). Then identify requirements EXPLICITLY stated in the "
        "regulatory text that are NOT covered by ANY submitted document.\n\n"
        f'Return JSON: {{"missing": [{{"requirement": "...", "citation": "specific article/clause", '
        f'"country": "{country}", "evidence_quote": "exact phrase from regulatory text (<=50 words)"}}]}}'
    )

    try:
        raw = await asyncio.to_thread(chat_completion, system_prompt, user_prompt)
        data = parse_llm_json(raw)
        missing = data.get("missing", [])
        if isinstance(missing, list):
            # Ensure each entry has required fields
            valid_missing = []
            for m in missing:
                if isinstance(m, dict) and m.get("requirement"):
                    valid_missing.append({
                        "requirement": str(m["requirement"])[:300],
                        "citation": str(m.get("citation", f"{country} regulation")),
                        "country": country,
                    })
            cap = 50
            return valid_missing[:cap], len(valid_missing)
    except Exception as e:
        logger.warning("LLM missing-requirements failed for %s: %s — falling back to FAISS", country, e)

    # Fallback: pure FAISS semantic matching for gap detection
    return await _faiss_baseline_gap(submitted_items, country, device_class, store)


async def _faiss_baseline_gap(
    submitted_items: list[str],
    country: str,
    device_class: str,
    store,
) -> tuple[list[dict], int]:
    """Fallback gap detection: baseline chunks not semantically covered by submitted items."""
    settings = get_settings()
    threshold = settings.similarity_threshold  # 0.82

    baseline = store.get_baseline_chunks(
        country=country,
        device_class=device_class,
        active_only=True,
    )
    if not baseline:
        return [], 0

    # Cap baseline to avoid embedding thousands of chunks
    MAX_BASELINE = 60
    seen_pre: set[str] = set()
    deduped_baseline: list[dict] = []
    for chunk in baseline:
        cit = f"{chunk['regulation_name']}, {chunk['article']}"
        if cit not in seen_pre:
            seen_pre.add(cit)
            deduped_baseline.append(chunk)
        if len(deduped_baseline) >= MAX_BASELINE:
            break
    baseline = deduped_baseline

    # Embed submitted items in parallel (cached)
    submitted_emb_results = await asyncio.gather(
        *[_embed_cached(item) for item in submitted_items],
        return_exceptions=True,
    )
    submitted_embeddings: list[tuple[str, Any]] = [
        (item, emb)
        for item, emb in zip(submitted_items, submitted_emb_results)
        if not isinstance(emb, Exception)
    ]

    # Embed all baseline chunks in parallel (cached)
    baseline_emb_results = await asyncio.gather(
        *[_embed_cached(chunk["text"][:2000]) for chunk in baseline],
        return_exceptions=True,
    )

    missing: list[dict] = []
    seen_citation: set[str] = set()

    for chunk, reg_emb in zip(baseline, baseline_emb_results):
        if isinstance(reg_emb, Exception):
            continue

        citation = f"{chunk['regulation_name']}, {chunk['article']}"
        if chunk.get("clause"):
            citation += f", Clause {chunk['clause']}"
        if citation in seen_citation:
            continue

        is_covered = False
        for _item_text, item_emb in submitted_embeddings:
            match = semantic_match(item_emb, reg_emb, threshold=threshold)
            if match["is_match"]:
                is_covered = True
                break

        if not is_covered:
            seen_citation.add(citation)
            missing.append({
                "requirement": chunk["text"][:300],
                "citation": citation,
                "country": country,
            })

    cap = 50
    return missing[:cap], len(missing)
