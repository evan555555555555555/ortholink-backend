"""
IntegrityGuard — Background auto fact-checker for all agent outputs.

Every completed agent job is automatically scanned:
  1. Regulatory claims are extracted via regex sentence filtering
  2. Each claim is verified against the FAISS vector store (no LLM — pure cosine sim)
  3. An integrity report is injected into the job result as `_integrity`

This gives every agent response an automatic "truth score" grounded in
24,000+ chunks of official regulatory text from 15 countries.
Zero latency impact — runs in a daemon thread after job completion.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that flag a sentence as a "regulatory claim" worth verifying
_CLAIM_PATTERNS = [
    r"\bArticle\s+\d+",             # Article 12
    r"\bRule\s+\d+",                # Rule 7
    r"\bSection\s+\d+",             # Section 4.1
    r"\bAnnex\s+[IVXivx]+",        # Annex XIV
    r"\brequir(?:es?|ed|ement)",    # required / requirement
    r"\bmust\b",                    # must
    r"\bshall\b",                   # shall (regulatory language)
    r"\bcompl(?:y|iant|iance)",     # comply / compliance
    r"\bcertif(?:y|ied|ication)",   # certification
    r"\bregulat(?:ion|ory|ed)",     # regulatory / regulation
    r"\bstandard\b",                # standard
    r"\bsubmit(?:ted)?|submission", # submission
    r"\bapproval|approved",         # approval
    r"\bISO\s+\d+",                 # ISO 13485
    r"\bIEC\s+\d+",                 # IEC 62304
    r"\bClass\s+[IVXivxab]+",       # Class IIb
    r"\bPMA\b|\b510\(k\)",          # PMA / 510(k)
    r"\bCE\s+mark",                 # CE mark
    r"\bFDA\b|\bEMA\b|\bTGA\b",    # regulatory bodies
    r"\bMDR\b|\bIVDR\b",            # EU MDR / IVDR
    r"\bnotif(?:y|ied|ication)",    # notification
    r"\bconformity\b",              # conformity assessment
    r"\bclinical\s+(?:eval|trial|data)", # clinical evaluation
    r"\bpost.?market",              # post-market surveillance
]

_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

# Text fields to scan for claims, in priority order
_TEXT_FIELDS = [
    "summary", "analysis", "output", "plan", "executive_summary",
    "change_summary", "checklist_json", "recommendations", "assessment",
    "findings", "strategy_rationale", "gaps", "actions", "description",
]


def extract_regulatory_claims(text: str, max_claims: int = 15) -> list[str]:
    """
    Extract sentences containing regulatory claims from agent output text.
    Filters to sentences 20–400 chars that match at least one regulatory pattern.
    """
    # Normalise newlines, split on sentence boundaries
    text = re.sub(r"\n{2,}", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if 20 <= len(sent) <= 400 and _CLAIM_RE.search(sent):
            claims.append(sent)
        if len(claims) >= max_claims:
            break
    return claims


def _collect_text_from_result(result: dict) -> str:
    """Pull all useful text out of a job result dict."""
    parts: list[str] = []
    for key in _TEXT_FIELDS:
        val = result.get(key)
        if isinstance(val, str) and len(val) > 30:
            parts.append(val)
        elif isinstance(val, list):
            for item in val[:10]:
                if isinstance(item, str) and len(item) > 20:
                    parts.append(item)
                elif isinstance(item, dict):
                    # e.g. TDA sections have description/regulation_cite
                    for sub_key in ("description", "regulation_cite", "text", "content"):
                        sv = item.get(sub_key, "")
                        if isinstance(sv, str) and len(sv) > 20:
                            parts.append(sv)
    return " ".join(parts)[:6000]


def auto_verify_result(
    result: dict,
    country: str = "",
    device_class: str = "",
) -> dict | None:
    """
    Extract regulatory claims from a job result and verify against FAISS.
    Returns compact integrity report dict, or None on error.

    Designed to run in a daemon background thread — never blocks the API.
    """
    try:
        from app.crews.verify_claims import run_claim_verification

        text = _collect_text_from_result(result)
        if not text:
            return {
                "claims_checked": 0,
                "overall_verdict": "NO_CLAIMS",
                "overall_confidence": 1.0,
                "verified": 0,
                "partial": 0,
                "unverified": 0,
                "top_citations": [],
                "message": "No extractable text in result.",
            }

        claims = extract_regulatory_claims(text)
        if not claims:
            return {
                "claims_checked": 0,
                "overall_verdict": "NO_CLAIMS",
                "overall_confidence": 1.0,
                "verified": 0,
                "partial": 0,
                "unverified": 0,
                "top_citations": [],
                "message": "No regulatory claims detected in output.",
            }

        report = run_claim_verification(
            claims=claims,
            country=country or "US",
            device_class=device_class or "",
        )

        top_citations = list(
            dict.fromkeys(  # deduplicate, preserve order
                r.citation
                for r in report.claim_results
                if r.citation and r.verdict in ("VERIFIED", "PARTIALLY_VERIFIED")
            )
        )[:4]

        return {
            "claims_checked": report.total_claims,
            "overall_verdict": report.overall_verdict,
            "overall_confidence": report.overall_confidence,
            "verified": report.verified,
            "partial": report.partially_verified,
            "unverified": report.unverified + report.contradicted,
            "top_citations": top_citations,
            "message": f"{report.verified}/{report.total_claims} claims verified against FAISS.",
        }

    except Exception as e:
        logger.debug(f"IntegrityGuard.auto_verify_result: {e}")
        return None
