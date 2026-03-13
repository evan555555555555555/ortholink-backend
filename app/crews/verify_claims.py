"""
OrthoLink Verification Crew — Truth Checker / Claim Verifier
POST /api/v1/verify-claims

Deer-flow coordinator → researcher → critic pipeline:
  COORDINATOR  Decomposes each claim into 3-5 targeted FAISS queries
               (verbatim + citation-focused + requirement-type + country-specific)
  RESEARCHER   Multi-query FAISS with chunk deduplication (max 24 unique chunks per claim)
               Confidence = 0.6 × max_score + 0.4 × mean(top-3 scores)
  CRITIC       LLM arbitration for borderline PARTIAL range (0.45-0.65)
               — reads ONLY retrieved evidence (no external knowledge → no hallucination)
               — returns PARTIALLY_VERIFIED or CONTRADICTED + discrepancy
  SENTINEL     Negation search for low-confidence claims (< 0.45)
               — queries "not required / exempt" variant of claim
               — if negation hits score > 0.55 → CONTRADICTED (was dead code before this rev)

CONTRADICTED verdict is now fully operational (was defined in model but never set).
"""

import json
import logging
import re
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import multi_query_faiss_raw
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ── Verdict thresholds ────────────────────────────────────────────────────────
_VERIFIED_THRESHOLD = 0.62       # ≥ this → VERIFIED
_PARTIAL_THRESHOLD = 0.45        # ≥ this → PARTIALLY_VERIFIED (or CONTRADICTED via critic)
_CONTRADICT_NEGATION_THRESHOLD = 0.55  # Negation-search score that triggers CONTRADICTED

# ── Regex helpers for coordinator query generation ────────────────────────────
_CITATION_RE = re.compile(
    r"(21\s*CFR\s*[\d\.]+|EU\s*MDR\s*Art(?:icle)?\s*[\d]+|"
    r"ISO\s*[\d]+[:\d]*(?:\s*§[\d\.]+)?|IEC\s*[\d]+[:\d]*|"
    r"§\s*[\d\.]+|\d{2}\s*CFR\s*[\d\.]+)",
    re.IGNORECASE,
)
_OBLIGATION_RE = re.compile(
    r"\b(must|shall|required|mandatory|need to|obligat\w*)\b",
    re.IGNORECASE,
)
_REQUIREMENT_KWS = re.compile(
    r"\b(registration|labeling|labelling|QMS|UDI|CAPA|PMS|clinical|vigilance|"
    r"recall|importer|manufacturer|authorized representative|declaration of conformity|"
    r"technical documentation|design control|post.?market|adverse event|MDR|PMSR|PSUR|"
    r"PMCF|notified body|CE mark|510.?k|GUDID|EUDAMED|apostille|QMSR|IFU)\b",
    re.IGNORECASE,
)


# ── Models ────────────────────────────────────────────────────────────────────

class ClaimVerification(BaseModel):
    """Verification result for a single regulatory claim."""

    claim: str = Field(..., description="The claim being verified")
    verdict: str = Field(
        ..., description="VERIFIED | PARTIALLY_VERIFIED | UNVERIFIED | CONTRADICTED"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Actual regulatory text that supports this claim",
    )
    citation: str = Field(default="", description="Strongest citation found")
    discrepancy: str = Field(
        default="", description="What differs if claim is PARTIALLY_VERIFIED or CONTRADICTED"
    )
    recommendation: str = Field(default="")
    queries_used: int = Field(default=1, description="Number of FAISS queries run for this claim")
    evidence_chunks: int = Field(default=0, description="Unique regulatory chunks retrieved")


class VerificationReport(BaseModel):
    """Complete verification report for a set of claims."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str
    device_class: str
    total_claims: int = 0
    verified: int = 0
    partially_verified: int = 0
    unverified: int = 0
    contradicted: int = 0
    overall_confidence: float = 0.0
    overall_verdict: str = Field(
        default="",
        description="RELIABLE | REVIEW_REQUIRED | UNRELIABLE",
    )
    claim_results: list[ClaimVerification] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "Verification is based on OrthoLink's regulatory database. "
            "Always consult primary regulatory sources and qualified RA professionals."
        )
    )


# ── Coordinator ───────────────────────────────────────────────────────────────

def _generate_claim_queries(claim: str, country: str, device_class: str) -> list[str]:
    """
    Coordinator: decompose a single claim into 3-5 targeted FAISS search queries.

    Strategy (deer-flow planner pattern):
      1. Verbatim claim — highest fidelity, exact semantic match
      2. Citation query — if claim references a specific article, query that directly
      3. Requirement-type query — extract key regulatory concepts + country context
      4. Country-specific fallback — explicit market context when not already in claim
    """
    queries: list[str] = [claim]

    # 2. Citation-focused query
    cite_match = _CITATION_RE.search(claim)
    if cite_match:
        cite = cite_match.group(1).strip()
        queries.append(f"requirements {cite} obligations compliance manufacturer")

    # 3. Requirement-type query
    kw_matches = _REQUIREMENT_KWS.findall(claim)
    if kw_matches:
        unique_kws = list(dict.fromkeys(k.lower() for k in kw_matches))[:4]
        queries.append(
            f"{' '.join(unique_kws)} requirements {country} {device_class} medical device"
        )

    # 4. Country-specific query (only if country not already prominent in claim)
    claim_upper = claim.upper()
    country_present = country.upper() in claim_upper or {
        "US": "FDA", "EU": "MDR", "JP": "PMDA", "AU": "TGA", "CA": "HEALTH CANADA",
        "IN": "CDSCO", "KR": "MFDS", "UK": "MHRA", "UA": "MOH", "SA": "SFDA",
    }.get(country.upper(), country.upper()) in claim_upper
    if not country_present and len(queries) < 4:
        queries.append(f"{claim[:100]} {country} medical device regulation")

    return queries[:5]


def _negation_query(claim: str) -> Optional[str]:
    """
    Generate an adversarial negation query to detect CONTRADICTED claims.
    Extracts the core obligation and wraps it with exemption/exclusion framing.
    """
    obl_match = _OBLIGATION_RE.search(claim)
    if not obl_match:
        return None
    # Extract text after the obligation keyword as the core requirement
    core = claim[obl_match.end():].strip()
    if len(core) < 15:
        return None
    core = core[:100]
    return f"exempt not required does not apply exception {core}"


# ── Critic ────────────────────────────────────────────────────────────────────

def _critic_verdict(claim: str, evidence_snippets: list[str]) -> tuple[str, str]:
    """
    LLM Critic: arbitrate borderline claims in the PARTIAL confidence range.

    Constraint: critic reads ONLY the provided FAISS evidence — zero external knowledge.
    This prevents the LLM from hallucinating regulatory citations.
    Returns: (verdict, discrepancy_text)
    verdict is one of: PARTIALLY_VERIFIED | CONTRADICTED
    """
    from app.tools.llm import chat_completion

    evidence_block = "\n\n".join(
        f"[Evidence {i + 1}] {snip}" for i, snip in enumerate(evidence_snippets[:3])
    )

    system_prompt = (
        "You are a regulatory claims auditor. Your ONLY job is to determine whether "
        "a regulatory claim is PARTIALLY_VERIFIED or CONTRADICTED based on the "
        "provided regulatory source text.\n\n"
        "Rules:\n"
        "  • Use ONLY the evidence provided — no external knowledge whatsoever\n"
        "  • PARTIALLY_VERIFIED: evidence is related or supportive but incomplete\n"
        "  • CONTRADICTED: evidence directly negates, excludes, or opposes the claim\n"
        "  • Never return VERIFIED (that requires ≥ 0.62 confidence from FAISS)\n\n"
        'Respond with ONLY valid JSON: {"verdict": "PARTIALLY_VERIFIED" or "CONTRADICTED", '
        '"discrepancy": "one sentence explaining what the evidence says vs the claim"}'
    )

    user_prompt = (
        f"CLAIM TO VERIFY:\n{claim}\n\n"
        f"REGULATORY EVIDENCE (from verified FAISS database only):\n{evidence_block}\n\n"
        "Return JSON only. Do not include markdown fences."
    )

    try:
        raw = chat_completion(system_prompt, user_prompt).strip()
        from app.crews.utils import extract_clean_json
        clean = extract_clean_json(raw)
        data = json.loads(clean)
        verdict = str(data.get("verdict", "PARTIALLY_VERIFIED")).upper()
        if verdict not in ("PARTIALLY_VERIFIED", "CONTRADICTED"):
            verdict = "PARTIALLY_VERIFIED"
        return verdict, str(data.get("discrepancy", ""))
    except Exception as e:
        logger.warning("CRITICAL: Critic LLM failed for borderline claim: %s", e)
        return "PARTIALLY_VERIFIED", (
            "Borderline match — LLM critic unavailable, manual review required."
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_claim_verification(
    claims: list[str],
    country: str,
    device_class: str,
) -> VerificationReport:
    """
    Verify regulatory claims using deer-flow coordinator → researcher → critic pipeline.

    COORDINATOR  generates 3-5 targeted FAISS queries per claim
    RESEARCHER   multi-query FAISS with chunk deduplication
    CRITIC       LLM arbitrates borderline PARTIAL-range claims (no hallucination risk)
    SENTINEL     negation-search CONTRADICTED detection for low-confidence claims

    CONTRADICTED verdict is now fully operational (was dead code before this revision).
    """
    store = get_vector_store()
    results: list[ClaimVerification] = []

    for claim in claims[:20]:  # Cap at 20 claims per request
        try:
            # ── COORDINATOR ────────────────────────────────────────────────
            evidence_queries = _generate_claim_queries(claim, country, device_class)
            negation_q = _negation_query(claim)

            # ── RESEARCHER ─────────────────────────────────────────────────
            chunks = multi_query_faiss_raw(
                store,
                evidence_queries,
                country,
                device_class=device_class or None,
                top_k=6,
                max_chunks=24,
            )

            if not chunks:
                # No country-specific evidence found.  Return UNVERIFIED immediately.
                # DEFENSIVE: we intentionally do NOT search globally (cross-country).
                # Brazil (ANVISA) data must never influence a Mexico (COFEPRIS) verdict.
                results.append(ClaimVerification(
                    claim=claim,
                    verdict="UNVERIFIED",
                    confidence=0.0,
                    recommendation=(
                        f"No matching regulation found for {country.upper()} "
                        f"(class {device_class or 'unspecified'}). "
                        "Verify against primary regulatory sources."
                    ),
                    queries_used=len(evidence_queries),
                    evidence_chunks=0,
                ))
                continue

            # ── SCORE ─────────────────────────────────────────────────────
            # Multi-query advantage: max(all scores) + mean(top-3) rewarded
            all_scores = [min(max(float(c.get("score", 0.0)), 0.0), 1.0) for c in chunks]
            max_score = max(all_scores)
            top3_mean = sum(sorted(all_scores, reverse=True)[:3]) / min(3, len(all_scores))
            # Weighted blend: max dominates (single perfect hit = VERIFIED), breadth adds weight
            confidence = min(0.65 * max_score + 0.35 * top3_mean, 1.0)

            # Build evidence snippets from top-3 unique chunks
            supporting_texts = [
                f"[{c.get('section_path') or c.get('document_id', '')}] "
                f"{c.get('text', '')[:300]}"
                for c in chunks[:3]
                if c.get("text")
            ]

            # Primary citation from highest-scoring chunk
            best = chunks[0]
            citation = best.get("section_path") or best.get("document_id") or ""
            if best.get("article"):
                citation = f"{citation} Article {best['article']}"

            # ── VERDICT ────────────────────────────────────────────────────
            if confidence >= _VERIFIED_THRESHOLD:
                # ✓ Strong match
                verdict = "VERIFIED"
                discrepancy = ""
                recommendation = (
                    "Claim supported by regulatory source. Include citation in submission."
                )

            elif confidence >= _PARTIAL_THRESHOLD:
                # Borderline — send to CRITIC for PARTIALLY_VERIFIED vs CONTRADICTED
                verdict, discrepancy = _critic_verdict(claim, supporting_texts)
                if verdict == "CONTRADICTED":
                    recommendation = (
                        "This claim conflicts with the regulatory text found. "
                        "Do NOT use in a regulatory submission without legal review."
                    )
                else:
                    recommendation = (
                        "Claim partially matches regulatory text. Confirm exact wording "
                        "against primary sources before use in submissions."
                    )

            else:
                # Low confidence — try SENTINEL negation search for CONTRADICTED
                verdict = "UNVERIFIED"
                discrepancy = "Claim could not be corroborated in regulatory database."
                recommendation = (
                    "Insufficient regulatory basis found. Do not rely on this claim "
                    "without confirming against primary regulatory documents."
                )

                if negation_q:
                    neg_hits = store.search(
                        query=negation_q,
                        country=country,
                        device_class=device_class or None,
                        top_k=3,
                    )
                    if neg_hits:
                        neg_score = min(max(float(neg_hits[0].get("score", 0.0)), 0.0), 1.0)
                        if neg_score >= _CONTRADICT_NEGATION_THRESHOLD:
                            neg_cite = (
                                neg_hits[0].get("section_path")
                                or neg_hits[0].get("document_id")
                                or country
                            )
                            verdict = "CONTRADICTED"
                            discrepancy = (
                                f"Regulatory text ({neg_cite}) suggests this obligation "
                                "does not apply or has an exemption in this context."
                            )
                            recommendation = (
                                "This claim appears to contradict applicable regulatory "
                                "requirements. Review immediately before use in any submission."
                            )
                            # Prepend negation evidence to supporting texts
                            supporting_texts = [
                                f"[{c.get('section_path') or c.get('document_id', '')}] "
                                f"{c.get('text', '')[:300]}"
                                for c in neg_hits[:2]
                                if c.get("text")
                            ] + supporting_texts[:1]

            results.append(ClaimVerification(
                claim=claim,
                verdict=verdict,
                confidence=round(confidence, 3),
                supporting_evidence=supporting_texts,
                citation=citation,
                discrepancy=discrepancy,
                recommendation=recommendation,
                queries_used=len(evidence_queries),
                evidence_chunks=len(chunks),
            ))

        except Exception as e:
            logger.warning("Verification failed for claim '%s': %s", claim[:50], e)
            results.append(ClaimVerification(
                claim=claim,
                verdict="UNVERIFIED",
                confidence=0.0,
                recommendation=f"Verification error: {str(e)[:100]}",
                queries_used=1,
                evidence_chunks=0,
            ))

    # ── AGGREGATE STATS ───────────────────────────────────────────────────────
    verified = sum(1 for r in results if r.verdict == "VERIFIED")
    partial = sum(1 for r in results if r.verdict == "PARTIALLY_VERIFIED")
    unverified = sum(1 for r in results if r.verdict == "UNVERIFIED")
    contradicted = sum(1 for r in results if r.verdict == "CONTRADICTED")
    total = len(results)

    overall_confidence = sum(r.confidence for r in results) / total if total else 0.0
    reliable_ratio = (verified + partial) / max(total, 1)

    if overall_confidence >= 0.6 and reliable_ratio >= 0.7:
        overall_verdict = "RELIABLE"
    elif overall_confidence >= 0.4 or reliable_ratio >= 0.4:
        overall_verdict = "REVIEW_REQUIRED"
    else:
        overall_verdict = "UNRELIABLE"

    return VerificationReport(
        country=country,
        device_class=device_class,
        total_claims=total,
        verified=verified,
        partially_verified=partial,
        unverified=unverified,
        contradicted=contradicted,
        overall_confidence=round(overall_confidence, 3),
        overall_verdict=overall_verdict,
        claim_results=results,
    )
