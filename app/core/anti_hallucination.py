"""
OrthoLink Anti-Hallucination Module
Confidence gating + structured refusal per HC-4.
Every claim must cite source (HC-3). Confidence < 0.7 = structured refusal.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings


class Citation(BaseModel):
    """A regulatory citation with source tracking."""

    regulation_name: str = Field(..., description="Name of the regulation (e.g., 'Resolution 753')")
    article: str = Field(..., description="Article/Section number")
    clause: Optional[str] = Field(None, description="Specific clause if applicable")
    country: str = Field(..., description="Country code (e.g., 'UA', 'US')")
    text_excerpt: str = Field(..., description="Relevant excerpt from the regulation")
    chunk_id: Optional[str] = Field(None, description="Reference to vector store chunk")


class ConfidenceGate(BaseModel):
    """Result of confidence gating check."""

    passed: bool = Field(..., description="Whether the confidence threshold was met")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    threshold: float = Field(..., description="Applied threshold")
    reason: Optional[str] = Field(None, description="Reason for refusal if not passed")


class StructuredRefusal(BaseModel):
    """Structured refusal when confidence is below threshold."""

    refused: bool = True
    reason: str
    confidence: float
    threshold: float
    suggestion: str = Field(
        default="Please consult a qualified regulatory affairs professional for this query.",
        description="Suggested action for the user",
    )


def check_confidence(
    confidence: float,
    threshold: Optional[float] = None,
) -> ConfidenceGate:
    """
    Check if confidence meets the threshold.
    Returns ConfidenceGate with pass/fail and details.
    """
    settings = get_settings()
    effective_threshold = threshold or settings.confidence_threshold

    if confidence >= effective_threshold:
        return ConfidenceGate(
            passed=True,
            confidence=confidence,
            threshold=effective_threshold,
        )

    return ConfidenceGate(
        passed=False,
        confidence=confidence,
        threshold=effective_threshold,
        reason=f"Confidence {confidence:.2f} is below threshold {effective_threshold:.2f}. "
        "Result not reliable enough to present without human review.",
    )


def create_refusal(
    confidence: float,
    reason: str,
    threshold: Optional[float] = None,
) -> StructuredRefusal:
    """
    Create a structured refusal response.
    Used when confidence is below threshold or query is out of scope.
    """
    settings = get_settings()
    effective_threshold = threshold or settings.confidence_threshold

    return StructuredRefusal(
        refused=True,
        reason=reason,
        confidence=confidence,
        threshold=effective_threshold,
    )


def validate_citations(citations: list[Citation]) -> bool:
    """
    Validate that all citations have required fields (HC-3).
    Every claim must have a traceable source.
    """
    if not citations:
        return False

    for citation in citations:
        if not citation.regulation_name or not citation.article or not citation.country:
            return False
        if not citation.text_excerpt:
            return False

    return True


def is_out_of_scope(query: str, country: str, device_class: str) -> Optional[StructuredRefusal]:
    """
    Check if a query is outside the scope of OrthoLink's regulatory database.
    Returns a StructuredRefusal if out of scope, None if in scope.
    """
    supported_countries = {
        "US", "EU", "UK", "CA", "JP", "AU", "IN", "BR",
        "UA", "CN", "RU", "CH", "MX", "KR", "SA",
    }

    if country.upper() not in supported_countries:
        return StructuredRefusal(
            refused=True,
            reason=f"Country '{country}' is not currently in OrthoLink's regulatory database. "
            f"Supported countries: {', '.join(sorted(supported_countries))}",
            confidence=0.0,
            threshold=0.0,
            suggestion="Contact support to request coverage for this country.",
        )

    out_of_scope_keywords = [
        "nuclear", "fuel rod", "weapon", "explosive", "pharmaceutical",
        "drug", "biologic", "vaccine", "food", "cosmetic",
    ]
    query_lower = query.lower()
    for keyword in out_of_scope_keywords:
        if keyword in query_lower:
            return StructuredRefusal(
                refused=True,
                reason=f"Query contains '{keyword}' which is outside the scope of medical device "
                "regulatory intelligence. OrthoLink covers medical devices only.",
                confidence=0.0,
                threshold=0.0,
                suggestion="OrthoLink is designed for medical device regulatory intelligence only.",
            )

    return None
