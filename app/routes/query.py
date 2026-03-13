"""
OrthoLink Regulatory Query Route
GET /api/v1/query — Plain-English regulatory questions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.anti_hallucination import check_confidence, is_out_of_scope
from app.middleware.auth import AuthenticatedUser, get_current_user
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


class QueryResponse(BaseModel):
    """Response to a regulatory query. HC-4: out-of-scope returns status REFUSED."""

    question: str
    answer: str
    confidence: float
    citations: list[dict] = Field(default_factory=list)
    country: str
    refused: bool = False
    refusal_reason: Optional[str] = None
    status: str = Field(
        default="ANSWERED",
        description="ANSWERED | REFUSED — REFUSED when out-of-scope or confidence below threshold",
    )


@router.get("", response_model=QueryResponse)
async def regulatory_query(
    q: str = Query(..., min_length=5, max_length=1000, description="Regulatory question"),
    country: str = Query(..., min_length=2, max_length=4, pattern=r"^[A-Za-z]{2,4}$", description="Country code (ISO 3166-1 alpha-2, e.g. US, EU)"),
    device_class: Optional[str] = Query(None, description="Device class"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Answer plain-English regulatory questions with citations.
    HC-4: If confidence < 0.7, returns structured refusal.
    """
    # Scope check — HC-4: structured refusal with status REFUSED
    scope_check = is_out_of_scope(q, country, device_class or "")
    if scope_check:
        return QueryResponse(
            question=q,
            answer="",
            confidence=0.0,
            country=country,
            refused=True,
            refusal_reason=scope_check.reason,
            status="REFUSED",
        )

    # Search vector store
    store = get_vector_store()
    results = store.search(
        query=q,
        country=country,
        device_class=device_class,
        top_k=5,
    )

    if not results:
        return QueryResponse(
            question=q,
            answer="",
            confidence=0.0,
            country=country,
            refused=True,
            refusal_reason=f"No relevant regulatory information found for this query in {country}.",
            status="REFUSED",
        )

    # Use LLM to synthesize answer from retrieved chunks
    from app.tools.llm import chat_completion

    context = "\n\n".join(
        f"[{r['regulation_name']}, {r['article']}]: {r['text'][:500]}"
        for r in results
    )

    system_prompt = (
        "You are a regulatory affairs expert. Answer the question using ONLY the provided "
        "regulatory text. Cite specific articles/sections. If the provided text does not "
        "contain enough information to answer confidently, say so explicitly."
    )

    user_prompt = f"Question: {q}\nCountry: {country}\n\nRegulatory Context:\n{context}"

    try:
        answer = chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"Query LLM failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

    # Build citations
    citations = []
    for r in results[:3]:
        citation = {
            "regulation": r["regulation_name"],
            "article": r["article"],
            "text_excerpt": r["text"][:200],
            "score": r["score"],
        }
        if r.get("clause"):
            citation["clause"] = r["clause"]
        citations.append(citation)

    # Use top result score as confidence proxy
    confidence = results[0]["score"] if results else 0.0
    gate = check_confidence(confidence)

    if not gate.passed:
        return QueryResponse(
            question=q,
            answer="",
            confidence=confidence,
            citations=citations,
            country=country,
            refused=True,
            refusal_reason=gate.reason,
            status="REFUSED",
        )

    return QueryResponse(
        question=q,
        answer=answer,
        confidence=confidence,
        citations=citations,
        country=country,
        status="ANSWERED",
    )
