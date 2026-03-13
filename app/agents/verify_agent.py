"""
OrthoLink Verify Agent — Claims Verification Agent
CrewAI Agent stub: role='Regulatory Claims Verifier'

Production uses FAISS-only (no LLM) for truth checking.
This agent definition exists for CrewAI consistency but the production
verify_claims.py pipeline does not use CrewAI — it's a pure FAISS
cosine similarity check (VERIFIED ≥ 0.62, PARTIAL ≥ 0.45).
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory text to verify claims against."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=10)
    return str([
        {
            "text": r.get("text", "")[:500],
            "regulation_name": r.get("regulation_name"),
            "document_id": r.get("document_id"),
            "score": r.get("score"),
        }
        for r in results
    ])


_verify_agent_instance: Optional[Agent] = None


def get_verify_agent() -> Agent:
    """Return the Verify CrewAI agent (lazy-initialized).

    Note: Production verify_claims.py uses FAISS-only (no LLM) for speed.
    This agent is defined for CrewAI consistency and potential future use.
    """
    global _verify_agent_instance
    if _verify_agent_instance is None:
        _verify_agent_instance = Agent(
            role="Regulatory Claims Verifier",
            goal=(
                "Verify regulatory claims against FAISS vector store data. "
                "Classify claims as VERIFIED, PARTIAL, or UNVERIFIED based on "
                "cosine similarity scores against official regulatory text."
            ),
            backstory=(
                "Fact-checking specialist focused on medical device regulatory claims. "
                "Uses semantic similarity to verify whether claims are supported by "
                "official regulatory documents in the FAISS store. "
                "VERIFIED ≥ 0.62 similarity, PARTIAL ≥ 0.45, UNVERIFIED below 0.45."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=4,
            memory=False,
        )
    return _verify_agent_instance
