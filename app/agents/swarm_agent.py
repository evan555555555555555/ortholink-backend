"""
OrthoLink GCO Agent — Global Compliance Orchestrator
CrewAI Agent stub: role='Chief Regulatory Officer'

Production uses ThreadPoolExecutor (max_workers=4) to run TDA+PMS+ROA+CAPA in parallel.
This agent definition exists for CrewAI consistency but the production
gco_analysis.py pipeline orchestrates sub-agents via ThreadPoolExecutor,
not CrewAI hierarchical process.
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for cross-cutting regulatory requirements."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=15)
    return str([
        {
            "text": r.get("text", "")[:500],
            "regulation_name": r.get("regulation_name"),
            "article": r.get("article"),
            "document_id": r.get("document_id"),
        }
        for r in results
    ])


_gco_agent_instance: Optional[Agent] = None


def get_gco_agent() -> Agent:
    """Return the GCO CrewAI agent (lazy-initialized).

    Note: Production gco_analysis.py uses ThreadPoolExecutor for parallel
    execution of TDA+PMS+ROA+CAPA sub-agents. This agent is the orchestrator
    definition for CrewAI consistency.
    """
    global _gco_agent_instance
    if _gco_agent_instance is None:
        _gco_agent_instance = Agent(
            role="Chief Regulatory Officer",
            goal=(
                "Orchestrate multi-agent regulatory analysis by coordinating "
                "TDA, PMS, ROA, and CAPA sub-agents for comprehensive compliance assessment. "
                "Synthesize findings across all four regulatory domains."
            ),
            backstory=(
                "Chief Regulatory Officer overseeing global medical device compliance strategy. "
                "Coordinates parallel regulatory assessments: technical documentation (TDA), "
                "post-market surveillance (PMS), regulatory operations (ROA), and CAPA quality. "
                "Synthesizes cross-cutting findings and identifies systemic compliance gaps."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=6,
            memory=False,
        )
    return _gco_agent_instance
