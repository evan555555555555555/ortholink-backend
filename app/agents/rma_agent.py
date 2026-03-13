"""
OrthoLink RMA Agent — Risk Management Agent ("The Assessor")
CrewAI Agent: role='Principal Risk Engineer'
goal='Perform ISO 14971:2019 hazard analysis with specific clause citations'

Tools: search_vector_store_tool (FAISS search for ISO 14971 + country-specific risk regs).
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for risk management regulations (ISO 14971, IEC 62304, country regs). Returns formatted regulatory excerpts."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=20)
    if not results:
        return "No regulatory matches found for this query and country."
    formatted = []
    for i, r in enumerate(results[:15]):
        formatted.append(
            f"[{i+1}] {r.get('regulation_name', 'Unknown')} — {r.get('article', '')}\n"
            f"    Section: {r.get('section_path') or r.get('document_id', '')}\n"
            f"    Text: {r.get('text', '')[:300]}"
        )
    return f"Found {len(results)} matches for '{query}' in {country}:\n\n" + "\n\n".join(formatted)


_rma_agent_instance: Optional[Agent] = None


def get_rma_agent() -> Agent:
    """Return the RMA CrewAI agent (lazy-initialized)."""
    global _rma_agent_instance
    if _rma_agent_instance is None:
        _rma_agent_instance = Agent(
            role="Principal Risk Engineer",
            goal=(
                "Perform ISO 14971:2019 hazard analysis with specific clause citations. "
                "Every hazard must cite a specific ISO 14971 clause (e.g. §5.5, §6.2, §7.4). "
                "Risk levels are computed server-side from the 5x5 risk matrix — never override them."
            ),
            backstory=(
                "Lead risk engineer with 20 years experience in medical device risk management. "
                "Expert in ISO 14971:2019, IEC 62304, and country-specific risk requirements. "
                "Performs systematic hazard identification per ISO 14971 §5, risk estimation per §6, "
                "and risk evaluation per §7. Never fabricates risk levels; relies on server-side "
                "5x5 probability × severity matrix from ISO 14971 Annex D."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=5,
            memory=False,
        )
    return _rma_agent_instance
