"""
OrthoLink CAPA Agent — Corrective and Preventive Action Agent
CrewAI Agent: role='Lead CAPA Quality Engineer'
goal='Draft CAPA plans citing FDA 21 CFR 820.100 + ISO 13485 §8.5.2'

Tools: search_vector_store_tool (FAISS search for CAPA, QMS, and corrective action regs).
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for CAPA-related regulations (21 CFR 820.100, ISO 13485, QMS)."""
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


_capa_agent_instance: Optional[Agent] = None


def get_capa_agent() -> Agent:
    """Return the CAPA CrewAI agent (lazy-initialized)."""
    global _capa_agent_instance
    if _capa_agent_instance is None:
        _capa_agent_instance = Agent(
            role="Lead CAPA Quality Engineer",
            goal=(
                "Draft CAPA plans citing FDA 21 CFR 820.100 and ISO 13485:2016 §8.5.2. "
                "Every corrective action must reference the specific regulatory clause that mandates it. "
                "Root cause analysis must follow systematic methodology."
            ),
            backstory=(
                "Senior quality engineer with 15 years in medical device CAPA management. "
                "Expert in FDA 21 CFR 820.100 (corrective and preventive action), "
                "ISO 13485:2016 §8.5.2 (corrective action) and §8.5.3 (preventive action), "
                "and EU MDR post-market surveillance requirements. "
                "Trained in 5 Whys, Ishikawa, and FMEA root cause analysis. "
                "Always cites specific regulatory sections and never fabricates citations."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=5,
            memory=False,
        )
    return _capa_agent_instance
