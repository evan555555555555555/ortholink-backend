"""
OrthoLink PMS Agent — Post-Market Surveillance Agent
CrewAI Agent: role='Post-Market Surveillance Scientist'
goal='Design PMS plans per EU MDR Article 84 and country-specific requirements'

Tools: search_vector_store_tool (FAISS search for PMS, vigilance, and surveillance regs).
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for post-market surveillance regulations (EU MDR Art 83-86, FDA)."""
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


_pms_agent_instance: Optional[Agent] = None


def get_pms_agent() -> Agent:
    """Return the PMS CrewAI agent (lazy-initialized)."""
    global _pms_agent_instance
    if _pms_agent_instance is None:
        _pms_agent_instance = Agent(
            role="Post-Market Surveillance Scientist",
            goal=(
                "Design PMS plans per EU MDR Article 84 and country-specific post-market requirements. "
                "Include PMSR/PSUR timelines, vigilance reporting thresholds, and trend analysis methodologies. "
                "Every plan element must cite the specific regulatory article that mandates it."
            ),
            backstory=(
                "Post-market surveillance specialist with 10 years experience in medical device vigilance. "
                "Expert in EU MDR Articles 83-86 (PMS system), Article 87-92 (vigilance), "
                "FDA 21 CFR 803 (MDR), and IMDRF NCAR guidance. "
                "Designs PMS plans that include proactive and reactive surveillance activities, "
                "PSUR/PMSR report schedules per device class, and trend analysis per Article 88. "
                "Never fabricates reporting thresholds; grounds all requirements in FAISS regulatory data."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=5,
            memory=False,
        )
    return _pms_agent_instance
