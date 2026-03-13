"""
OrthoLink TDA Agent — Technical Documentation Agent
CrewAI Agent: role='Technical Documentation Specialist'
goal='Map documents to EU MDR Annex II/III structure with specific section citations'

Tools: search_vector_store_tool (FAISS search for technical documentation requirements).
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for technical documentation requirements (EU MDR Annex II/III, FDA)."""
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


_tda_agent_instance: Optional[Agent] = None


def get_tda_agent() -> Agent:
    """Return the TDA CrewAI agent (lazy-initialized)."""
    global _tda_agent_instance
    if _tda_agent_instance is None:
        _tda_agent_instance = Agent(
            role="Technical Documentation Specialist",
            goal=(
                "Map technical documentation to EU MDR 2017/745 Annex II and Annex III structure. "
                "Every section must cite the specific Annex subsection it corresponds to. "
                "Identify gaps between submitted documentation and regulatory requirements."
            ),
            backstory=(
                "Technical documentation specialist with 12 years experience preparing "
                "EU MDR Annex II (Technical Documentation) and Annex III (Post-Market Surveillance) "
                "technical files. Expert in FDA 510(k) STED format, IMDRF documentation standards, "
                "and harmonized ISO standards (ISO 13485, ISO 14971, IEC 60601). "
                "Maps every document section to the specific regulatory Annex subsection. "
                "Never fabricates document requirements; grounds all recommendations in FAISS data."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=5,
            memory=False,
        )
    return _tda_agent_instance
