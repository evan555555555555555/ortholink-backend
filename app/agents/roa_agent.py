"""
OrthoLink ROA Agent — Regulatory Operations Agent ("The Navigator")
CrewAI Agent: role='Regulatory Operations Manager'
goal='Generate role-split compliance checklists with legal citations'

ChecklistItem: item, role (MANUFACTURER|IMPORTER|BOTH), regulation_cite, deadline_days, apostille_required, notes.
PRD: Manufacturer ∩ Importer = ∅; QMSR for US (Feb 2026); UDI per country; PDF via WeasyPrint.
"""

import logging
from enum import Enum
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class ChecklistRole(str, Enum):
    MANUFACTURER = "MANUFACTURER"
    IMPORTER = "IMPORTER"
    EXPORTER = "EXPORTER"
    DISTRIBUTOR = "DISTRIBUTOR"
    BOTH = "BOTH"


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory requirements. Returns regulation names, articles, and text excerpts for checklist generation."""
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


_roa_agent_instance: Optional[Agent] = None


def get_roa_agent() -> Agent:
    """Return the ROA CrewAI agent (lazy-initialized)."""
    global _roa_agent_instance
    if _roa_agent_instance is None:
        _roa_agent_instance = Agent(
            role="Regulatory Operations Manager",
            goal="Generate role-split compliance checklists with exact legal citations; Manufacturer and Importer lists must not overlap",
            backstory=(
                "Expert in FDA 21 CFR, QMSR (Feb 2026), EU MDR, and 15-country requirements. "
                "Every checklist item cites specific article/section. Includes UDI and QMSR for US."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=5,
            memory=False,
        )
    return _roa_agent_instance
