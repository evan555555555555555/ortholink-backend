"""
OrthoLink CRA Agent — Compliance Review Agent ("The Reviewer")
CrewAI Agent: role='Medical Device Compliance Auditor'
goal='Stream compliance reviews grounded in regulatory text'
process=Sequential

Tools: search_vector_store_tool (reuse), extract_clauses_tool, compare_to_standard_tool.
PRD: RAG-grounded prompts; confidence < 0.7 → structured refusal; all output includes disclaimer.
"""

import logging
import re
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = "Reference tool only. Verify with official sources."


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory requirements matching the query. Use for RAG grounding."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=20)
    return str([{"text": r.get("text", ""), "citation": r.get("section_path") or r.get("document_id"), "regulation_name": r.get("regulation_name")} for r in results])


@tool
def extract_clauses_tool(document_text: str) -> str:
    """Parse uploaded document into clauses/sections for review. Splits by Article/Section/paragraph boundaries."""
    if not document_text or not document_text.strip():
        return "[]"
    text = document_text.strip()
    # Split by common regulatory patterns: Article N, Section N, or double newline
    parts = re.split(r"\n\s*\n+|\bArticle\s+\d+|\bSection\s+\d+", text, flags=re.IGNORECASE)
    clauses = [p.strip() for p in parts if len(p.strip()) > 50]
    if not clauses:
        clauses = [text[:4000]] if len(text) > 4000 else [text]
    return str(clauses[:50])  # Cap at 50 for review


@tool
def compare_to_standard_tool(
    clause_text: str,
    standard_reference: str,
    country: str,
    regulation_snippets: str,
) -> str:
    """Compare a document clause to regulatory standard. Returns JSON: severity (CRITICAL|MAJOR|MINOR|INFO), citation, suggestion, standard_reference."""
    from app.tools.llm import compare_clause_to_standard

    result = compare_clause_to_standard(
        clause_text=clause_text,
        standard_reference=standard_reference,
        country=country,
        regulation_snippets=regulation_snippets,
    )
    return str(result)


_cra_agent_instance: Optional[Agent] = None


def get_cra_agent() -> Agent:
    """Return the CRA CrewAI agent (lazy-initialized)."""
    global _cra_agent_instance
    if _cra_agent_instance is None:
        _cra_agent_instance = Agent(
            role="Medical Device Compliance Auditor",
            goal="Stream compliance reviews grounded in regulatory text; cite specific clauses",
            backstory=(
                "Senior compliance auditor. Reviews documents against ISO 13485, "
                "FDA 21 CFR 820, EU MDR, and country-specific regulations. Never hardcodes "
                "classifications; always grounds findings in retrieved regulation text."
            ),
            tools=[search_vector_store_tool, extract_clauses_tool, compare_to_standard_tool],
            llm="gpt-4o",
            verbose=True,
            max_iter=5,
            memory=False,
        )
    return _cra_agent_instance
