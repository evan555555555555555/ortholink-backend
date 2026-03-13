"""
OrthoLink Briefing Agent — Daily Regulatory Intelligence Brief
CrewAI Agent stub: role='Regulatory Intelligence Analyst'

Production uses no-LLM data aggregation (APScheduler cron).
This agent definition exists for CrewAI consistency but the production
briefing pipeline does not use CrewAI — it aggregates alerts, FAISS coverage,
and quality metrics without LLM calls.
"""

import logging
from typing import Optional

from crewai import Agent
from crewai.tools import tool

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory intelligence data."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=10)
    return str([
        {
            "text": r.get("text", "")[:500],
            "regulation_name": r.get("regulation_name"),
            "document_id": r.get("document_id"),
            "country": r.get("country"),
        }
        for r in results
    ])


_briefing_agent_instance: Optional[Agent] = None


def get_briefing_agent() -> Agent:
    """Return the Briefing CrewAI agent (lazy-initialized).

    Note: Production briefing pipeline uses no-LLM data aggregation.
    This agent is defined for CrewAI consistency and potential future use.
    """
    global _briefing_agent_instance
    if _briefing_agent_instance is None:
        _briefing_agent_instance = Agent(
            role="Regulatory Intelligence Analyst",
            goal=(
                "Generate daily regulatory intelligence briefs by aggregating "
                "alerts, FAISS coverage audits, and quality metrics. "
                "Identify regulatory drift and coverage gaps across all monitored markets."
            ),
            backstory=(
                "Regulatory intelligence analyst who monitors 15+ medical device markets daily. "
                "Aggregates FDA recalls, MAUDE adverse events, EU DHPCs, and regulatory changes. "
                "Computes FAISS coverage metrics per country and identifies remediation priorities. "
                "Uses ComplianceAuditor + Reality Checker patterns for quality assurance."
            ),
            tools=[search_vector_store_tool],
            llm="gpt-4o",
            verbose=False,
            max_iter=4,
            memory=False,
        )
    return _briefing_agent_instance
