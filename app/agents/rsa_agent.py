"""
OrthoLink RSA Agent — Regulatory Strategy Agent ("The Strategist")
CrewAI: 4 agents (Planner, Retriever, Analyzer, Critic) → StrategyReport.

Planner: Pathway per country (510(k), EU MDR, CDSCO, etc.)
Retriever: Document reuse % per country
Analyzer: Timeline estimates (months, dependencies)
Critic: Cost estimates (agency, NB, translation, etc.)

PRD §4.5: Hierarchical pattern; output: country sequence, reuse%, timelines, costs.
"""

import logging

from crewai import Agent
from crewai.tools import tool

from app.tools.fda_search import search_fda_predicates_tool
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory requirements. Use for pathway, timeline, cost, reuse context."""
    store = get_vector_store()
    results = store.search(
        query, country=country, device_class=device_class or None, top_k=15
    )
    return str([
        {"text": r.get("text", "")[:600], "regulation_name": r.get("regulation_name")}
        for r in results
    ])


def _make_agent(role: str, goal: str, backstory: str) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=[search_vector_store_tool, search_fda_predicates_tool],
        llm="gpt-4o",
        verbose=False,
        max_iter=6,
        memory=False,
    )


# Planner — Pathway Analyst: identify pathway per country
def get_rsa_planner() -> Agent:
    return _make_agent(
        role="Regulatory Pathway Planner",
        goal="Identify the correct regulatory pathway per country (510(k)/PMA/De Novo, EU MDR, CDSCO, etc.) with citation",
        backstory=(
            "Expert in FDA, EU MDR, UK MHRA, CDSCO, ANVISA pathways. "
            "Check entity memory for existing device profile and prior pathway decisions before requesting redundant inputs. "
            "You output only valid JSON: {\"pathway_map\": {\"US\": \"510(k) Traditional\", \"UA\": \"Resolution 753\"}, \"citations\": {\"US\": \"21 CFR 807\", ...}}"
        ),
    )


# Retriever — Reuse Analyst: document reuse % per country
def get_rsa_retriever() -> Agent:
    return _make_agent(
        role="Document Reuse Analyst",
        goal="Estimate document reuse percentage per country (reusable vs modify vs create-new) from regulatory snippets",
        backstory=(
            "You map existing Technical File / eSTAR sections to each market. "
            "Output only valid JSON: {\"reuse_matrix\": {\"US\": 65, \"UA\": 40}, \"by_country\": {\"US\": {\"reusable\": [\"QMS\"], \"modify\": [\"IFU\"]}}}"
        ),
    )


# Analyzer — Timeline Analyst: approval timelines
def get_rsa_analyzer() -> Agent:
    return _make_agent(
        role="Regulatory Timeline Analyst",
        goal="Estimate approval timelines per country (months, key dependencies). FDA ~130 days, EU NB review, etc.",
        backstory=(
            "You use regulatory snippets and known benchmarks (FDA 510(k) ~4 months, EU NB 6-12 months). "
            "Output only valid JSON: {\"timeline_estimates\": {\"US\": {\"months\": 4, \"dependencies\": []}, \"UA\": {\"months\": 9}}}"
        ),
    )


# Critic — Cost Analyst: fees and costs
def get_rsa_critic() -> Agent:
    return _make_agent(
        role="Market Entry Cost Analyst",
        goal="Estimate market entry costs per country: agency fees, NB fees, testing, translation, in-country agent",
        backstory=(
            "You estimate costs from regulatory context. "
            "Output only valid JSON: {\"cost_estimates\": {\"US\": {\"agency_fee\": 5000, \"total_estimate_usd\": 19000}, \"UA\": {\"total_estimate_usd\": 15000}}}"
        ),
    )
