"""
OrthoLink DVA Agent — Distributor Verification Agent ("The Sentinel")
CrewAI Agent: role='Distributor Verification Specialist'
goal='Detect fraudulent document requests with legal citations'
process=Sequential

7-Step Pipeline:
1. Parse CSV → extract distributor items
2. Embed each item with text-embedding-3-large
3. For each item: search FAISS for matching regulatory requirements (country-filtered)
4. Compute cosine similarity between item and top matches
5. LLM classification with structured output
6. Confidence gating (HC-4: < 0.7 → refusal)
7. Aggregate results → GapAnalysisReport
"""

import logging

from crewai import Agent
from crewai.tools import tool

from typing import Optional

from app.tools.embeddings import embed_text
from app.tools.llm import classify_item
from app.tools.similarity import semantic_match
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@tool
def search_vector_store_tool(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory requirements matching the query. Returns top matches with regulation names, articles, and text excerpts."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=20)
    if not results:
        return "No regulatory matches found for this query and country."
    formatted = []
    for i, r in enumerate(results[:15]):
        formatted.append(
            f"[{i+1}] {r.get('regulation_name', 'Unknown')} — {r.get('article', '')}\n"
            f"    Score: {r.get('score', 0):.4f}\n"
            f"    Text: {r.get('text', '')[:300]}"
        )
    return f"Found {len(results)} matches for '{query}' in {country}:\n\n" + "\n\n".join(formatted)


@tool
def compute_similarity_tool(item_text: str, regulation_text: str) -> str:
    """Compute cosine similarity between a distributor item and a regulation."""
    item_emb = embed_text(item_text)
    reg_emb = embed_text(regulation_text)
    score = semantic_match(item_emb, reg_emb)
    return str(score)


@tool
def classify_status_tool(
    distributor_item: str,
    matched_regulation: str,
    regulation_text: str,
    country: str = "",
    device_class: str = "",
) -> str:
    """LLM classification: REQUIRED | EXTRA | MISSING | OPTIONAL with citation."""
    return str(
        classify_item(
            distributor_item,
            regulation_text,
            country=country or "UA",
            device_class=device_class or "IIb",
        )
    )


_dva_agent_instance: Optional[Agent] = None


def get_dva_agent() -> Agent:
    """Return the DVA CrewAI agent (lazy-initialized so tests can set OPENAI_API_KEY first)."""
    global _dva_agent_instance
    if _dva_agent_instance is None:
        _dva_agent_instance = Agent(
            role="Distributor Verification Specialist",
            goal="Detect fraudulent document requests with precise legal citations",
            backstory=(
                "Senior regulatory affairs auditor with 15 years experience across "
                "50+ countries. Expert in identifying when distributors pad document "
                "lists with unnecessary requirements. Never fabricates citations. "
                "Always cites the specific Article and Clause number."
            ),
            tools=[search_vector_store_tool, compute_similarity_tool, classify_status_tool],
            llm="gpt-4o",
            verbose=True,
            max_iter=5,
            memory=False,
        )
    return _dva_agent_instance


# For backwards compatibility: use get_dva_agent() in crew code.
dva_agent = None  # Set at runtime via get_dva_agent()
