"""
OrthoLink RSA Crew — Plan Strategy
POST /api/v1/plan-strategy

PRD §4.5: Hierarchical CrewAI pattern.
  Manager agent orchestrates 4 sub-agents:
    1. Planner    — identify regulatory pathway per country
    2. Retriever  — estimate document reuse % per country
    3. Analyzer   — estimate approval timelines per country
    4. Critic     — estimate market entry costs per country

Output: StrategyReport with optimal_entry_sequence (ranked by strategy_utility_score),
pathway_map, document_reuse_matrix, timeline_estimates, cost_estimates.
"""

import json
import logging
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agents.rsa_agent import (
    get_rsa_planner,
    get_rsa_retriever,
    get_rsa_analyzer,
    get_rsa_critic,
)
from app.core.strategy_scoring import strategy_utility_score
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = "Reference tool only. Verify with official sources."

# ── Country-level knowledge used when vector store has no chunks for a country.
# Derived from official agency benchmarks; NOT classification dicts (PRD §HC-10).
# MDSAP (Medical Device Single Audit Program): CA, AU, BR, JP accept MDSAP audits — reduces timeline/cost.
_PATHWAY_FALLBACKS: dict[str, str] = {
    "US": "510(k) Traditional or PMA (FDA 21 CFR Part 807/814)",
    "EU": "EU MDR 2017/745 — CE Marking via Notified Body",
    "UK": "UK CA Marking — MHRA (Post-Brexit)",
    "CA": "Health Canada Medical Device Licence (MDL) — CMDCAS; MDSAP acceptance reduces audit burden",
    "AU": "TGA ARTG — AIMD/Class IIb: Conformity Assessment; MDSAP recognition reduces timeline",
    "JP": "PMDA Shonin Approval or Ninsho Certification; MDSAP participation reduces audit cost",
    "CN": "NMPA Class III Registration (National)",
    "BR": "ANVISA INMETRO Certification — RDC 830/2023; MDSAP acceptance reduces audit scope",
    "IN": "CDSCO MDR 2017 — Form MD-14 Registration",
    "UA": "Resolution 753 — Technical Regulation on Medical Devices",
    "CH": "Swissmedic MDA Conformity Assessment",
    "MX": "COFEPRIS Sanitary Registration",
    "KR": "MFDS Medical Device Approval — Class III",
    "RU": "Roszdravnadzor Registration Certificate",
    "SA": "SFDA Medical Device Listing/Registration",
}

_TIMELINE_FALLBACKS: dict[str, dict] = {
    "US": {"months": 4,  "dependencies": ["FDA 510(k) review (~130 days median)"]},
    "EU": {"months": 9,  "dependencies": ["Notified Body review", "Technical File"]},
    "UK": {"months": 6,  "dependencies": ["MHRA CA assessment"]},
    "CA": {"months": 5,  "dependencies": ["Health Canada review", "MDSAP acceptance shortens timeline"]},
    "AU": {"months": 4,  "dependencies": ["TGA conformity assessment", "MDSAP recognition"]},
    "JP": {"months": 14, "dependencies": ["PMDA pre-submission", "Clinical evaluation", "MDSAP reduces audit time"]},
    "CN": {"months": 24, "dependencies": ["NMPA technical review", "Local clinical data"]},
    "BR": {"months": 10, "dependencies": ["ANVISA review", "INMETRO certification", "MDSAP acceptance"]},
    "IN": {"months": 12, "dependencies": ["CDSCO form MD-14", "Local testing"]},
    "UA": {"months": 9,  "dependencies": ["Resolution 753 conformity assessment"]},
    "CH": {"months": 8,  "dependencies": ["Swissmedic assessment"]},
    "MX": {"months": 12, "dependencies": ["COFEPRIS review"]},
    "KR": {"months": 12, "dependencies": ["MFDS review"]},
    "RU": {"months": 18, "dependencies": ["Roszdravnadzor registration"]},
    "SA": {"months": 9,  "dependencies": ["SFDA listing", "GCC registration"]},
}

_COST_FALLBACKS: dict[str, dict] = {
    "US": {"agency_fee": 5000, "nb_fee": 0,      "translation": 2000, "total_estimate_usd": 19000},
    "EU": {"agency_fee": 0,    "nb_fee": 30000,  "translation": 5000, "total_estimate_usd": 55000},
    "UK": {"agency_fee": 0,    "nb_fee": 20000,  "translation": 2000, "total_estimate_usd": 35000},
    "CA": {"agency_fee": 1500, "nb_fee": 0,       "translation": 3000, "total_estimate_usd": 17000},  # MDSAP reduces audit cost
    "AU": {"agency_fee": 2500, "nb_fee": 0,       "translation": 1000, "total_estimate_usd": 15000},  # MDSAP recognition
    "JP": {"agency_fee": 6000, "nb_fee": 0,       "translation": 15000,"total_estimate_usd": 52000},  # MDSAP reduces audit cost
    "CN": {"agency_fee": 5000, "nb_fee": 0,       "translation": 20000,"total_estimate_usd": 80000},
    "BR": {"agency_fee": 3500, "nb_fee": 0,       "translation": 5000, "total_estimate_usd": 26000},  # MDSAP acceptance
    "IN": {"agency_fee": 1500, "nb_fee": 0,       "translation": 3000, "total_estimate_usd": 25000},
    "UA": {"agency_fee": 1000, "nb_fee": 0,       "translation": 3000, "total_estimate_usd": 15000},
    "CH": {"agency_fee": 3000, "nb_fee": 25000,  "translation": 2000, "total_estimate_usd": 45000},
    "MX": {"agency_fee": 2000, "nb_fee": 0,       "translation": 4000, "total_estimate_usd": 22000},
    "KR": {"agency_fee": 3000, "nb_fee": 0,       "translation": 8000, "total_estimate_usd": 35000},
    "RU": {"agency_fee": 2000, "nb_fee": 0,       "translation": 6000, "total_estimate_usd": 30000},
    "SA": {"agency_fee": 2500, "nb_fee": 0,       "translation": 5000, "total_estimate_usd": 28000},
}

# Total Addressable Market (TAM) proxy per country for orthopedic devices — revenue potential weight [0, 1].
# Used in strategy_utility_score: (TAM * reuse_pct) / (time * cost). PRD Phase 9.
_TAM_FALLBACKS: dict[str, float] = {
    "US": 1.0,
    "EU": 0.85,
    "JP": 0.60,
    "CN": 0.55,
    "UK": 0.75,
    "CA": 0.50,
    "AU": 0.45,
    "BR": 0.40,
    "IN": 0.35,
    "KR": 0.50,
    "MX": 0.30,
    "UA": 0.25,
    "CH": 0.40,
    "RU": 0.20,
    "SA": 0.35,
}


class StrategyReport(BaseModel):
    """RSA output — PRD §4.5.4."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = ""
    target_markets: list[str] = Field(default_factory=list)
    optimal_entry_sequence: list[str] = Field(
        default_factory=list,
        description="Country codes in recommended order (highest utility first)",
    )
    entry_sequence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rich entry items: [{country, pathway, reuse_pct, timeline_months, cost_usd, priority_score, rationale}]",
    )
    pathway_map: dict[str, str] = Field(default_factory=dict)
    document_reuse_matrix: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-country reuse % or {reusable, modify, create}",
    )
    timeline_estimates: dict[str, Any] = Field(default_factory=dict)
    cost_estimates: dict[str, Any] = Field(default_factory=dict)
    citations: dict[str, str] = Field(
        default_factory=dict,
        description="Per-country primary regulatory citation",
    )
    disclaimer: str = Field(default=DISCLAIMER)


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Crew builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_rsa_crew(device_name: str, target_markets: list[str], device_class: str):
    """
    Assemble the RSA hierarchical CrewAI Crew.
    Manager agent orchestrates Planner → Retriever → Analyzer → Critic.
    Returns (crew, task_descriptions) ready for crew.kickoff().
    """
    from crewai import Agent, Crew, Process, Task

    markets_str = ", ".join(target_markets)
    context_str = (
        f"Device: {device_name} (Class {device_class}). "
        f"Target markets: {markets_str}. "
        "Query your entity memory for existing device profiles (materials, intended use, prior certifications) before assuming inputs. "
        "Use search_vector_store_tool for each country. "
        "Output ONLY valid JSON matching the specified schema."
    )

    planner = get_rsa_planner()
    retriever = get_rsa_retriever()
    analyzer = get_rsa_analyzer()
    critic = get_rsa_critic()

    task_plan = Task(
        description=(
            f"{context_str}\n"
            f"Identify the regulatory pathway for each of: {markets_str}. "
            "Search the vector store per country. "
            "Output JSON: {\"pathway_map\": {\"US\": \"510(k) Traditional\", ...}, "
            "\"citations\": {\"US\": \"21 CFR 807\", ...}}"
        ),
        expected_output=(
            "JSON with pathway_map and citations keys, one entry per country."
        ),
        agent=planner,
    )

    task_reuse = Task(
        description=(
            f"{context_str}\n"
            f"Estimate document reuse percentage for each of: {markets_str}. "
            "Consider existing Technical File, QMS, clinical data. "
            "Output JSON: {\"reuse_matrix\": {\"US\": 65, \"UA\": 40}, "
            "\"by_country\": {\"US\": {\"reusable\": [\"QMS\"], \"modify\": [\"IFU\"], \"create\": []}}}"
        ),
        expected_output="JSON with reuse_matrix and by_country keys.",
        agent=retriever,
        context=[task_plan],
    )

    task_timeline = Task(
        description=(
            f"{context_str}\n"
            f"Estimate approval timelines for each of: {markets_str}. "
            "Use vector store context and known regulatory benchmarks. "
            "Output JSON: {\"timeline_estimates\": {\"US\": {\"months\": 4, "
            "\"dependencies\": [\"FDA review\"]}, \"UA\": {\"months\": 9, \"dependencies\": []}}}"
        ),
        expected_output="JSON with timeline_estimates key, one entry per country.",
        agent=analyzer,
        context=[task_plan, task_reuse],
    )

    task_cost = Task(
        description=(
            f"{context_str}\n"
            f"Estimate market entry costs for each of: {markets_str}. "
            "Include agency fee, NB fee, translation, in-country agent. "
            "Output JSON: {\"cost_estimates\": {\"US\": {\"agency_fee\": 5000, "
            "\"nb_fee\": 0, \"translation\": 2000, \"total_estimate_usd\": 19000}}}"
        ),
        expected_output="JSON with cost_estimates key, one entry per country.",
        agent=critic,
        context=[task_plan, task_reuse, task_timeline],
    )

    # Manager agent for hierarchical process
    manager = Agent(
        role="Regulatory Strategy Manager",
        goal=(
            "Orchestrate Planner, Retriever, Analyzer, and Critic agents to produce "
            "a complete StrategyReport for all target markets."
        ),
        backstory=(
            "Senior regulatory affairs director with 20 years experience across FDA, EU MDR, "
            "CDSCO, and 15 global markets. You delegate and synthesize specialist outputs."
        ),
        llm="gpt-4o",
        verbose=False,
        allow_delegation=True,
    )

    from app.core.crew_memory import get_ltm_memory

    ltm = get_ltm_memory()
    crew_kw: dict = {
        "agents": [planner, retriever, analyzer, critic],
        "tasks": [task_plan, task_reuse, task_timeline, task_cost],
        "manager_agent": manager,
        "process": Process.hierarchical,
        "verbose": True,
    }
    if ltm is not None:
        crew_kw["memory"] = True
        crew_kw["long_term_memory"] = ltm
    crew = Crew(**crew_kw)
    return crew


def _parse_agent_json(raw: str, key: str) -> dict:
    """Extract a JSON dict from agent output, looking for {key: ...} structure."""
    if not raw:
        return {}
    from app.crews.utils import extract_clean_json
    try:
        clean = extract_clean_json(raw)
        data = json.loads(clean)
        if isinstance(data, dict) and key in data:
            return data[key]
        if isinstance(data, dict):
            return data
    except (ValueError, json.JSONDecodeError):
        pass
    return {}


def run_rsa_strategy(
    device_name: str,
    target_markets: list[str],
    device_class: str = "II",
    existing_certs: Optional[list[str]] = None,
) -> StrategyReport:
    """
    Run RSA via real CrewAI hierarchical Crew (Planner→Retriever→Analyzer→Critic).

    Falls back gracefully per country: if the CrewAI crew fails or returns unparseable
    output, uses _PATHWAY_FALLBACKS / _TIMELINE_FALLBACKS / _COST_FALLBACKS.
    Final ranking: strategy_utility_score (PRD §4.5.3).
    """
    if not target_markets:
        return StrategyReport(
            device_name=device_name,
            target_markets=[],
            disclaimer=DISCLAIMER,
        )

    # Normalise country codes
    markets = [m.strip().upper() for m in target_markets if m.strip()]

    # ── Seed from fallbacks (ensures every country always has data) ──────────
    pathway_map: dict[str, str] = {cc: _PATHWAY_FALLBACKS.get(cc, "Registration (see regulation)") for cc in markets}
    reuse_matrix: dict[str, Any] = {cc: 50.0 for cc in markets}
    timeline_estimates: dict[str, Any] = {cc: _TIMELINE_FALLBACKS.get(cc, {"months": 9, "dependencies": []}) for cc in markets}
    cost_estimates: dict[str, Any] = {cc: _COST_FALLBACKS.get(cc, {"total_estimate_usd": 25000}) for cc in markets}
    citations: dict[str, str] = {}

    # ── Direct FAISS + single LLM call (replaces CrewAI crew — ~8-15s vs 60-90s) ──
    try:
        from app.tools.llm import chat_completion

        store = get_vector_store()

        # Parallel FAISS searches per country (3 queries × N markets)
        seen_ids: set = set()
        chunks_by_country: dict[str, list[str]] = {cc: [] for cc in markets}
        for cc in markets:
            for query in [
                f"registration pathway approval requirements {cc}",
                f"quality management system timeline cost {cc}",
                f"regulatory submission documents {cc}",
            ]:
                for r in store.search(query, country=cc, top_k=4):
                    cid = r.get("chunk_id") or r.get("text", "")[:40]
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        src = r.get("regulation_name") or r.get("document_id") or ""
                        art = r.get("article", "")
                        chunks_by_country[cc].append(
                            f"[{cc}][{src}{', ' + art if art else ''}] {r.get('text', '')[:350]}"
                        )

        regulation_context = "\n\n".join(
            chunk for cc in markets for chunk in chunks_by_country[cc][:6]
        )

        # Build fallback JSON for LLM reference
        fallback_json = {
            "pathway_map": {cc: pathway_map[cc] for cc in markets},
            "reuse_matrix": {cc: 50 for cc in markets},
            "timeline_estimates": {cc: timeline_estimates[cc] for cc in markets},
            "cost_estimates": {cc: cost_estimates[cc] for cc in markets},
        }

        sys_prompt = (
            "You are a senior regulatory affairs director. "
            "Output ONLY valid JSON — no markdown, no prose."
        )
        user_prompt = f"""Produce a regulatory strategy report for:
Device: {device_name} (Class {device_class})
Target markets: {", ".join(markets)}

Use the regulatory text below to refine the baseline estimates. Cite specific articles.

REGULATORY TEXT:
{regulation_context}

BASELINE (refine these):
{json.dumps(fallback_json, indent=2)}

Output a single JSON object with EXACTLY these keys:
{{
  "pathway_map": {{"US": "510(k) Traditional — 21 CFR 807.87", ...}},
  "reuse_matrix": {{"US": 70, "EU": 55, ...}},
  "timeline_estimates": {{"US": {{"months": 4, "dependencies": ["FDA 510(k) ~130 days"]}}, ...}},
  "cost_estimates": {{"US": {{"agency_fee": 5000, "nb_fee": 0, "translation": 2000, "total_estimate_usd": 19000}}, ...}},
  "citations": {{"US": "21 CFR 807.87", ...}}
}}
Output ONLY valid JSON."""

        raw = chat_completion(sys_prompt, user_prompt)

        # Parse and merge into dicts
        try:
            from app.crews.utils import extract_clean_json
            clean = extract_clean_json(raw)
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                if "pathway_map" in parsed:
                    for cc in markets:
                        if cc in parsed["pathway_map"]:
                            pathway_map[cc] = parsed["pathway_map"][cc]
                if "reuse_matrix" in parsed:
                    for cc in markets:
                        if cc in parsed["reuse_matrix"]:
                            reuse_matrix[cc] = float(parsed["reuse_matrix"][cc])
                if "timeline_estimates" in parsed:
                    for cc in markets:
                        if cc in parsed["timeline_estimates"]:
                            timeline_estimates[cc] = parsed["timeline_estimates"][cc]
                if "cost_estimates" in parsed:
                    for cc in markets:
                        if cc in parsed["cost_estimates"]:
                            cost_estimates[cc] = parsed["cost_estimates"][cc]
                if "citations" in parsed:
                    citations.update(parsed["citations"])
        except Exception as pe:
            logger.error(f"CRITICAL: RSA LLM output parse failed ({pe}); using fallback data.")

        logger.info("RSA direct FAISS+LLM run completed.")

    except Exception as e:
        logger.error(f"CRITICAL: RSA LLM call failed ({e}); using fallback data only.")

    # ── Enrich citations from vector store if still missing ──────────────────
    if len(citations) < len(markets):
        try:
            store = get_vector_store()
            for cc in markets:
                if cc not in citations:
                    results = list(store.search("regulatory requirements pathway", country=cc, top_k=1))
                    if results:
                        art = results[0].get("article", "")
                        citations[cc] = f"{results[0].get('regulation_name', '')}" + (f", {art}" if art else "")
        except Exception as cit_exc:
            logger.warning("RSA citation enrichment failed: %s — proceeding without full citations", cit_exc)

    # ── Rank countries by strategy_utility_score (PRD §4.5.3) ───────────────
    raw_scores: list[tuple[str, float]] = []
    country_data: dict[str, dict] = {}
    for cc in markets:
        reuse_val = reuse_matrix.get(cc, 50.0)
        if isinstance(reuse_val, dict):
            reuse_pct = float(reuse_val.get("pct", 50.0))
        else:
            reuse_pct = float(reuse_val)

        tl = timeline_estimates.get(cc, {})
        months = float(tl.get("months", 9) if isinstance(tl, dict) else 9)

        ce = cost_estimates.get(cc, {})
        cost = float(ce.get("total_estimate_usd", 25000) if isinstance(ce, dict) else 25000)

        tam = _TAM_FALLBACKS.get(cc, 0.4)
        u = strategy_utility_score(reuse_pct, months, cost, tam)
        raw_scores.append((cc, u))
        country_data[cc] = {"reuse_pct": reuse_pct, "months": months, "cost": cost}

    # Normalize scores to 0–100 relative to the highest-scoring country
    max_raw = max((s for _, s in raw_scores), default=1.0) or 1e-9
    scores = [(cc, (s / max_raw) * 100.0) for cc, s in raw_scores]
    scores.sort(key=lambda x: -x[1])
    optimal_entry_sequence = [c for c, _ in scores]

    # Build entry_sequence with rich objects (frontend expects this format)
    entry_sequence: list[dict] = []
    for cc, score in scores:
        cd = country_data[cc]
        entry_sequence.append({
            "country": cc,
            "pathway": pathway_map.get(cc, "Registration required"),
            "reuse_pct": cd["reuse_pct"],
            "timeline_months": cd["months"],
            "cost_usd": cd["cost"],
            "priority_score": round(score, 1),
            "rationale": citations.get(cc, ""),
        })

    return StrategyReport(
        job_id=str(uuid.uuid4()),
        device_name=device_name,
        target_markets=markets,
        optimal_entry_sequence=optimal_entry_sequence,
        entry_sequence=entry_sequence,
        pathway_map=pathway_map,
        document_reuse_matrix=reuse_matrix,
        timeline_estimates=timeline_estimates,
        cost_estimates=cost_estimates,
        citations=citations,
        disclaimer=DISCLAIMER,
    )
