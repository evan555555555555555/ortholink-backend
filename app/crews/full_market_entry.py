"""
OrthoLink ROA Crew — Full Market Entry Plan
POST /api/v1/prep-submission (called via submission route)

PRD §4.4: Full Market Entry extends the role-split checklist with:
  - Pre-submission meeting requirements
  - Technical File / eSTAR section checklist
  - Clinical evaluation requirements
  - Post-market surveillance obligations
  - Country-specific apostille / notarisation requirements
  - Estimated total timeline (Gantt-ready: phase → days)

Uses a sequential 2-agent CrewAI Crew:
  1. Requirements Analyst  — retrieves all relevant regulatory chunks per country
  2. Plan Compiler         — synthesises into a structured FullMarketEntryPlan
"""

import json
import logging
import uuid
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from pydantic import BaseModel, Field

from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Reference tool only. Verify with official sources. "
    "This plan does not constitute legal or regulatory advice."
)


# ─────────────────────────────────────────────────────────────────────────────
# Output models
# ─────────────────────────────────────────────────────────────────────────────

class PhaseItem(BaseModel):
    phase: str = Field(..., description="Phase name (e.g. 'Pre-submission', 'Technical File')")
    tasks: list[str] = Field(default_factory=list)
    duration_days: int = Field(default=30)
    role: str = Field(default="MANUFACTURER", description="MANUFACTURER | IMPORTER | BOTH")
    regulation_cite: str = Field(default="")
    apostille_required: bool = Field(default=False)
    notes: str = Field(default="")


class FullMarketEntryPlan(BaseModel):
    """Full market entry plan — PRD ROA extended output."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str = ""
    device_class: str = ""
    device_type: Optional[str] = None
    total_timeline_days: int = Field(default=0, description="Sum of all phase durations")
    phases: list[PhaseItem] = Field(default_factory=list)
    clinical_evaluation_required: bool = False
    post_market_surveillance_required: bool = True
    pre_submission_meeting_recommended: bool = False
    technical_file_sections: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_requirements(query: str, country: str, device_class: str = "") -> str:
    """Search FAISS for regulatory requirements. Use top_k=20 for broad coverage."""
    store = get_vector_store()
    results = store.search(query, country=country, device_class=device_class or None, top_k=20)
    return json.dumps([
        {
            "text": r.get("text", "")[:600],
            "regulation_name": r.get("regulation_name", ""),
            "article": r.get("article", ""),
            "section_path": r.get("section_path", ""),
        }
        for r in results
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Crew builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_entry_crew(
    country: str,
    device_class: str,
    device_type: Optional[str],
) -> Crew:

    device_desc = f"Class {device_class}" + (f" {device_type}" if device_type else "")

    requirements_agent = Agent(
        role="Regulatory Requirements Analyst",
        goal=(
            f"Retrieve ALL regulatory requirements for a {device_desc} device "
            f"entering {country}: pre-submission, technical file, clinical evaluation, "
            "post-market surveillance, labelling, UDI, apostille."
        ),
        backstory=(
            "Expert in FDA, EU MDR, CDSCO, and global regulatory requirements for "
            "medical device market entry. You query the vector store exhaustively and "
            "return structured findings."
        ),
        tools=[search_requirements],
        llm="gpt-4o",
        max_iter=10,
        verbose=True,
    )

    plan_compiler_agent = Agent(
        role="Market Entry Plan Compiler",
        goal=(
            f"Compile a complete FullMarketEntryPlan for {country} {device_desc} "
            "from the analyst's findings. Output only valid JSON."
        ),
        backstory=(
            "You synthesise regulatory findings into actionable, phased market entry plans "
            "for medical device compliance teams. You output only valid JSON matching the "
            "FullMarketEntryPlan schema."
        ),
        tools=[],
        llm="gpt-4o",
        max_iter=8,
        verbose=True,
    )

    task_research = Task(
        description=(
            f"Search the vector store for country={country}, device_class={device_class}.\n"
            "Run searches for at minimum these topics:\n"
            "1. registration requirements manufacturer importer\n"
            "2. technical file documentation requirements\n"
            "3. clinical evaluation clinical data requirements\n"
            "4. post-market surveillance requirements\n"
            "5. UDI unique device identification\n"
            "6. labelling instructions for use\n"
            "7. pre-submission meeting\n"
            "8. apostille notarisation\n"
            "Return all relevant snippets with citations."
        ),
        expected_output="List of regulatory requirements with citation for each topic.",
        agent=requirements_agent,
    )

    task_compile = Task(
        description=(
            "Using the research from the previous task, compile a FullMarketEntryPlan.\n"
            "Output ONLY this JSON structure:\n"
            "{\n"
            '  "country": "...",\n'
            '  "device_class": "...",\n'
            '  "total_timeline_days": <int>,\n'
            '  "clinical_evaluation_required": <bool>,\n'
            '  "post_market_surveillance_required": <bool>,\n'
            '  "pre_submission_meeting_recommended": <bool>,\n'
            '  "technical_file_sections": ["...", ...],\n'
            '  "phases": [\n'
            '    {"phase": "Pre-submission", "tasks": ["..."], "duration_days": 30, '
            '"role": "MANUFACTURER", "regulation_cite": "...", "apostille_required": false, "notes": "..."},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "Include at least 5 phases. Total_timeline_days = sum of all phase durations. "
            "No markdown, only raw JSON."
        ),
        expected_output="Valid JSON matching the FullMarketEntryPlan structure.",
        agent=plan_compiler_agent,
        context=[task_research],
    )

    return Crew(
        agents=[requirements_agent, plan_compiler_agent],
        tasks=[task_research, task_compile],
        process=Process.sequential,
        verbose=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_market_entry(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> FullMarketEntryPlan:
    """
    Run the full market entry planning crew for a country + device.

    Falls back to a minimal plan with known regulatory phases if the crew
    fails or returns unparseable output.
    """
    try:
        crew = _build_full_entry_crew(country, device_class, device_type)
        crew_output = crew.kickoff(inputs={
            "country": country,
            "device_class": device_class,
            "device_type": device_type or "",
        })

        tasks_output = getattr(crew_output, "tasks_output", None) or []
        raw = ""
        if tasks_output:
            raw = getattr(tasks_output[-1], "raw", "") or str(tasks_output[-1])

        # Parse JSON from compiler output via universal manifold
        from app.crews.utils import extract_clean_json
        clean = extract_clean_json(raw)
        if clean:
            data = json.loads(clean)
            phases = [
                PhaseItem(
                    phase=p.get("phase", ""),
                    tasks=p.get("tasks", []),
                    duration_days=int(p.get("duration_days", 30)),
                    role=(p.get("role") or "BOTH").upper(),
                    regulation_cite=p.get("regulation_cite", ""),
                    apostille_required=bool(p.get("apostille_required", False)),
                    notes=p.get("notes", ""),
                )
                for p in data.get("phases", [])
            ]
            total = sum(p.duration_days for p in phases)
            return FullMarketEntryPlan(
                job_id=str(uuid.uuid4()),
                country=country,
                device_class=device_class,
                device_type=device_type,
                total_timeline_days=int(data.get("total_timeline_days", total)),
                phases=phases,
                clinical_evaluation_required=bool(data.get("clinical_evaluation_required", False)),
                post_market_surveillance_required=bool(data.get("post_market_surveillance_required", True)),
                pre_submission_meeting_recommended=bool(data.get("pre_submission_meeting_recommended", False)),
                technical_file_sections=data.get("technical_file_sections", []),
                disclaimer=DISCLAIMER,
            )

    except Exception as e:
        logger.warning(f"full_market_entry crew failed ({e}); returning fallback plan.")

    # Fallback: minimal plan from known regulatory phases
    fallback_phases = [
        PhaseItem(
            phase="Pre-submission preparation",
            tasks=["Device classification", "Regulatory pathway selection", "Gap assessment"],
            duration_days=30,
            role="MANUFACTURER",
            regulation_cite=f"See {country} regulatory authority guidelines",
            notes="Confirm pathway with regulatory affairs consultant.",
        ),
        PhaseItem(
            phase="Technical File / eSTAR compilation",
            tasks=["Design dossier", "Risk management (ISO 14971)", "Clinical evaluation report"],
            duration_days=90,
            role="MANUFACTURER",
            regulation_cite=f"See {country} technical documentation requirements",
            apostille_required=country.upper() not in ("US", "EU", "UK"),
        ),
        PhaseItem(
            phase="Regulatory submission",
            tasks=["Submit application", "Pay agency fees", "Respond to queries"],
            duration_days=60,
            role="BOTH",
            regulation_cite=f"See {country} registration authority",
        ),
        PhaseItem(
            phase="Registration / Approval",
            tasks=["Await regulatory decision", "Receive registration number"],
            duration_days=120,
            role="BOTH",
        ),
        PhaseItem(
            phase="Post-market surveillance setup",
            tasks=["PMS plan", "Complaint handling", "Annual safety reports"],
            duration_days=30,
            role="MANUFACTURER",
            notes="Post-market surveillance is mandatory for all device classes.",
        ),
    ]
    total = sum(p.duration_days for p in fallback_phases)
    return FullMarketEntryPlan(
        job_id=str(uuid.uuid4()),
        country=country,
        device_class=device_class,
        device_type=device_type,
        total_timeline_days=total,
        phases=fallback_phases,
        clinical_evaluation_required=device_class.upper() in ("III", "IIB", "IIb"),
        post_market_surveillance_required=True,
        pre_submission_meeting_recommended=country.upper() in ("US", "JP", "CN"),
        technical_file_sections=[
            "Device description and specification",
            "Design and manufacturing information",
            "Essential safety and performance requirements",
            "Benefit-risk analysis and risk management",
            "Product verification and validation",
            "Post-market information",
        ],
        disclaimer=DISCLAIMER,
    )
