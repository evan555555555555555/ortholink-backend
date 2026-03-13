"""
OrthoLink CAPA Crew — Corrective and Preventive Action Analysis
POST /api/v1/capa

Regulatory basis:
  - ISO 13485:2016 §8.5.2 (Corrective Action: review nonconformities → determine cause →
    evaluate need → implement → record → verify effectiveness)
  - ISO 13485:2016 §8.5.3 (Preventive Action: same cycle for POTENTIAL nonconformities)
  - 21 CFR 820.100 (CAPA — QMSR 2026): analyze complaints, audits, concessions, service records,
    process data, returned products; trend analysis; documented effectiveness check
  - EU MDR Article 10(9)(l): QMS CAPA procedure mandatory for CE-marked devices
  - EU MDR Article 87: regulatory notification required if CAPA relates to serious incident
  - MDSAP Chapter 6: single CAPA system covers US/CA/AU/BR/JP
"""

import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import build_regulation_context, multi_query_faiss, parse_llm_json
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "CAPA guidance is for educational purposes. All CAPA records must be reviewed "
    "by qualified QA/RA professionals and maintained per 21 CFR 820 / ISO 13485."
)


class RootCauseCategory(BaseModel):
    """A potential root cause category."""

    category: str = Field(..., description="Root cause category (e.g. Design, Process, Human Error)")
    likelihood: str = Field(..., description="High | Medium | Low")
    investigation_questions: list[str] = Field(
        default_factory=list, description="Questions to investigate this category"
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Root cause analysis tools (5-Why, Fishbone, FTA, FMEA)",
    )


class CorrectiveAction(BaseModel):
    """A single corrective or preventive action."""

    action_id: str = Field(default="", description="Unique action identifier (e.g. CA-001)")
    action_type: str = Field(..., description="Corrective | Preventive | Containment")
    description: str = Field(..., description="What needs to be done")
    responsible_department: str = Field(default="Quality")
    target_completion_days: int = Field(default=30)
    effectiveness_criteria: str = Field(
        default="", description="How to verify this action was effective"
    )
    regulation_cite: str = Field(default="")


class CAPARegObligation(BaseModel):
    """Regulatory obligation related to CAPA."""

    obligation: str = Field(..., description="What the regulation requires")
    regulation_cite: str
    timeline: str = Field(default="", description="Required timeline for completion")
    documentation_required: list[str] = Field(default_factory=list)


class CAPAAnalysis(BaseModel):
    """Complete CAPA analysis and action plan."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    problem_statement: str
    country: str
    device_class: str
    severity: str = Field(
        default="", description="Critical | Major | Minor based on patient risk"
    )
    root_cause_categories: list[RootCauseCategory] = Field(default_factory=list)
    corrective_actions: list[CorrectiveAction] = Field(default_factory=list)
    regulatory_obligations: list[CAPARegObligation] = Field(default_factory=list)
    requires_regulatory_notification: bool = Field(
        default=False,
        description="Whether this CAPA triggers regulatory reporting (MDR, field safety notice)",
    )
    notification_rationale: str = Field(default="")
    recommended_timeline_days: int = Field(default=30)
    disclaimer: str = Field(default=DISCLAIMER)
    is_fallback: bool = Field(
        default=False,
        description="True if LLM failed and fallback data was served. "
        "Frontend MUST display degraded-data warning when True.",
    )


def _run_capa_crew(
    problem_statement: str,
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> "CAPAAnalysis":
    """CrewAI path: CAPA agent runs analysis via crew.kickoff(), then structured pipeline."""
    from crewai import Crew, Process, Task

    from app.agents.capa_agent import get_capa_agent

    agent = get_capa_agent()
    task = Task(
        description=(
            f"Analyze this CAPA problem and generate a corrective/preventive action plan.\n"
            f"Problem: {problem_statement}\n"
            f"Country: {country}, Device Class: {device_class}\n"
            f"{'Device type: ' + device_type if device_type else ''}\n\n"
            "Cite 21 CFR 820.100 and ISO 13485 §8.5.2/§8.5.3. "
            "Identify root cause categories, corrective actions, and regulatory obligations."
        ),
        expected_output="CAPA analysis with root cause, corrective actions, and regulatory citations",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff(inputs={"problem": problem_statement, "country": country, "device_class": device_class})
    logger.info("CAPA CrewAI crew completed; running direct pipeline for structured output")
    return run_capa_analysis(problem_statement, country, device_class, device_type)


def run_capa_analysis(
    problem_statement: str,
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
    use_crewai: bool = False,
) -> CAPAAnalysis:
    """Generate a CAPA analysis backed by regulatory FAISS data.

    Args:
        use_crewai: When True, runs the CAPA CrewAI agent first, then produces
                    the structured result via the direct pipeline.
    """
    if use_crewai:
        return _run_capa_crew(problem_statement, country, device_class, device_type)

    from app.tools.llm import chat_completion

    store = get_vector_store()

    queries = [
        # Core CAPA framework
        "corrective action preventive action CAPA procedure requirements",
        "ISO 13485 8.5 corrective preventive action nonconformity",
        "21 CFR 820.100 CAPA quality system corrective action procedure",
        "nonconformity root cause analysis investigation",
        "complaint investigation corrective action trending",
        "regulatory reporting field safety corrective action FSCA recall",
        "serious incident vigilance report mandatory notification",
        # Root cause analysis methodology
        "root cause analysis 5-Why Fishbone Ishikawa FTA FMEA",
        "effectiveness verification CAPA closure criteria",
        "design change control CAPA 21 CFR 820.30",
        # MDSAP and multi-market
        "MDSAP corrective action preventive action audit chapter",
        "EU MDR Article 10 quality management system CAPA",
        "ISO 13485 nonconformity product corrective action record",
        # Notification thresholds
        "serious incident reporting 15 days EU MDR Article 87",
        "MDR 30 days adverse event reporting 21 CFR 803",
        "field safety corrective action FSCA user notice",
    ]

    # Problem-statement-aware dynamic queries
    problem_lower = problem_statement.lower()
    if any(w in problem_lower for w in ("complaint", "incident", "injury", "patient")):
        queries.append("serious incident vigilance reporting manufacturer PMS")
    if any(w in problem_lower for w in ("design", "specification", "drawing")):
        queries.append("design change control 21 CFR 820.30 design FMEA")
    if any(w in problem_lower for w in ("software", "firmware", "algorithm")):
        queries.append("IEC 62304 software problem resolution software CAPA")
    if any(w in problem_lower for w in ("supplier", "component", "material", "raw")):
        queries.append("supplier control corrective action purchasing controls 21 CFR 820.50")
    if any(w in problem_lower for w in ("steril", "contamin", "particulate")):
        queries.append("sterility failure investigation corrective action ISO 11135")

    chunks = multi_query_faiss(
        store, queries, country, device_class=device_class or None, top_k=4, max_chunks=45
    )
    regulation_context = build_regulation_context(chunks, max_chunks=38)

    system_prompt = (
        "You are a Lead Quality Engineer and Regulatory Compliance Specialist (RAPS CRC certified) "
        "with 16 years managing CAPA systems at Class III device manufacturers under FDA Warning "
        "Letters. You know exactly what FDA, TÜV SÜD, and BSI auditors look for.\n\n"
        "ISO 13485:2016 §8.5.2 Corrective Action — MANDATORY elements:\n"
        "  1. Review nonconformities (including complaints) with statistical trend analysis\n"
        "  2. Determine CAUSES of nonconformity (not just symptoms)\n"
        "  3. Evaluate NEED for action to prevent recurrence\n"
        "  4. IMPLEMENT corrective action\n"
        "  5. RECORD results of action taken\n"
        "  6. VERIFY effectiveness before CAPA closure (re-occurrence check ≥90 days)\n\n"
        "§8.5.3 Preventive Action — same cycle for POTENTIAL nonconformities identified via "
        "process data, audit findings, service records, risk assessment updates.\n\n"
        "21 CFR 820.100 CAPA (QMSR 2026) — analyze: all complaint files (820.198), "
        "service records, deviations/concessions, audit results, process performance, "
        "returned goods. Trending rule: 3+ similar complaints may trigger CAPA automatically. "
        "Effectiveness check: no recurrence within specified monitoring window.\n\n"
        "EU MDR Art 10(9)(l): QMS must include CAPA procedure as documented process. "
        "Art 87: if CAPA relates to an event that meets 'serious incident' definition, "
        "regulatory notification MANDATORY within 15 days of awareness.\n\n"
        "Severity classification for medical devices:\n"
        "  Critical = potential/actual patient death or serious injury, OR field recall potential\n"
        "  Major = nonconformity requires 30-day resolution plan, QMS effectiveness impacted\n"
        "  Minor = systemic issue but no immediate patient risk, addressable at next review\n\n"
        "MDSAP Chapter 6: single CAPA system valid across US, CA, AU, BR, JP.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Base every corrective action and regulatory obligation on the REGULATORY TEXT provided below\n"
        "- Every regulation_cite must reference text you were actually given\n"
        "- Never infer obligations not explicitly stated in the provided text\n"
        "- If no regulatory text supports a finding, do not include it\n\n"
        "Output ONLY valid JSON — no markdown, no prose, no preamble."
    )

    user_prompt = f"""Perform a comprehensive CAPA analysis for:

Problem Statement: {problem_statement}
Country: {country}
Device Class: {device_class}
{f"Device Type: {device_type}" if device_type else ""}

Apply ISO 13485:2016 §8.5.2/8.5.3 and 21 CFR 820.100 methodology.
Use ONLY the regulatory text below for citations.

REGULATORY TEXT:
{regulation_context}

Return a JSON object with ALL these fields:
- severity: "Critical" | "Major" | "Minor" (based on patient risk and regulatory impact)
- requires_regulatory_notification: boolean (true if CAPA involves a serious incident/FSCA trigger)
- notification_rationale: string (explain why notification is/is not required, cite regulation)
- recommended_timeline_days: integer (30=Major, 90=Minor, 15=Critical)
- root_cause_categories: array of 3-5 categories, each with:
  - category: "Design" | "Process" | "Training" | "Supplier" | "Software" | "Human Error" |
    "Environmental" | "Materials" | "Labeling/IFU" | "Use Error"
  - likelihood: "High" | "Medium" | "Low"
  - investigation_questions: array of 5-7 specific targeted questions for THIS problem
  - tools: array of 2-3 RCA tools (e.g. "5-Why", "Fishbone/Ishikawa", "FTA", "FMEA",
    "Is-Is-Not Analysis", "Failure Mode Analysis")
- corrective_actions: array of 5-8 actions (include Containment, Corrective, AND Preventive):
  - action_id: "CA-001", "CA-002", etc.
  - action_type: "Containment" | "Corrective" | "Preventive"
  - description: specific, actionable description of what to do
  - responsible_department: "Quality Engineering" | "R&D" | "Manufacturing" | "Regulatory" |
    "Clinical" | "Supply Chain" | "Training"
  - target_completion_days: integer (Containment: 1-5, Corrective: 15-30, Preventive: 30-90)
  - effectiveness_criteria: measurable criterion (e.g. "Zero recurrence within 90-day monitoring period")
  - regulation_cite: specific citation
- regulatory_obligations: array of 3-6 obligations triggered by this CAPA:
  - obligation: what the regulation requires
  - regulation_cite: exact citation
  - timeline: specific deadline
  - documentation_required: array of required records

Output ONLY valid JSON. Be technically precise — FDA may review this CAPA record."""

    try:
        raw = chat_completion(system_prompt, user_prompt)
        data = parse_llm_json(raw)
        if not isinstance(data, dict):
            data = {}
        root_causes = [RootCauseCategory(**rc) for rc in data.get("root_cause_categories", [])]
        actions = [CorrectiveAction(**ca) for ca in data.get("corrective_actions", [])]
        obligations = [CAPARegObligation(**o) for o in data.get("regulatory_obligations", [])]

        # Citation assertion: ensure baseline CAPA obligations are always cited
        has_21cfr = any("820.100" in o.regulation_cite for o in obligations)
        has_iso13485 = any("13485" in o.regulation_cite for o in obligations)
        if not has_21cfr and country.upper() in ("US", "CA"):
            obligations.append(CAPARegObligation(
                obligation="Implement CAPA procedure per 21 CFR 820.100",
                regulation_cite="21 CFR 820.100",
                timeline="30 days",
                documentation_required=["CAPA investigation record", "Effectiveness check report"],
            ))
        if not has_iso13485:
            obligations.append(CAPARegObligation(
                obligation="Corrective and preventive action per ISO 13485:2016 §8.5.2/§8.5.3",
                regulation_cite="ISO 13485:2016 §8.5.2",
                timeline="30 days",
                documentation_required=["Corrective action record", "Root cause analysis", "Effectiveness verification"],
            ))

        return CAPAAnalysis(
            problem_statement=problem_statement,
            country=country,
            device_class=device_class,
            severity=data.get("severity", "Major"),
            root_cause_categories=root_causes,
            corrective_actions=actions,
            regulatory_obligations=obligations,
            requires_regulatory_notification=data.get("requires_regulatory_notification", False),
            notification_rationale=data.get("notification_rationale", ""),
            recommended_timeline_days=data.get("recommended_timeline_days", 30),
        )
    except Exception as e:
        logger.error("CRITICAL: CAPA OpenAI/LLM failure — serving static fallback CAPA: %s", e)
        report = _fallback_capa(problem_statement, country, device_class, device_type)
        report.is_fallback = True
        return report


def _fallback_capa(
    problem_statement: str, country: str, device_class: str, device_type: Optional[str]
) -> CAPAAnalysis:
    store = get_vector_store()
    obligations = []
    for r in store.search("CAPA corrective action requirements", country=country, top_k=3):
        obligations.append(
            CAPARegObligation(
                obligation="Implement corrective action per QMS requirements",
                regulation_cite=r.get("document_id") or country,
                timeline="30 days",
                documentation_required=["CAPA record", "Effectiveness check"],
            )
        )
    return CAPAAnalysis(
        problem_statement=problem_statement,
        country=country,
        device_class=device_class,
        severity="Major",
        root_cause_categories=[
            RootCauseCategory(
                category="Process",
                likelihood="High",
                investigation_questions=["What process failed?", "When did it first occur?"],
                tools=["5-Why", "Fishbone"],
            )
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="CA-001",
                action_type="Corrective",
                description="Investigate root cause and implement corrective measures",
                responsible_department="Quality",
                target_completion_days=30,
                effectiveness_criteria="Zero recurrence within 90 days",
                regulation_cite="ISO 13485:2016 §8.5.2",
            )
        ],
        regulatory_obligations=obligations,
    )
