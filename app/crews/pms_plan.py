"""
OrthoLink PMS Crew — Post-Market Surveillance Plan Generator
POST /api/v1/pms-plan

Regulatory basis:
  - EU MDR 2017/745 Art 83 (PMS obligation), Art 84 (PMSR vs PSUR determination),
    Art 85 (PSUR content — Class IIb annual; Class III annual before NB assessment),
    Art 86 (PMCF plan per Art 61(11) — mandatory Class III),
    Art 87 (Serious incident: 15 days; non-serious trend: 30 days; immediate hazard: 2 days)
  - FDA 21 CFR 803 (MDR reporting: 30 days; 5-day malfunction MDRs)
  - FDA 21 CFR 820.198 (Complaint files — investigate ALL, close within 30 days)
  - FDA 21 CFR 820.100 (CAPA triggered from complaint trending)
  - ISO 13485:2016 §8.2.1 (Systematic PMS data collection)
  - MDSAP: single PMS system covering US/CA/AU/BR/JP simultaneously
"""

import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import build_regulation_context, multi_query_faiss, parse_llm_json
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "PMS plan guidance only. Regulatory submissions must be prepared by qualified "
    "regulatory affairs professionals and reviewed by legal counsel."
)


class PMSActivity(BaseModel):
    """A single post-market surveillance activity."""

    activity: str = Field(..., description="Activity name")
    frequency: str = Field(..., description="How often (e.g. Continuous, Annual, Quarterly)")
    responsible_party: str = Field(..., description="Who is responsible")
    regulation_cite: str = Field(..., description="Regulatory basis")
    outputs: list[str] = Field(default_factory=list, description="Required outputs/reports")
    trigger_threshold: str = Field(
        default="", description="Threshold that triggers action (e.g. ≥1 serious incident)"
    )
    notes: str = Field(default="")


class ReportingRequirement(BaseModel):
    """A regulatory reporting requirement."""

    report_type: str = Field(
        ..., description="e.g. MDR (Medical Device Report), PSUR, Vigilance Report, PMCF"
    )
    trigger: str = Field(..., description="What triggers this report")
    timeline: str = Field(..., description="Deadline from trigger event")
    recipient: str = Field(..., description="Who receives the report (authority name)")
    regulation_cite: str
    template_hint: str = Field(default="", description="Key fields to include")


class PMSPlan(BaseModel):
    """Complete Post-Market Surveillance plan."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str
    device_class: str
    device_type: Optional[str] = None
    pms_level: str = Field(
        default="",
        description="Required PMS level: Basic PMS Plan | Full PMS Plan + PMSR | PMCF Required",
    )
    activities: list[PMSActivity] = Field(default_factory=list)
    reporting_requirements: list[ReportingRequirement] = Field(default_factory=list)
    key_data_sources: list[str] = Field(
        default_factory=list,
        description="Data sources to monitor (complaints, literature, vigilance databases)",
    )
    review_frequency: str = Field(default="Annual")
    executive_summary: str = Field(default="")
    disclaimer: str = Field(default=DISCLAIMER)
    is_fallback: bool = Field(
        default=False,
        description="True if LLM failed and fallback data was served. "
        "Frontend MUST display degraded-data warning when True.",
    )


def _run_pms_crew(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> "PMSPlan":
    """CrewAI path: PMS agent designs surveillance plan via crew.kickoff()."""
    from crewai import Crew, Process, Task

    from app.agents.pms_agent import get_pms_agent

    agent = get_pms_agent()
    task = Task(
        description=(
            f"Design a post-market surveillance (PMS) plan for a {device_class} medical device.\n"
            f"Country/market: {country}\n"
            f"{'Device type: ' + device_type if device_type else ''}\n\n"
            "Include EU MDR Article 84 requirements, vigilance reporting obligations, "
            "PMCF plan if required, and PMS data sources. Cite specific regulation articles."
        ),
        expected_output="PMS plan with activities, reporting requirements, and regulatory citations",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff(inputs={"country": country, "device_class": device_class})
    logger.info("PMS CrewAI crew completed; running direct pipeline for structured output")
    return run_pms_plan(country, device_class, device_type)


def run_pms_plan(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
    use_crewai: bool = False,
) -> PMSPlan:
    """Generate a PMS plan backed by regulatory FAISS data.

    Args:
        use_crewai: When True, runs the PMS CrewAI agent first for EU MDR Article 84
                    narrative, then produces the structured result via the direct pipeline.
    """
    if use_crewai:
        return _run_pms_crew(country, device_class, device_type)

    from app.tools.llm import chat_completion

    store = get_vector_store()

    queries = [
        # Core PMS framework
        "post-market surveillance plan PMS requirements medical device",
        "vigilance reporting serious incident adverse event manufacturer",
        "periodic safety update report PSUR PMSR content requirements",
        "post-market clinical follow-up PMCF plan class III",
        "complaint handling investigation complaint files customer feedback",
        "medical device reporting MDR adverse event 30 days",
        "post-market surveillance ISO 13485 section 8.2 systematic",
        "trending analysis signal detection safety issues",
        "literature surveillance systematic review published data",
        "field safety corrective action FSCA recall withdrawal",
        # PMS data sources and KPIs
        "complaint data vigilance database EUDAMED MedWatch",
        "post-market surveillance data analysis evaluation",
        "MAUDE database FDA device event reports",
        "sales and distribution data post-market monitoring",
        # Country-specific
        "post-market surveillance plan submission requirements country",
        "vigilance system national competent authority reporting",
    ]
    if country.upper() == "US":
        queries += [
            "MDR reporting 21 CFR 803 manufacturer device report",
            "complaint files 21 CFR 820.198 investigation requirements",
            "FDA mandatory problem reporting 5-day 30-day MDR",
            "QMSR post-market surveillance 21 CFR 820",
            "FDA CAPA trending complaints corrective action",
        ]
    elif country.upper() == "EU":
        queries += [
            "EU MDR Article 83 post-market surveillance system",
            "PMSR periodic safety update report Article 85 86",
            "serious incident reporting EU MDR Article 87 15 days",
            "PMCF post-market clinical follow-up Article 61 74",
            "EUDAMED vigilance module reporting electronic system",
            "EU MDR Article 84 PMSR versus PSUR class IIb III",
        ]
    elif country.upper() in ("JP", "JAPAN"):
        queries += [
            "PMDA post-market vigilance reporting Japan Ordinance",
            "Japan post-market surveillance plan PMDA requirements",
        ]
    elif country.upper() == "CA":
        queries += [
            "Health Canada MDSAP post-market surveillance",
            "Canada Medical Devices Regulations mandatory problem reporting",
        ]

    chunks = multi_query_faiss(
        store, queries, country, device_class=device_class or None, top_k=4, max_chunks=45
    )
    regulation_context = build_regulation_context(chunks, max_chunks=40)

    # ── Class-specific PMS level determination ─────────────────────────────
    eu_class_map = {
        "I": "Basic PMS Plan",
        "IIA": "Basic PMS Plan",
        "IIB": "Full PMS Plan + PMSR",
        "III": "PMCF Required",
    }
    pms_level_hint = eu_class_map.get(device_class.upper().replace(" ", ""), "Full PMS Plan + PMSR")

    system_prompt = (
        "You are a Principal Regulatory Affairs Scientist with 18 years of experience "
        "preparing PMS plans for Class IIb and III medical devices accepted by FDA, EMA notified "
        "bodies (TÜV SÜD, BSI, SGS), PMDA Japan, TGA Australia, and Health Canada.\n\n"
        "EU MDR mastery:\n"
        "  Art 83: PMS system mandatory for all devices — proactive, systematic data collection.\n"
        "  Art 84: Class I/IIa → PMSR (produced at any time on request); Class IIb → PMSR updated "
        "annually; Class III → PSUR updated annually (submitted to NB before annual review).\n"
        "  Art 85/86: PSUR must include device usage numbers, serious incident rate, FSCA summary, "
        "benefit-risk conclusion, PMCF conclusions, proposed labeling changes.\n"
        "  Art 87: SERIOUS incident: report within 15 days of awareness; NON-SERIOUS trend: 30 days; "
        "FIELD SAFETY CORRECTIVE ACTION with immediate hazard: 2 days + FSCA notice to users.\n"
        "  Art 88: Reporting of incidents — trended signals that exceed statistical thresholds.\n"
        "  PMCF (Art 61/74): mandatory for Class III; must include plan AND report in Annex III tech doc.\n\n"
        "FDA mastery:\n"
        "  21 CFR 803: MDR reporting — death/serious injury: 30 days from awareness; malfunction "
        "with potential to cause serious injury if recurs: 30 days; 5-day report if imminent hazard.\n"
        "  21 CFR 820.198: complaint files — ALL complaints investigated; documented resolution "
        "within 30 calendar days. PMS and CAPA are linked: trending ≥3 similar complaints triggers CAPA.\n"
        "  QMSR 21 CFR 820 (Feb 2026): aligns QMS with ISO 13485 — PMS directly referenced.\n\n"
        "ISO 13485:2016 §8.2.1: systematic collection of post-production data mandatory across markets.\n"
        "MDSAP: single PMS system valid for US, CA, AU, BR, JP simultaneously.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Base every PMS activity and reporting requirement on the REGULATORY TEXT provided below\n"
        "- Every regulation_cite must reference text you were actually given\n"
        "- Never infer obligations not explicitly stated in the provided text\n"
        "- If no regulatory text supports an activity, do not include it\n\n"
        "Output ONLY valid JSON — no markdown, no prose, no preamble."
    )

    user_prompt = f"""Generate a comprehensive, submission-ready Post-Market Surveillance plan for:

Country: {country}
Device Class: {device_class}
{f"Device Type: {device_type}" if device_type else ""}
Expected PMS Level: {pms_level_hint}

Use ONLY the regulatory text below for every citation. Be specific — cite actual article numbers.

REGULATORY TEXT:
{regulation_context}

Return a JSON object with ALL of these fields:
- pms_level: "Basic PMS Plan" | "Full PMS Plan + PMSR" | "PMCF Required"
- executive_summary: 3-4 sentence overview of PMS obligations for this device/country
- review_frequency: "Annual" | "Biannual" | "Per incident"
- key_data_sources: array of 6-10 specific data sources (e.g. "EUDAMED vigilance module",
  "MAUDE FDA database", "published literature searches PubMed/EMBASE", "customer complaint records",
  "field service reports", "clinical registry data", "FSCA database searches")
- activities: array of 8-12 PMS activities, each with:
  - activity: specific activity name (e.g. "Serious Incident Trend Analysis", "PMCF Data Review")
  - frequency: "Continuous" | "Monthly" | "Quarterly" | "Annual" | "Event-triggered"
  - responsible_party: specific role (e.g. "Regulatory Affairs Manager", "Quality Engineer",
    "Clinical Affairs", "Post-Market Surveillance Specialist")
  - regulation_cite: exact citation (e.g. "EU MDR Art 87", "21 CFR 803.10")
  - outputs: array of 2-4 required deliverables (e.g. ["Vigilance report", "EUDAMED entry"])
  - trigger_threshold: specific threshold (e.g. "≥1 serious incident", "≥3 similar complaints/quarter")
  - notes: regulatory nuance or practical implementation note
- reporting_requirements: array of 5-8 reporting obligations, each with:
  - report_type: specific report name (e.g. "Serious Incident Report", "PSUR", "MDR 3500A")
  - trigger: precise trigger condition
  - timeline: exact deadline from awareness (e.g. "15 calendar days", "30 days", "2 days")
  - recipient: exact authority name (e.g. "National Competent Authority via EUDAMED", "FDA MedWatch")
  - regulation_cite: exact legal citation
  - template_hint: key required fields for this report

Output ONLY valid JSON. Be technically precise — this plan will be submitted to regulators."""

    try:
        raw = chat_completion(system_prompt, user_prompt)
        data = parse_llm_json(raw)
        if not isinstance(data, dict):
            data = {}
        activities = [PMSActivity(**a) for a in data.get("activities", [])]
        reporting = [ReportingRequirement(**r) for r in data.get("reporting_requirements", [])]
        return PMSPlan(
            country=country,
            device_class=device_class,
            device_type=device_type,
            pms_level=data.get("pms_level", pms_level_hint),
            activities=activities,
            reporting_requirements=reporting,
            key_data_sources=data.get("key_data_sources", []),
            review_frequency=data.get("review_frequency", "Annual"),
            executive_summary=data.get("executive_summary", ""),
        )
    except Exception as e:
        logger.error("CRITICAL: PMS OpenAI/LLM failure — serving static fallback PMS plan: %s", e)
        report = _fallback_pms(country, device_class, device_type)
        report.is_fallback = True
        return report


def _fallback_pms(country: str, device_class: str, device_type: Optional[str]) -> PMSPlan:
    store = get_vector_store()
    activities = []
    for r in store.search("post-market surveillance vigilance", country=country, top_k=5):
        activities.append(
            PMSActivity(
                activity="Regulatory Monitoring",
                frequency="Continuous",
                responsible_party="Regulatory Affairs",
                regulation_cite=r.get("document_id") or country,
                outputs=["Surveillance report"],
                notes=r.get("text", "")[:200],
            )
        )
    return PMSPlan(
        country=country,
        device_class=device_class,
        device_type=device_type,
        pms_level="Full PMS Plan",
        activities=activities[:5],
        reporting_requirements=[],
    )
