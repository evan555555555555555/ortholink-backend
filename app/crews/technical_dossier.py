"""
OrthoLink TDA Crew — Technical Documentation Agent
POST /api/v1/technical-dossier

Generates a complete technical documentation checklist for regulatory submissions.
Maps every required document section to specific regulatory citations from FAISS.
Supports: 510(k), PMA, CE MDR Technical File (Annex II+III), Japan PMDA, TGA, KFDA, etc.

EU MDR Annex II Technical Documentation — mandatory 6-section structure:
  §1  Device description, UDI, intended use, indications, contraindications
  §2  Information supplied with device (labeling, IFU, language requirements)
  §3  Design and manufacturing information (specifications, validation)
  §4  GSPR compliance — General Safety & Performance Requirements matrix
  §5  Benefit-risk analysis and risk management (ISO 14971 risk file)
  §6  Product verification and validation summary (V&V reports, clinical evaluation)
  Annex III: PMS technical documentation (PMCF plan/report for Class III)
"""

import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import build_regulation_context, multi_query_faiss, parse_llm_json
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Technical documentation guidance only. Verify all requirements with the "
    "relevant competent authority before submission."
)


class TechDocSection(BaseModel):
    """A section of a technical documentation file."""

    section_title: str = Field(..., description="Document section title")
    section_number: str = Field(..., description="Section number (e.g. 1.1, Annex II.3)")
    required: bool = Field(default=True, description="Mandatory vs. recommended")
    regulation_cite: str = Field(..., description="Specific regulatory citation")
    description: str = Field(..., description="What must be included in this section")
    typical_contents: list[str] = Field(
        default_factory=list,
        description="List of expected documents/artefacts in this section",
    )
    estimated_effort_days: int = Field(
        default=0, description="Typical preparation time in working days"
    )
    notes: str = Field(default="", description="Regulatory guidance notes")


class TechnicalDossierPlan(BaseModel):
    """Complete technical documentation plan for a regulatory submission."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str
    device_class: str
    device_type: Optional[str] = None
    submission_type: str = Field(
        default="",
        description="e.g. 510(k), PMA, CE Technical File, MDR Technical Documentation",
    )
    sections: list[TechDocSection] = Field(default_factory=list)
    total_estimated_days: int = Field(default=0)
    executive_summary: str = Field(default="")
    disclaimer: str = Field(default=DISCLAIMER)
    is_fallback: bool = Field(
        default=False,
        description="True if LLM failed and fallback data was served. "
        "Frontend MUST display degraded-data warning when True.",
    )


def _run_tda_crew(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> "TechnicalDossierPlan":
    """CrewAI path: TDA agent maps documents to EU MDR Annex II/III via crew.kickoff()."""
    from crewai import Crew, Process, Task

    from app.agents.tda_agent import get_tda_agent

    agent = get_tda_agent()
    task = Task(
        description=(
            f"Generate a technical documentation plan for a {device_class} medical device.\n"
            f"Country/market: {country}\n"
            f"{'Device type: ' + device_type if device_type else ''}\n\n"
            "Map all required sections to EU MDR Annex II/III structure. "
            "Cite specific regulation articles and annexes for each section."
        ),
        expected_output="Technical dossier plan with sections mapped to regulation articles",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff(inputs={"country": country, "device_class": device_class})
    logger.info("TDA CrewAI crew completed; running direct pipeline for structured output")
    return run_technical_dossier(country, device_class, device_type)


def run_technical_dossier(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
    use_crewai: bool = False,
) -> TechnicalDossierPlan:
    """
    Generate a technical documentation plan backed by FAISS regulatory data.
    Uses direct LLM call (no CrewAI overhead) for speed.

    Args:
        use_crewai: When True, runs the TDA CrewAI agent first for EU MDR Annex
                    mapping, then produces the structured result via the direct pipeline.
    """
    if use_crewai:
        return _run_tda_crew(country, device_class, device_type)

    from app.tools.llm import chat_completion

    store = get_vector_store()

    # ── Comprehensive FAISS query set ─────────────────────────────────────
    queries = [
        # Core technical file structure
        "technical documentation technical file requirements medical device",
        "design dossier design history file DHF device master record",
        "risk management file ISO 14971 risk analysis technical documentation",
        "biocompatibility testing ISO 10993 biological evaluation",
        "clinical evaluation report clinical data evidence",
        "labeling instructions for use IFU requirements",
        "post-market surveillance technical documentation PMS plan",
        "quality management system technical documentation evidence",
        "software documentation IEC 62304 software lifecycle",
        "performance testing verification validation V&V",
        "sterilization validation ISO 11135 sterility assurance",
        "usability engineering IEC 62366 human factors",
        # General safety and performance
        "general safety performance requirements essential requirements",
        "GSPR compliance matrix checklist",
        "electromagnetic compatibility EMC IEC 60601",
        "electrical safety IEC 60601-1 mechanical testing",
    ]

    # US-specific queries
    if country.upper() == "US":
        queries += [
            "510k premarket notification substantial equivalence predicate",
            "design controls 21 CFR 820.30 design history file",
            "PMA premarket approval clinical data safety effectiveness",
            "21 CFR 807.87 510k content requirements",
            "de novo classification request 21 CFR 515B",
            "FDA software guidance OTS predetermined change control",
        ]
    # EU-specific queries
    elif country.upper() == "EU":
        queries += [
            "EU MDR Annex II technical documentation mandatory sections",
            "GSPR general safety performance requirements EU MDR Annex I",
            "notified body technical file assessment review",
            "EU MDR Annex III PMS technical documentation PMCF",
            "CE marking declaration of conformity DoC",
            "unique device identification UDI EUDAMED registration",
            "clinical evaluation MEDDEV 2.7/1 Rev 4 EU MDR Article 61",
        ]
    elif country.upper() in ("JP", "JAPAN"):
        queries += [
            "PMDA Japan Ordinance 169 technical documentation shonin",
            "Japan MHLW technical file requirements QMS",
        ]
    elif country.upper() == "AU":
        queries += [
            "TGA essential principles conformance assessment technical evidence",
            "Australia ARTG registration conformity assessment",
        ]
    elif country.upper() == "KR":
        queries += [
            "MFDS Korea technical documentation requirements",
            "Korea medical device registration GMP",
        ]

    chunks = multi_query_faiss(
        store, queries, country, device_class=device_class or None, top_k=4, max_chunks=50
    )
    regulation_context = build_regulation_context(chunks, max_chunks=45)

    # Determine submission type
    submission_map = {
        "US": {"I": "510(k) or Exempt", "II": "510(k)", "III": "PMA"},
        "EU": {
            "I": "Self-Declaration (Annex IV DoC)",
            "IIA": "CE Technical File (NB optional for Annex IX)",
            "IIB": "CE Technical File + NB Annex IX/X",
            "III": "CE Technical File + NB Annex IX/X + PMCF",
        },
        "JP": {"I": "Tsuuchi (notification)", "II": "Ninsho (certification)", "III": "Shonin (approval)"},
        "AU": {"I": "Self-assessment", "II": "Conformity assessment", "III": "Conformity assessment + TGA"},
        "KR": {"I": "Notification", "II": "Certification", "III": "Permission"},
    }
    submission_type = submission_map.get(country.upper(), {}).get(
        device_class.upper().replace(" ", ""), f"{country} Regulatory Submission"
    )

    system_prompt = (
        "You are a Technical Documentation Lead Specialist with 20 years preparing regulatory "
        "submissions for Class IIb/III orthopedic, cardiovascular, and active implantable devices. "
        "You've successfully submitted 510(k)s, PMAs, EU MDR Technical Files, and PMDA Japan Shonin "
        "applications. You know the exact content every notified body and FDA reviewer expects.\n\n"
        "EU MDR Annex II — 6 mandatory sections (no section can be omitted):\n"
        "  §1: Device identification — UDI (Art 27), intended purpose, indications, variants, accessories\n"
        "  §2: Information supplied — labeling in EU languages (Annex I §23), IFU, language per MDD Art 4\n"
        "  §3: Design & manufacture — design specifications, tolerances, manufacturing processes,\n"
        "      sterilization validation, materials list, biocompatibility data\n"
        "  §4: GSPR compliance matrix — maps EVERY GSPR requirement (Annex I) to specific design\n"
        "      feature, test standard, or documented evidence. 'Not applicable' requires justification.\n"
        "  §5: Benefit-risk analysis — ISO 14971 risk management file summary + benefit-risk per\n"
        "      Art 1(2)(f); for Class III must include benefit-risk conclusion per Art 52(3)\n"
        "  §6: Verification & validation — V&V test protocols + reports, clinical evaluation per\n"
        "      Art 61 referencing Annex XIV (clinical evaluation methodology)\n"
        "  Annex III: PMS tech doc — PMCF plan (mandatory Class III), PMCF reports (annual)\n\n"
        "FDA 510(k) — 21 CFR 807.87 required elements: device description, substantial equivalence\n"
        "comparison, performance data, 510(k) summary or statement, truthful/accurate declaration.\n"
        "For Class III: PMA per 21 CFR 814 — full clinical data, manufacturing info, proposed labeling.\n\n"
        "IEC 62304:2006+AMD1 software lifecycle — required when device contains software:\n"
        "  Software Development Plan, Architecture Document, Software Requirements Specification,\n"
        "  SOUP list (with risk classification), Verification & Validation records, Anomaly Reports.\n\n"
        "ISO 14971:2019: Risk Management File is NEVER a one-time document — maintained throughout\n"
        "device lifecycle; updated with every design change, field incident, or PMS signal.\n\n"
        "Effort estimates (working days): Device Description 3-5d, Risk Management File 15-30d,\n"
        "Clinical Evaluation Report 30-90d, Biocompatibility Testing 45-90d, V&V Testing 30-60d,\n"
        "Software Documentation 20-45d if software involved, GSPR Matrix 10-20d.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Base every documentation section on the REGULATORY TEXT provided below\n"
        "- Every regulation_cite must reference text you were actually given\n"
        "- Never infer requirements not explicitly stated in the provided text\n"
        "- If no regulatory text supports a section, do not include it\n\n"
        "Output ONLY valid JSON — no markdown, no prose, no preamble."
    )

    user_prompt = f"""Generate a complete, submission-ready technical documentation checklist for:

Country: {country}
Device Class: {device_class}
{f"Device Type: {device_type}" if device_type else ""}
Submission Type: {submission_type}

Base EVERY section on the regulatory text below. Cite specific articles/annexes/sections.

REGULATORY TEXT:
{regulation_context}

Return a JSON object with:
- submission_type: string (exact regulatory pathway)
- executive_summary: 3-4 sentence overview of the documentation burden and critical path items
- sections: array of 12-18 sections (be comprehensive), each with:
  - section_title: specific title (e.g. "Risk Management File per ISO 14971:2019",
    "GSPR Compliance Matrix (Annex I EU MDR)", "Clinical Evaluation Report per Art 61")
  - section_number: section identifier (e.g. "Annex II §1", "1.1", "Section 5")
  - required: boolean (true if mandatory for this submission type/class)
  - regulation_cite: specific citation (e.g. "EU MDR Annex II §4", "21 CFR 807.87(e)")
  - description: 2-3 sentences on what this section must contain
  - typical_contents: array of 3-6 specific documents/artefacts required
    (e.g. ["Risk management plan", "Hazard identification table", "Risk control verification records"])
  - estimated_effort_days: realistic estimate in working days
  - notes: important regulatory nuance, common deficiency, or NB/FDA expectation

MANDATORY sections to include: Device Description, GSPR/Essential Requirements Compliance,
Risk Management File (ISO 14971), Biocompatibility (ISO 10993), Clinical Evaluation,
Labeling/IFU, QMS Evidence, V&V Testing Summary, PMS Plan, Sterility (if applicable),
Software (IEC 62304, if applicable), Usability/Human Factors (IEC 62366).

Output ONLY valid JSON."""

    try:
        raw = chat_completion(system_prompt, user_prompt)
        plan_data = parse_llm_json(raw)
        if not isinstance(plan_data, dict):
            plan_data = {}
        sections = [TechDocSection(**s) for s in plan_data.get("sections", [])]
        total_days = sum(s.estimated_effort_days for s in sections)
        return TechnicalDossierPlan(
            country=country,
            device_class=device_class,
            device_type=device_type,
            submission_type=plan_data.get("submission_type", submission_type),
            sections=sections,
            total_estimated_days=total_days,
            executive_summary=plan_data.get("executive_summary", ""),
        )
    except Exception as e:
        logger.error("CRITICAL: TDA OpenAI/LLM failure — serving static fallback dossier: %s", e)
        report = _fallback_dossier(country, device_class, device_type, submission_type)
        report.is_fallback = True
        return report


def _fallback_dossier(
    country: str,
    device_class: str,
    device_type: Optional[str],
    submission_type: str,
) -> TechnicalDossierPlan:
    """Fallback: search FAISS and build minimal structure."""
    store = get_vector_store()
    seen: set = set()
    sections = []
    std_queries = [
        "technical documentation requirements",
        "risk management file",
        "clinical evaluation report",
        "labeling requirements",
    ]
    for q in std_queries:
        for r in store.search(q, country=country, device_class=device_class or None, top_k=3):
            cid = r.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                sections.append(
                    TechDocSection(
                        section_title=r.get("section_path") or "Requirements",
                        section_number=str(len(sections) + 1),
                        required=True,
                        regulation_cite=r.get("document_id") or country,
                        description=r.get("text", "")[:300],
                        typical_contents=[],
                        estimated_effort_days=5,
                    )
                )
    return TechnicalDossierPlan(
        country=country,
        device_class=device_class,
        device_type=device_type,
        submission_type=submission_type,
        sections=sections[:15],
        total_estimated_days=sum(s.estimated_effort_days for s in sections[:15]),
    )
