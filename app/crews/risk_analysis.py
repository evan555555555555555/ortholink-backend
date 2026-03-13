"""
Risk Management Agent (RMA) — ISO 14971:2019
POST /api/v1/risk-analysis

Regulatory basis:
  - ISO 14971:2019  (Risk management for medical devices)
  - ISO/TR 24971:2020  (Guidance on the application of ISO 14971)
  - EU MDR 2017/745 Annex I — General Safety & Performance Requirements (GSPR)
  - FDA Design Controls 21 CFR 820.30(g) — Risk Analysis
  - IEC 62304:2006+AMD1:2015  (Software risk classification)
  - ISO 10993-1:2018  (Biocompatibility — hazard identification)

ISO 14971 §4-9 process:
  §4  Risk analysis       — hazard identification, hazardous situations, harms, P×S
  §5  Risk evaluation     — acceptability matrix (ACCEPTABLE | ALARP | UNACCEPTABLE)
  §6  Risk control        — hierarchy: §6.2a inherent safe > §6.2b protective > §6.2c IFU
  §7  Residual risk eval  — post-control P×S re-assessment + benefit-risk
  §8  Risk management report
  §9  Production / post-production information (PMS feedback loop)

ComplianceAuditor philosophy:
  "Substance over checkboxes — every hazard must map to a specific harm and control."
Reality Checker verdict:
  Default ALARP; ACCEPTABLE requires ALL hazards at residual P×S ≤ 4 AND no UNACCEPTABLE.
"""

import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import build_regulation_context, multi_query_faiss, parse_llm_json
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Risk analysis outputs are for regulatory preparation support only. "
    "All risk management files must be reviewed and approved by qualified "
    "regulatory affairs and quality engineering professionals per ISO 14971:2019 §3.3. "
    "Not a substitute for formal Design FMEA, FTA, or HAZOP studies."
)

# ── ISO 14971 Annex D Example Risk Acceptability Matrix ──────────────────────
# _RISK_MATRIX[(severity, probability)] → "ACCEPTABLE" | "ALARP" | "UNACCEPTABLE"
# S=1 Negligible → S=5 Catastrophic
# P=1 Improbable → P=5 Frequent
_RISK_MATRIX: dict[tuple[int, int], str] = {
    # S=1: Negligible — always acceptable
    (1, 1): "ACCEPTABLE", (1, 2): "ACCEPTABLE", (1, 3): "ACCEPTABLE",
    (1, 4): "ACCEPTABLE", (1, 5): "ACCEPTABLE",
    # S=2: Marginal
    (2, 1): "ACCEPTABLE", (2, 2): "ACCEPTABLE", (2, 3): "ALARP",
    (2, 4): "ALARP",      (2, 5): "ALARP",
    # S=3: Serious
    (3, 1): "ACCEPTABLE", (3, 2): "ALARP",      (3, 3): "ALARP",
    (3, 4): "UNACCEPTABLE", (3, 5): "UNACCEPTABLE",
    # S=4: Critical
    (4, 1): "ALARP",      (4, 2): "ALARP",      (4, 3): "UNACCEPTABLE",
    (4, 4): "UNACCEPTABLE", (4, 5): "UNACCEPTABLE",
    # S=5: Catastrophic
    (5, 1): "ALARP",      (5, 2): "UNACCEPTABLE", (5, 3): "UNACCEPTABLE",
    (5, 4): "UNACCEPTABLE", (5, 5): "UNACCEPTABLE",
}

_SEVERITY_LABELS: dict[int, str] = {
    1: "Negligible",
    2: "Marginal",
    3: "Serious",
    4: "Critical",
    5: "Catastrophic",
}

_PROBABILITY_LABELS: dict[int, str] = {
    1: "Improbable",
    2: "Remote",
    3: "Occasional",
    4: "Probable",
    5: "Frequent",
}

# ISO 14971 §6.2 Risk Control hierarchy (descending preference)
_CONTROL_HIERARCHY: list[str] = [
    "§6.2a — Inherently safe design and manufacture",
    "§6.2b — Protective measures in device or manufacturing process",
    "§6.2c — Information for safety (IFU warnings, labeling)",
]


# ── Pydantic Models ───────────────────────────────────────────────────────────


class HazardEntry(BaseModel):
    """Single row in the ISO 14971 hazard analysis / risk assessment table."""

    hazard_id: str = Field(..., description="Sequential identifier: H-001, H-002, ...")
    hazard: str = Field(..., description="Source of potential harm (ISO 14971 §2.4)")
    hazardous_situation: str = Field(
        ..., description="Circumstance in which people are exposed to hazard (ISO 14971 §2.5)"
    )
    harm: str = Field(..., description="Physical injury or damage to health (ISO 14971 §2.3)")
    severity: int = Field(..., ge=1, le=5)
    severity_label: str = Field(default="")
    probability: int = Field(..., ge=1, le=5)
    probability_label: str = Field(default="")
    risk_score: int = Field(default=0, description="severity × probability (before controls)")
    risk_level: str = Field(default="", description="ACCEPTABLE | ALARP | UNACCEPTABLE")
    # Risk controls (ISO 14971 §6.2 hierarchy)
    control_measures: list[str] = Field(
        default_factory=list,
        description="Specific risk control measures applied",
    )
    control_level: str = Field(
        default=_CONTROL_HIERARCHY[2],
        description="ISO 14971 §6.2a/b/c hierarchy level applied",
    )
    # Residual risk (post-control)
    residual_severity: int = Field(
        default=0, ge=0, le=5,
        description="Post-control severity estimate (0 = unchanged)",
    )
    residual_probability: int = Field(
        default=0, ge=0, le=5,
        description="Post-control probability estimate",
    )
    residual_risk_level: str = Field(
        default="", description="Post-control acceptability verdict"
    )
    regulation_cite: str = Field(default="ISO 14971:2019")


class RiskManagementReport(BaseModel):
    """
    Complete ISO 14971:2019 Risk Management Report.

    Sections:
      hazard_analysis   → §4 Risk Analysis table
      overall_verdict   → §7 Overall Residual Risk evaluation
      benefit_risk_*    → §7.4 Benefit-risk analysis
      applicable_standards → §3.2 Scope of risk management
    """

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_description: str
    intended_use: str
    device_class: str
    country: str
    hazard_analysis: list[HazardEntry] = Field(default_factory=list)
    total_hazards: int = Field(default=0)
    unacceptable_count: int = Field(
        default=0, description="Hazards at UNACCEPTABLE level before controls"
    )
    alarp_count: int = Field(
        default=0, description="Hazards at ALARP before controls"
    )
    acceptable_count: int = Field(
        default=0, description="Hazards at ACCEPTABLE level"
    )
    residual_unacceptable: int = Field(
        default=0,
        description="Hazards STILL UNACCEPTABLE after all controls — must be 0 for submission",
    )
    overall_verdict: str = Field(
        default="ALARP",
        description="ACCEPTABLE | ALARP | UNACCEPTABLE (ISO 14971 §7 overall verdict)",
    )
    benefit_risk_conclusion: str = Field(
        default="",
        description="ISO 14971 §7.4: written justification that benefits outweigh residual risks",
    )
    risk_management_plan_summary: str = Field(
        default="",
        description="Summary of planned risk management activities (§3.4 Risk Management Plan)",
    )
    applicable_standards: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)
    is_fallback: bool = Field(
        default=False,
        description="True if LLM failed and generic fallback hazards were served. "
        "Frontend MUST display degraded-data warning when True.",
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────


def _run_rma_crew(
    device_description: str,
    intended_use: str,
    device_class: str,
    country: str,
    hazards_hint: Optional[str] = None,
) -> RiskManagementReport:
    """CrewAI path: RMA agent runs ISO 14971 hazard analysis via crew.kickoff()."""
    from crewai import Crew, Process, Task

    from app.agents.rma_agent import get_rma_agent

    agent = get_rma_agent()
    task = Task(
        description=(
            f"Perform an ISO 14971:2019 risk management analysis for the following device.\n"
            f"Device description: {device_description}\n"
            f"Intended use: {intended_use}\n"
            f"Device class: {device_class}\n"
            f"Country/market: {country}\n"
            f"{'Hazards hint: ' + hazards_hint if hazards_hint else ''}\n\n"
            "Identify ≥5 hazards. For every hazard, assign severity (1-5) and probability (1-5). "
            "Include specific ISO 14971 clause citations (e.g. §5.5, §6.2a). "
            "Return a complete hazard analysis JSON."
        ),
        expected_output="JSON hazard list with severity, probability, risk_controls, and ISO 14971 citations",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff(inputs={
        "device_description": device_description,
        "intended_use": intended_use,
        "device_class": device_class,
        "country": country,
    })
    # CrewAI path produces narrative output; fall back to direct pipeline for structured result
    logger.info("RMA CrewAI crew completed; running direct pipeline for structured output")
    return run_risk_analysis(device_description, intended_use, device_class, country, hazards_hint)


def run_risk_analysis(
    device_description: str,
    intended_use: str,
    device_class: str,
    country: str,
    hazards_hint: Optional[str] = None,
    use_crewai: bool = False,
) -> RiskManagementReport:
    """
    Generate ISO 14971:2019 risk management report backed by regulatory FAISS data.

    Args:
        use_crewai: When True, runs the RMA CrewAI agent first for narrative analysis,
                    then produces the structured report via the direct pipeline.

    Algorithm:
      1. Build targeted FAISS query set (ISO 14971 clauses + device-/country-specific)
      2. Retrieve top-k chunks → deduplicate → build regulation context string
      3. LLM call with structured JSON output schema
      4. Parse + validate: clamp S/P to 1-5, compute risk matrix verdicts server-side
         (never trust LLM for acceptability — compute from canonical matrix)
      5. Compute aggregate counts + Reality Checker overall verdict
      6. Fallback: return FAISS-grounded minimal report on LLM failure
    """
    if use_crewai:
        return _run_rma_crew(device_description, intended_use, device_class, country, hazards_hint)

    from app.tools.llm import chat_completion

    store = get_vector_store()

    # ── FAISS retrieval — comprehensive ISO 14971 coverage ─────────────────
    queries = [
        "ISO 14971 risk management medical device hazard analysis process",
        "risk acceptability criteria probability severity harm acceptability matrix",
        "risk control measures hierarchy inherently safe design protective measures IFU",
        f"risk management class {device_class} device requirements specific",
        "benefit risk analysis overall residual risk evaluation ISO 14971 section 7",
        "General Safety Performance Requirements GSPR Annex I MDR essential requirements",
        "IEC 62304 software risk classification safety class A B C",
        "ISO 10993 biocompatibility hazard identification biological evaluation",
        "foreseeable misuse hazardous situation harm sequence ISO 14971 section 4",
        "risk management report summary ISO 14971 section 8",
        "post-production information PMS feedback risk management ISO 14971 section 9",
        "hazard identification checklist mechanical electrical thermal radiation",
        "ISO TR 24971 guidance application ISO 14971 risk management",
        "usability engineering human factors risk IEC 62366",
        "sterility failure hazard medical device SAL",
    ]
    if country in ("US", "CA"):
        queries.append("21 CFR 820.30 design controls risk analysis FDA QMSR design FMEA")
    elif country in ("EU", "UK"):
        queries.append("EU MDR Annex I GSPR essential requirements risk management benefit risk")
    elif country == "JP":
        queries.append("PMDA MHLW risk management medical device Ordinance 169 risk analysis")
    elif country == "AU":
        queries.append("TGA risk management ISO 14971 essential principles conformity assessment")
    elif country == "KR":
        queries.append("MFDS Korea risk management ISO 14971 medical device registration")
    if hazards_hint:
        queries.append(f"hazard {hazards_hint} medical device risk control mitigation")

    chunks = multi_query_faiss(
        store, queries, country, device_class=device_class or None, top_k=4, max_chunks=45
    )
    regulation_context = build_regulation_context(chunks, max_chunks=38)

    # ── LLM structured generation ────────────────────────────────────────────
    system_prompt = (
        "You are a Principal Risk Engineer certified in ISO 14971:2019 with 18 years preparing "
        "risk management files accepted by FDA, CE notified bodies (TÜV SÜD, BSI), and PMDA Japan.\n\n"
        "ISO 14971:2019 Harm Sequence methodology (§4):\n"
        "  Hazard → Hazardous Situation → Harm (must be a specific patient/user injury)\n"
        "  Do NOT conflate hazard (energy source, substance) with harm (tissue damage, death).\n"
        "  Example: Hazard='electrical energy' → Situation='insulation failure during use' → "
        "Harm='cardiac arrest from electrocution'\n\n"
        "Risk Matrix (ISO 14971 Annex D example):\n"
        "  Severity: 1=Negligible (no injury), 2=Marginal (minor reversible), 3=Serious (major "
        "injury, hospitalization), 4=Critical (permanent impairment), 5=Catastrophic (death)\n"
        "  Probability: 1=Improbable (<1/million), 2=Remote (<1/100k), 3=Occasional (<1/1000), "
        "4=Probable (<1/100), 5=Frequent (>1/100)\n\n"
        "Risk Control Hierarchy §6.2 (apply in order, prefer higher levels):\n"
        "  §6.2a: Inherently safe design (eliminate hazard from design)\n"
        "  §6.2b: Protective measures in device or manufacturing process\n"
        "  §6.2c: Information for safety (IFU warnings, labeling, training — least preferred)\n\n"
        "Residual risk: always assess post-control S×P. Control measures MUST measurably "
        "reduce either severity (containment, barriers) or probability (design change, interlocks).\n\n"
        "Benefit-risk (§7.4): written conclusion that benefits of intended use outweigh ALL residual "
        "risks; required for ALL EU MDR submissions and FDA Class III devices.\n\n"
        "IMPORTANT: Output hazards SPECIFIC to the described device — not generic boilerplate. "
        "If it's an orthopedic implant, the hazards differ from a diagnostic software application.\n\n"
        "CITATION RULE: Every regulation_cite MUST include the standard year AND specific clause "
        "(e.g. 'ISO 14971:2019 §5.5', 'ISO 10993-1:2018 §4.3', 'IEC 60601-1:2005+AMD2:2020 §8.4'). "
        "Never cite just 'ISO 14971' without the year and section.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Base every hazard and control measure on the REGULATORY TEXT provided below\n"
        "- Never infer requirements not explicitly stated in the provided text\n"
        "- If the regulatory text does not address a specific hazard category, do not include it\n"
        "- Every regulation_cite must reference text you were actually given\n\n"
        "Output ONLY valid JSON — no markdown fences, no prose, no explanation."
    )

    user_prompt = f"""Perform an ISO 14971:2019 risk analysis for:

Device Description: {device_description}
Intended Use: {intended_use}
Device Class: {device_class}
Regulatory Market: {country}
{f"Known hazards or concerns: {hazards_hint}" if hazards_hint else ""}

Apply the ISO 14971 §4 Harm Sequence methodology. Every hazard must be SPECIFIC to this device.
Use the regulatory text below for citations.

REGULATORY TEXT:
{regulation_context}

Return a JSON object with exactly these fields:
- hazard_analysis: array of 7-12 hazard entries specific to THIS device, each with:
  - hazard_id: "H-001", "H-002", etc.
  - hazard: specific energy source or substance (NOT "device failure" — be precise)
  - hazardous_situation: specific circumstance of patient/user/bystander exposure
  - harm: specific physical injury or health damage (death, injury type, severity)
  - severity: integer 1-5 (per ISO 14971 Annex D scale above)
  - probability: integer 1-5 (per ISO 14971 Annex D scale above)
  - control_measures: array of 3-4 specific, implementable risk control measures with
    reference to the applicable standard or test (e.g. "ISO 11135 sterilization validation,
    SAL 10⁻⁶", "IEC 60601-1-2 EMC testing limits verified")
  - control_level: exactly "§6.2a — Inherently safe design" or
    "§6.2b — Protective measures" or "§6.2c — Information for safety"
  - residual_severity: integer 1-5 post-control estimate
  - residual_probability: integer 1-5 post-control estimate
  - regulation_cite: specific ISO 14971 clause + applicable standard
- overall_verdict: "ACCEPTABLE" | "ALARP" | "UNACCEPTABLE"
  (NOTE: this will be OVERRIDDEN server-side by the canonical risk matrix — provide your assessment)
- benefit_risk_conclusion: 3-4 sentence ISO 14971 §7.4 justification specific to this device
- risk_management_plan_summary: 3-4 sentence summary of planned risk management lifecycle
- applicable_standards: array of all relevant standards including their year

Output ONLY valid JSON. Every hazard must be SPECIFIC to the described device — not generic."""

    try:
        raw = chat_completion(system_prompt, user_prompt)
        data = parse_llm_json(raw)
        if not isinstance(data, dict):
            data = {}

        hazards: list[HazardEntry] = []
        for h in data.get("hazard_analysis", []):
            sev = max(1, min(5, int(h.get("severity", 3))))
            prob = max(1, min(5, int(h.get("probability", 2))))
            # Clamp residual estimates
            res_sev = max(1, min(5, int(h.get("residual_severity", max(1, sev - 1)))))
            res_prob = max(1, min(5, int(h.get("residual_probability", max(1, prob - 1)))))

            entry = HazardEntry(
                hazard_id=h.get("hazard_id", f"H-{len(hazards)+1:03d}"),
                hazard=h.get("hazard", "Unspecified hazard"),
                hazardous_situation=h.get("hazardous_situation", ""),
                harm=h.get("harm", "Unspecified harm"),
                severity=sev,
                severity_label=_SEVERITY_LABELS[sev],
                probability=prob,
                probability_label=_PROBABILITY_LABELS[prob],
                # Compute from canonical matrix — NEVER from LLM
                risk_score=sev * prob,
                risk_level=_RISK_MATRIX.get((sev, prob), "ALARP"),
                control_measures=h.get("control_measures", []),
                control_level=h.get("control_level", _CONTROL_HIERARCHY[2]),
                residual_severity=res_sev,
                residual_probability=res_prob,
                residual_risk_level=_RISK_MATRIX.get((res_sev, res_prob), "ALARP"),
                regulation_cite=h.get("regulation_cite", "ISO 14971:2019"),
            )
            hazards.append(entry)

        # ── Aggregate risk counts ───────────────────────────────────────────
        unacceptable = sum(1 for h in hazards if h.risk_level == "UNACCEPTABLE")
        alarp = sum(1 for h in hazards if h.risk_level == "ALARP")
        acceptable = sum(1 for h in hazards if h.risk_level == "ACCEPTABLE")
        res_unacceptable = sum(1 for h in hazards if h.residual_risk_level == "UNACCEPTABLE")

        # Reality Checker: ACCEPTABLE only if ALL residual hazards ≤ ACCEPTABLE
        # UNACCEPTABLE if any residual hazard remains UNACCEPTABLE after controls
        if res_unacceptable > 0:
            computed_verdict = "UNACCEPTABLE"
        elif all(h.residual_risk_level == "ACCEPTABLE" for h in hazards):
            computed_verdict = "ACCEPTABLE"
        else:
            computed_verdict = "ALARP"

        return RiskManagementReport(
            device_description=device_description,
            intended_use=intended_use,
            device_class=device_class,
            country=country,
            hazard_analysis=hazards,
            total_hazards=len(hazards),
            unacceptable_count=unacceptable,
            alarp_count=alarp,
            acceptable_count=acceptable,
            residual_unacceptable=res_unacceptable,
            # Override LLM verdict with computed Reality Checker verdict
            overall_verdict=computed_verdict,
            benefit_risk_conclusion=data.get("benefit_risk_conclusion", ""),
            risk_management_plan_summary=data.get("risk_management_plan_summary", ""),
            applicable_standards=data.get(
                "applicable_standards",
                ["ISO 14971:2019", "ISO/TR 24971:2020"],
            ),
        )

    except Exception as e:
        logger.error("CRITICAL: RMA OpenAI/LLM failure — serving static fallback hazards: %s", e)
        report = _fallback_rma(device_description, intended_use, device_class, country)
        report.is_fallback = True
        return report


# ── FAISS-only fallback ───────────────────────────────────────────────────────


def _fallback_rma(
    device_description: str,
    intended_use: str,
    device_class: str,
    country: str,
) -> RiskManagementReport:
    """
    Minimal valid risk file built from FAISS-retrieved standards.
    Covers the 5 universal hazard categories applicable to all medical devices.
    """
    store = get_vector_store()
    cites: list[str] = []
    for r in store.search(
        "ISO 14971 risk analysis hazard identification harm sequence",
        country=country,
        top_k=3,
    ):
        cites.append(r.get("document_id") or "ISO 14971:2019")

    primary_cite = cites[0] if cites else "ISO 14971:2019"

    hazards = [
        HazardEntry(
            hazard_id="H-001",
            hazard="Mechanical failure of device components",
            hazardous_situation="Device component fractures during use in patient",
            harm="Tissue laceration or internal injury",
            severity=4, severity_label="Critical",
            probability=2, probability_label="Remote",
            risk_score=8,
            risk_level=_RISK_MATRIX[(4, 2)],
            control_measures=[
                "Design verification: fatigue and fracture testing per ISO 10993-1",
                "Material selection: validated biocompatible materials with fracture toughness data",
                "Labeling: contraindications for specific patient anatomies listed in IFU",
            ],
            control_level=_CONTROL_HIERARCHY[0],
            residual_severity=3, residual_probability=1,
            residual_risk_level=_RISK_MATRIX[(3, 1)],
            regulation_cite=primary_cite,
        ),
        HazardEntry(
            hazard_id="H-002",
            hazard="Biocompatibility failure — cytotoxic or sensitizing materials",
            hazardous_situation="Patient tissue contacts device material for extended period",
            harm="Allergic reaction, local tissue toxicity, systemic immune response",
            severity=3, severity_label="Serious",
            probability=2, probability_label="Remote",
            risk_score=6,
            risk_level=_RISK_MATRIX[(3, 2)],
            control_measures=[
                "ISO 10993-1 biocompatibility testing protocol (cytotoxicity, sensitization, genotoxicity)",
                "Material certification: only validated biocompatible grades used",
                "IFU: contraindications for known material sensitivities",
            ],
            control_level=_CONTROL_HIERARCHY[0],
            residual_severity=2, residual_probability=1,
            residual_risk_level=_RISK_MATRIX[(2, 1)],
            regulation_cite="ISO 10993-1:2018",
        ),
        HazardEntry(
            hazard_id="H-003",
            hazard="Sterility failure — contaminated device delivered to sterile field",
            hazardous_situation="Device with compromised sterile barrier implanted in patient",
            harm="Surgical site infection, sepsis",
            severity=4, severity_label="Critical",
            probability=1, probability_label="Improbable",
            risk_score=4,
            risk_level=_RISK_MATRIX[(4, 1)],
            control_measures=[
                "ISO 11135/11137: validated sterilization process with sterility assurance level (SAL) 10⁻⁶",
                "Sterile barrier: validated per ISO 11607 with integrity testing",
                "Labeling: visual sterile barrier check instructions in IFU",
            ],
            control_level=_CONTROL_HIERARCHY[0],
            residual_severity=4, residual_probability=1,
            residual_risk_level=_RISK_MATRIX[(4, 1)],
            regulation_cite="ISO 14971:2019 §4.4",
        ),
        HazardEntry(
            hazard_id="H-004",
            hazard="Use error — incorrect technique by clinician",
            hazardous_situation="Device deployed incorrectly due to inadequate training or unclear IFU",
            harm="Malpositioning, device migration, re-operation required",
            severity=3, severity_label="Serious",
            probability=3, probability_label="Occasional",
            risk_score=9,
            risk_level=_RISK_MATRIX[(3, 3)],
            control_measures=[
                "Usability engineering per IEC 62366-1: summative usability evaluation",
                "Simplified surgical technique with visual size/orientation indicators on device",
                "IFU with step-by-step illustrated surgical technique",
            ],
            control_level=_CONTROL_HIERARCHY[1],
            residual_severity=3, residual_probability=2,
            residual_risk_level=_RISK_MATRIX[(3, 2)],
            regulation_cite="IEC 62366-1:2015 + ISO 14971:2019 §5",
        ),
        HazardEntry(
            hazard_id="H-005",
            hazard="Electromagnetic interference with other active medical devices",
            hazardous_situation=(
                "Device interacts electromagnetically with implanted active device "
                "(e.g., pacemaker, neurostimulator)"
            ),
            harm="Interference with pacemaker causing cardiac arrhythmia",
            severity=5, severity_label="Catastrophic",
            probability=1, probability_label="Improbable",
            risk_score=5,
            risk_level=_RISK_MATRIX[(5, 1)],
            control_measures=[
                "IEC 60601-1-2:2014 EMC testing — emission and immunity limits verified",
                "Design: non-active components verified no ferromagnetic materials",
                "IFU: contraindication for concurrent use with listed active devices",
            ],
            control_level=_CONTROL_HIERARCHY[0],
            residual_severity=5, residual_probability=1,
            residual_risk_level=_RISK_MATRIX[(5, 1)],
            regulation_cite="IEC 60601-1-2:2014 + ISO 14971:2019 §4",
        ),
    ]

    unacceptable = sum(1 for h in hazards if h.risk_level == "UNACCEPTABLE")
    alarp_ct = sum(1 for h in hazards if h.risk_level == "ALARP")
    acceptable = sum(1 for h in hazards if h.risk_level == "ACCEPTABLE")
    res_unacceptable = sum(1 for h in hazards if h.residual_risk_level == "UNACCEPTABLE")

    return RiskManagementReport(
        device_description=device_description,
        intended_use=intended_use,
        device_class=device_class,
        country=country,
        hazard_analysis=hazards,
        total_hazards=len(hazards),
        unacceptable_count=unacceptable,
        alarp_count=alarp_ct,
        acceptable_count=acceptable,
        residual_unacceptable=res_unacceptable,
        overall_verdict="ALARP",
        benefit_risk_conclusion=(
            "Based on the intended use and foreseeable use, the identified risks are considered "
            "acceptable when weighed against the clinical benefits of the device. All residual risks "
            "have been reduced to ALARP through the risk controls documented above. "
            "Formal benefit-risk determination requires clinical evaluation per MEDDEV 2.7/1 Rev.4."
        ),
        risk_management_plan_summary=(
            f"Risk management activities for this {device_class} device include: pre-clinical "
            "verification testing (mechanical, biocompatibility, sterility), usability engineering "
            "per IEC 62366-1, post-market surveillance as per ISO 14971 §9, and periodic risk "
            "management file review. All risk management documentation maintained per ISO 13485 §7.1."
        ),
        applicable_standards=[
            "ISO 14971:2019",
            "ISO/TR 24971:2020",
            "ISO 13485:2016",
            "ISO 10993-1:2018",
            "IEC 62366-1:2015",
            "IEC 60601-1-2:2014",
        ],
    )
