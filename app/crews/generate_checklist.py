"""
OrthoLink ROA Crew — Generate role-split compliance checklist
POST /api/v1/generate-checklist

Rules: Manufacturer ∩ Importer = ∅; QMSR for US (Feb 2026); UDI per country.
Returns ChecklistItem[]; async with job_id.

Role obligations knowledge:
  MANUFACTURER: Technical documentation, QMS (ISO 13485), UDI assignment,
    clinical evaluation, PMS, regulatory registration, DoC/510(k) clearance
  IMPORTER (EU MDR Art 13): Verify CE mark + DoC; verify EU AR exists (Art 11);
    register in EUDAMED (Art 30); translate labeling/IFU; store DoC 10 years
  DISTRIBUTOR (EU MDR Art 14): Verify CE + labeling before supply; complaint forwarding
  US IMPORTER: Designate US Agent (21 CFR 807.40); register establishment (807.20)
  APOSTILLE: Required for Ukraine MHU, Saudi SFDA, Indonesia BPOM, many LATAM markets
"""

import json
import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from app.crews.utils import build_regulation_context, multi_query_faiss
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DISCLAIMER = "Reference tool only. Verify with official sources."


class ChecklistItem(BaseModel):
    """Single checklist item — PRD ROA output."""

    item: str = Field(..., description="Document or requirement name")
    role: str = Field(..., description="MANUFACTURER | IMPORTER | EXPORTER | DISTRIBUTOR | BOTH")
    regulation_cite: str = Field(..., description="Exact legal citation (e.g. 21 CFR 820.30)")
    deadline_days: int = Field(default=0, description="Typical days to complete")
    apostille_required: bool = Field(default=False)
    notes: str = Field(default="", description="AI-generated guidance")


class RoleSplitChecklist(BaseModel):
    """ROA endpoint result."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    country: str = ""
    device_class: str = ""
    items: list[ChecklistItem] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)
    is_fallback: bool = Field(
        default=False,
        description="True if LLM/FAISS failed and fallback data was served. "
        "Frontend MUST display degraded-data warning when True.",
    )


def _parse_crew_output_to_items(raw: str) -> list[ChecklistItem]:
    """Extract JSON array from CrewAI agent output using Universal Extraction Manifold."""
    if not raw or not raw.strip():
        return []
    from app.crews.utils import extract_clean_json
    try:
        clean = extract_clean_json(raw)
        data = json.loads(clean)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("ROA _parse_crew_output_to_items: extraction failed: %s", exc)
        return []
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    items = []
    for o in data:
        if not isinstance(o, dict):
            continue
        role = (o.get("role") or "BOTH").upper()
        if role not in ("MANUFACTURER", "IMPORTER", "EXPORTER", "DISTRIBUTOR", "BOTH"):
            role = "BOTH"
        items.append(
            ChecklistItem(
                item=str(o.get("item", "")),
                role=role,
                regulation_cite=str(o.get("regulation_cite", "")),
                deadline_days=int(o.get("deadline_days", 0)),
                apostille_required=bool(o.get("apostille_required", False)),
                notes=str(o.get("notes", "")),
            )
        )
    return items


def _validate_device_scope(device_type: Optional[str], country: str) -> Optional[str]:
    """FAISS query to verify device_type is a plausible medical device.

    Returns an error message if the device is not recognized, None if valid or not specified.
    """
    if not device_type or not device_type.strip():
        return None  # No type given, proceed without validation
    store = get_vector_store()
    results = store.search(
        f"{device_type} medical device",
        country=country,
        top_k=3,
        active_only=True,
    )
    if not results or float(results[0].get("score", 0)) < 0.25:
        return (
            f"'{device_type}' is not recognized as a medical device type in our regulatory database. "
            "Please provide a valid medical device type (e.g., 'orthopedic implant', "
            "'blood glucose monitor', 'surgical instrument')."
        )
    return None


def _run_roa_crew(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
) -> "RoleSplitChecklist":
    """CrewAI path: ROA agent generates role-split checklist via crew.kickoff()."""
    from crewai import Crew, Process, Task

    from app.agents.roa_agent import get_roa_agent

    agent = get_roa_agent()
    task = Task(
        description=(
            f"Generate a role-split regulatory compliance checklist.\n"
            f"Country: {country}, Device Class: {device_class}\n"
            f"{'Device type: ' + device_type if device_type else ''}\n\n"
            "Split obligations between MANUFACTURER and IMPORTER — no overlap. "
            "For each item, specify: regulation_cite, deadline_days, apostille_required. "
            "Include QMSR requirements for US, UDI obligations where applicable."
        ),
        expected_output="Role-split checklist JSON with MANUFACTURER/IMPORTER/BOTH assignments",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    crew.kickoff(inputs={"country": country, "device_class": device_class})
    logger.info("ROA CrewAI crew completed; running direct pipeline for structured output")
    return run_roa_checklist(country, device_class, device_type)


def run_roa_checklist(
    country: str,
    device_class: str,
    device_type: Optional[str] = None,
    use_crewai: bool = False,
) -> RoleSplitChecklist:
    """
    Generate role-split checklist: parallel FAISS retrieval → single gpt-4o call.
    Manufacturer ∩ Importer = ∅. QMSR + UDI for US. ~5-10s vs 60-90s with CrewAI.

    Args:
        use_crewai: When True, runs the ROA CrewAI agent first for narrative checklist,
                    then produces the structured result via the direct pipeline.
    """
    if use_crewai:
        return _run_roa_crew(country, device_class, device_type)

    from app.tools.llm import chat_completion

    # Anti-hallucination: refuse non-medical-device inputs
    scope_error = _validate_device_scope(device_type, country)
    if scope_error:
        return RoleSplitChecklist(
            job_id=str(uuid.uuid4()),
            country=country,
            device_class=device_class or "",
            items=[],
            disclaimer=scope_error,
        )

    store = get_vector_store()

    queries = [
        # Registration and market access
        "registration requirements manufacturer importer medical device approval",
        "market authorization product registration regulatory submission",
        # QMS and quality obligations
        "quality management system QMS ISO 13485 manufacturer obligations",
        "quality system regulation manufacturer requirements documentation",
        # Labeling and identification
        "labeling instructions for use IFU requirements language",
        "unique device identification UDI labeling database",
        # Post-market obligations
        "post-market surveillance vigilance reporting manufacturer importer",
        "complaint handling adverse event reporting obligations",
        # Technical documentation
        "clinical evaluation technical documentation manufacturer",
        "design verification validation testing requirements",
        # Import/distribution specific
        "importer obligations distributor requirements compliance",
        "authorized representative local agent responsibilities",
        "foreign manufacturer registration requirements importer",
        # Apostille and notarization
        "apostille notarization certified translation regulatory submission",
        # Country-specific
        "country specific registration requirements medical device import",
    ]

    if country.upper() == "US":
        queries += [
            "QMSR 21 CFR 820 quality system regulation 2026 manufacturer",
            "510k premarket notification 21 CFR 807",
            "FDA establishment registration 21 CFR 807.20 device listing",
            "US agent 21 CFR 807.40 foreign manufacturer",
            "UDI FDA 21 CFR 830 GUDID database",
        ]
    elif country.upper() == "EU":
        queries += [
            "EU MDR Article 13 importer obligations CE mark verification",
            "EU MDR Article 14 distributor obligations language labeling",
            "EU MDR Article 11 authorized representative non-EU manufacturer",
            "EUDAMED registration economic operator Article 30",
            "EU MDR declaration of conformity DoC requirements",
        ]
    elif country.upper() in ("UA", "UKRAINE"):
        queries += [
            "Ukraine Ministry of Health registration medical device MOH",
            "Ukraine apostille requirements certified documents",
        ]
    elif country.upper() == "IN":
        queries += [
            "CDSCO India medical device registration ICMED",
            "India MDR 2017 importer registration Class",
        ]
    elif country.upper() == "JP":
        queries += [
            "PMDA Japan registration shonin marketing approval",
            "Japan MHLW local authorized importer Class",
        ]
    elif country.upper() == "SA":
        queries += [
            "SFDA Saudi Arabia medical device registration MDMA",
            "Saudi Arabia apostille requirements",
        ]

    chunks = multi_query_faiss(
        store, queries, country, device_class=device_class or None, top_k=5, max_chunks=50
    )
    regulation_context = build_regulation_context(chunks, max_chunks=42)

    system_prompt = (
        "You are a Senior Regulatory Affairs Manager at a global medical device company. "
        "You write compliance checklists used operationally by manufacturers, importers, "
        "and distributors placing Class I through III devices in 15+ markets.\n\n"
        "MANUFACTURER core obligations (all markets):\n"
        "  • Technical documentation + design controls (ISO 14971, IEC 62304 if software)\n"
        "  • QMS per ISO 13485 or national equivalent\n"
        "  • Clinical evaluation / clinical data package\n"
        "  • UDI assignment and database registration (EUDAMED/GUDID/JUDI)\n"
        "  • PMS system + PMSR/PSUR per device class\n"
        "  • Declaration of Conformity / 510(k) / registration certificate\n\n"
        "IMPORTER obligations — EU MDR Art 13 (different from Manufacturer!):\n"
        "  • Verify device has CE mark + valid DoC BEFORE placing on market\n"
        "  • Verify manufacturer has EU Authorized Representative (Art 11)\n"
        "  • Register as economic operator in EUDAMED (Art 30) before first supply\n"
        "  • Translate labeling and IFU into language of each EU member state (Art 13(2))\n"
        "  • Store copy of DoC for 10 years from last device placed on market\n"
        "  • Report non-compliant devices and serious incidents to Competent Authority\n\n"
        "DISTRIBUTOR — EU MDR Art 14:\n"
        "  • Verify CE mark, language of labeling/IFU matches state of supply\n"
        "  • Forward complaints and reports to manufacturer/importer\n"
        "  • No modification of device or its conditions\n\n"
        "US-specific rules (QMSR effective Feb 2026):\n"
        "  • Foreign manufacturers MUST designate US Agent per 21 CFR 807.40\n"
        "  • Establishment registration + device listing required (807.20/807.25)\n"
        "  • UDI: GUDID database, direct-part marking for implants\n"
        "  • QMSR aligns with ISO 13485 — QMS must include design controls (820.30)\n\n"
        "APOSTILLE required for: Ukraine (MOH registration), Saudi Arabia (SFDA), "
        "Indonesia (BPOM), Brazil (ANVISA), Argentina (ANMAT) — manufacturer declarations "
        "and power of attorney documents typically need legalization.\n\n"
        "IRON RULE: Manufacturer ∩ Importer = EMPTY SET. Every item belongs to ONE role. "
        "Only use BOTH when the IDENTICAL obligation applies to both (e.g. UDI submission "
        "when manufacturer AND importer must each register separately).\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "- Base every checklist item on the REGULATORY TEXT provided below\n"
        "- Every regulation_cite must reference text you were actually given\n"
        "- Never infer obligations not explicitly stated in the provided text\n"
        "- If you cannot find a citation in the text, do not include the item\n\n"
        "Output ONLY a valid JSON array — no markdown, no prose, no explanation."
    )

    user_prompt = f"""Generate a complete role-split compliance checklist for:
Country: {country}
Device Class: {device_class or "II"}
{f"Device Type: {device_type}" if device_type else ""}

Use ONLY the regulatory text below for citations. Every item must cite a specific article/section.

REGULATORY TEXT:
{regulation_context}

Output a JSON array of 18-28 items. Each object must have:
  - item: specific document or obligation name
  - role: MANUFACTURER | IMPORTER | EXPORTER | DISTRIBUTOR | BOTH
  - regulation_cite: exact citation (e.g. "EU MDR Art 13(2)", "21 CFR 820.30", "ISO 13485 §7.3")
  - deadline_days: integer days from start of registration process (0 if ongoing)
  - apostille_required: boolean
  - notes: regulatory nuance, authority to submit to, common rejection reason, or practical tip

Rules:
  - Manufacturer ∩ Importer = empty set (no duplicate items across roles)
  - Use BOTH only for obligations that genuinely apply identically to both
  - Include obligations for BEFORE market entry, DURING registration, AND ongoing post-market
  - For {country}: include country-specific documentation (local language, local agent, notarization)
{"  - Include QMSR/21 CFR 820 and UDI (21 CFR 830) for US" if country.upper() == "US" else ""}
{"  - Include EU MDR Art 13 importer and Art 14 distributor obligations" if country.upper() == "EU" else ""}

Output ONLY a valid JSON array."""

    used_fallback = False
    try:
        raw = chat_completion(system_prompt, user_prompt)
        items = _parse_crew_output_to_items(raw)
        if not items:
            logger.error("CRITICAL: ROA LLM returned empty/unparseable — serving FAISS fallback")
            items = _fallback_checklist(country, device_class)
            used_fallback = True
    except Exception as e:
        logger.error("CRITICAL: ROA OpenAI/LLM failure — serving static fallback: %s", e)
        items = _fallback_checklist(country, device_class)
        used_fallback = True

    return RoleSplitChecklist(
        job_id=str(uuid.uuid4()),
        country=country,
        device_class=device_class or "",
        items=items,
        disclaimer=DISCLAIMER,
        is_fallback=used_fallback,
    )


_STATIC_FALLBACK_ITEMS: list[dict] = [
    {
        "item": "Product registration application",
        "role": "BOTH",
        "cite": "National competent authority",
        "notes": "Core market-entry requirement for all medical devices.",
    },
    {
        "item": "Quality Management System (ISO 13485:2016)",
        "role": "MANUFACTURER",
        "cite": "ISO 13485:2016",
        "notes": "QMS certification required for most regulated markets.",
    },
    {
        "item": "Technical documentation / design dossier",
        "role": "MANUFACTURER",
        "cite": "ISO/IEC 63502",
        "notes": "Device description, intended use, risk management file.",
    },
    {
        "item": "Labelling in local language",
        "role": "BOTH",
        "cite": "Local labelling regulations",
        "notes": "Labels must comply with local language requirements.",
    },
    {
        "item": "Authorized local representative designation",
        "role": "IMPORTER",
        "cite": "Local representative regulations",
        "notes": "Required in EU, UK, AU, CA and other markets.",
    },
]


def _fallback_checklist(country: str, device_class: str) -> list[ChecklistItem]:
    """When LLM fails, use vector store + minimal structure so PDF export still works.

    Falls back to a static baseline if the vector store is unavailable (e.g. test
    environment with a fake OpenAI API key, or FAISS index not yet loaded).
    """
    store = get_vector_store()
    queries = [
        "registration requirements documents manufacturer importer",
        "quality management system QMS",
        "UDI unique device identification",
    ]
    if country.upper() == "US":
        queries.extend(["QMSR 2026", "21 CFR 820"])
    seen: set[str] = set()
    items: list[ChecklistItem] = []

    try:
        for q in queries:
            for r in store.search(q, country=country, device_class=device_class or None, top_k=5):
                cid = r.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    section = r.get("section_path") or r.get("document_id") or "Regulation"
                    items.append(
                        ChecklistItem(
                            item=section,
                            role="BOTH",
                            regulation_cite=section,
                            deadline_days=0,
                            apostille_required=False,
                            notes=r.get("text", "")[:200],
                        )
                    )
    except Exception as exc:  # noqa: BLE001
        # Covers openai.AuthenticationError (fake test key), APIError, FAISS load errors, etc.
        logger.error(
            "CRITICAL: ROA FAISS fallback also failed (%s: %s); falling to static baseline",
            type(exc).__name__,
            exc,
        )

    if not items:
        # Ultimate static floor — guarantees PDF export never crashes in any environment.
        for s in _STATIC_FALLBACK_ITEMS:
            items.append(
                ChecklistItem(
                    item=s["item"],
                    role=s["role"],  # type: ignore[arg-type]
                    regulation_cite=s["cite"],
                    deadline_days=0,
                    apostille_required=False,
                    notes=s["notes"],
                )
            )

    return items[:20]
