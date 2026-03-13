"""
OrthoLink — Monitored Regulatory Documents Registry.

Provides the canonical list of regulatory documents that the RAA crew
monitors for changes.  Each entry contains:

  document_id    — stable unique slug used as the FAISS document key
  source_url     — public government URL the scraper fetches
  regulation_name — human-readable name of the regulation
  country        — ISO country code (uppercase)

PRD §4.6: 15 markets in scope.

Usage:
    from app.ingestion.monitored_docs import get_monitored_docs, get_monitored_doc

    # All docs for a country
    docs = get_monitored_docs("US")

    # A specific document
    doc = get_monitored_doc("EU", "EU_MDR_2017_745")
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# Format: {country: [{document_id, source_url, regulation_name}]}
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, list[dict]] = {
    "US": [
        {
            "document_id": "US_FDA_QMSR_2026",
            "source_url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820",
            "regulation_name": "FDA 21 CFR Part 820 — Quality Management System Regulation (QMSR, Feb 2026)",
        },
        {
            "document_id": "US_FDA_21CFR814",
            "source_url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-814",
            "regulation_name": "FDA 21 CFR Part 814 — Premarket Approval (PMA)",
        },
        {
            "document_id": "US_FDA_510K_GUIDANCE",
            "source_url": "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k",
            "regulation_name": "FDA 510(k) Premarket Notification Guidance",
        },
        {
            "document_id": "US_FDA_CDS_GUIDANCE_2026",
            "source_url": "https://www.fda.gov/media/211464/download",
            "regulation_name": "FDA Guidance — Clinical Decision Support Software (Jan 2026)",
        },
        {
            "document_id": "US_FDA_BENEFIT_RISK_GUIDANCE",
            "source_url": "https://www.fda.gov/media/99769/download",
            "regulation_name": "FDA Guidance — Benefit-Risk Determinations in PMA (ISO 14971 framework)",
        },
        {
            "document_id": "US_FDA_QMSR_PREAMBLE",
            "source_url": "https://www.federalregister.gov/documents/2022/02/23/2022-03567/medical-device-quality-system-regulation",
            "regulation_name": "FDA QMSR Final Rule Preamble — ISO 13485 Alignment (2022)",
        },
    ],
    "EU": [
        {
            "document_id": "EU_MDR_2017_745",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0745",
            "regulation_name": "EU Medical Device Regulation 2017/745 (MDR)",
        },
        {
            "document_id": "EU_IVDR_2017_746",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0746",
            "regulation_name": "EU In Vitro Diagnostic Regulation 2017/746 (IVDR)",
        },
        {
            "document_id": "EU_MDR_2024_1860",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1860",
            "regulation_name": "EU Regulation 2024/1860 — MDR Transitional Period Extension (to Dec 2028)",
        },
        {
            "document_id": "EU_MDR_MEDDEV_271_REV4",
            "source_url": "https://ec.europa.eu/health/md_sector/new_approach/guidance_en",
            "regulation_name": "EU MEDDEV 2.7.1 Rev 4 — Clinical Evaluation Guide",
        },
        {
            "document_id": "EU_MDCG_2021_24_PMCF",
            "source_url": "https://health.ec.europa.eu/system/files/2021-11/mdcg_2021-24_en_0.pdf",
            "regulation_name": "EU MDCG 2021-24 — Post-Market Clinical Follow-up (PMCF) Guidance",
        },
    ],
    "UK": [
        {
            "document_id": "UK_MHRA_MDR2002",
            "source_url": "https://www.legislation.gov.uk/uksi/2002/618/contents",
            "regulation_name": "UK Medical Devices Regulations 2002 (SI 2002/618)",
        },
        {
            "document_id": "UK_MHRA_GUIDANCE",
            "source_url": "https://www.gov.uk/guidance/regulating-medical-devices-in-the-uk",
            "regulation_name": "MHRA — Regulating Medical Devices in the UK",
        },
    ],
    "UA": [
        {
            "document_id": "UA_MOH_ORDER690",
            "source_url": "https://zakon.rada.gov.ua/laws/show/z1931-12",
            "regulation_name": "Ukraine MoH Order 690 — Medical Device Registration",
        },
        {
            "document_id": "UA_MOH_ORDER522",
            "source_url": "https://zakon.rada.gov.ua/laws/show/522-2014-%D0%BF",
            "regulation_name": "Ukraine CMU Resolution 522 — Medical Device Market Surveillance",
        },
    ],
    "IN": [
        {
            "document_id": "IN_MDR_2017",
            "source_url": "https://cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/",
            "regulation_name": "India Medical Devices Rules 2017 (MDR 2017)",
        },
        {
            "document_id": "IN_CDSCO_GUIDANCE",
            "source_url": "https://cdsco.gov.in/opencms/opencms/en/Guidance-Documents/Guidance-Documents/",
            "regulation_name": "CDSCO Medical Device Guidance Documents",
        },
        {
            "document_id": "IN_CDSCO_SAMD_SOFTWARE_2025",
            "source_url": "https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/UploadPublic_NoticesFiles/Draft%20guidance%20document%20on%20Medical%20Device%20Software%2021%2010%202025.pdf",
            "regulation_name": "CDSCO Draft Guidance — Medical Device Software & SaMD (Oct 2025)",
        },
    ],
    "CA": [
        {
            "document_id": "CA_MDR_SOR98_282",
            "source_url": "https://laws-lois.justice.gc.ca/eng/regulations/SOR-98-282/",
            "regulation_name": "Canada Medical Devices Regulations SOR/98-282",
        },
        {
            "document_id": "CA_HEALTH_CANADA_GUIDANCE",
            "source_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/guidance-documents.html",
            "regulation_name": "Health Canada — Medical Device Guidance Documents",
        },
        {
            "document_id": "CA_HEALTH_CANADA_CLASS_IIIIV_GUIDANCE",
            "source_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents/guidance-document-medical-device-licensing.html",
            "regulation_name": "Health Canada — Medical Device Licensing Guidance (Class II-IV)",
        },
        {
            "document_id": "CA_HEALTH_CANADA_MDB_OVERVIEW",
            "source_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/guidance-documents.html",
            "regulation_name": "Health Canada Medical Devices Bureau — Regulatory Overview and Guidance Index",
        },
    ],
    "AU": [
        {
            "document_id": "AU_TGA_ARTG",
            "source_url": "https://www.legislation.gov.au/Details/F2021C00376",
            "regulation_name": "Australia Therapeutic Goods (Medical Devices) Regulations 2002",
        },
        {
            "document_id": "AU_TGA_GUIDANCE",
            "source_url": "https://www.tga.gov.au/how-we-regulate/manufacturing/regulation-medical-devices",
            "regulation_name": "TGA — Medical Device Regulation Overview",
        },
    ],
    "JP": [
        {
            "document_id": "JP_PMDA_PMDACT",
            "source_url": "https://www.pmda.go.jp/english/review-services/r-d/0006.html",
            "regulation_name": "Japan Act on Securing Quality, Efficacy and Safety of Products including Pharmaceuticals and Medical Devices",
        },
        {
            "document_id": "JP_MHLW_GUIDANCE",
            "source_url": "https://www.mhlw.go.jp/english/policy/health-medical/pharmaceuticals/index.html",
            "regulation_name": "MHLW — Medical Device Regulatory Guidance",
        },
    ],
    "CN": [
        {
            "document_id": "CN_NMPA_ORDER739",
            "source_url": "https://www.nmpa.gov.cn/ylqx/ylqxgzdt/",
            "regulation_name": "China NMPA Medical Device Registration Regulations",
        },
        {
            "document_id": "CN_NMPA_CLASSIFICATION",
            "source_url": "https://www.nmpa.gov.cn/ylqx/ylqxfl/",
            "regulation_name": "China NMPA Medical Device Classification Rules",
        },
    ],
    "BR": [
        {
            "document_id": "BR_ANVISA_RDC185",
            "source_url": "https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2022/anvisa-atualiza-norma-sobre-registro-de-produtos-para-saude",
            "regulation_name": "Brazil ANVISA RDC 185 — Medical Device Registration",
        },
        {
            "document_id": "BR_ANVISA_LEGISLATION",
            "source_url": "https://www.gov.br/anvisa/pt-br/acessoainformacao/legislacao/produtos-para-saude",
            "regulation_name": "ANVISA — Medical Device (Produtos para Saúde) Legislation",
        },
    ],
    "KR": [
        {
            "document_id": "KR_MFDS_MDA",
            "source_url": "https://www.mfds.go.kr/eng/index.do",
            "regulation_name": "South Korea Ministry of Food and Drug Safety — Medical Device Act",
        },
        {
            "document_id": "KR_MFDS_APPROVAL",
            "source_url": "https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=73047",
            "regulation_name": "MFDS — Medical Device Approval and License Guidance",
        },
    ],
    "CH": [
        {
            "document_id": "CH_SWISSMEDIC_MDO",
            "source_url": "https://www.swissmedic.ch/swissmedic/en/home/medical-devices/market-authorisation/classification-and-conformity-assessment/classification-of-medical-devices.html",
            "regulation_name": "Swissmedic — Medical Devices Ordinance (MedDO)",
        },
        {
            "document_id": "CH_SWISSMEDIC_GUIDANCE",
            "source_url": "https://www.swissmedic.ch/swissmedic/en/home/medical-devices/market-authorisation.html",
            "regulation_name": "Swissmedic — Medical Device Market Authorisation Guidance",
        },
    ],
    "MX": [
        {
            "document_id": "MX_COFEPRIS_DEVICES",
            "source_url": "https://www.gob.mx/cofepris/acciones-y-programas/dispositivos-medicos",
            "regulation_name": "Mexico COFEPRIS — Dispositivos Médicos (Medical Device Regulation)",
        },
        {
            "document_id": "MX_SSA_NOM137",
            "source_url": "https://www.dof.gob.mx/nota_detalle.php?codigo=5476148&fecha=28/09/2016",
            "regulation_name": "NOM-137-SSA1-2008 — Requirements for Medical Device Marketing",
        },
    ],
    "RU": [
        {
            "document_id": "RU_ROSZDRAV_FZ323",
            "source_url": "https://roszdravnadzor.gov.ru/services/licenses",
            "regulation_name": "Russia Federal Law 323-FZ — Fundamentals of Health Protection",
        },
        {
            "document_id": "RU_ROSZDRAV_REGISTRATION",
            "source_url": "https://roszdravnadzor.gov.ru/medical_devices",
            "regulation_name": "Roszdravnadzor — Medical Device State Registration",
        },
    ],
    "SA": [
        {
            "document_id": "SA_SFDA_MDR",
            "source_url": "https://www.sfda.gov.sa/en/medical-devices",
            "regulation_name": "Saudi Arabia SFDA — Medical Device Regulation",
        },
        {
            "document_id": "SA_SFDA_CLASSIFICATION",
            "source_url": "https://www.sfda.gov.sa/en/node/94766",
            "regulation_name": "SFDA — Medical Device Classification and Conformity Assessment",
        },
    ],
    # ── International Standards & Frameworks (STANDARDS pseudo-country) ────
    # Real standards text is paywalled (ISO/IEC). We monitor public guidance docs
    # that describe and reference these standards (FDA, WHO, IMDRF — all free).
    "STANDARDS": [
        {
            "document_id": "WHO_MEDEVIS_CLASSIFICATION",
            "source_url": "https://www.who.int/teams/health-product-and-policy-standards/assistive-and-medical-technology/medical-devices/medevis",
            "regulation_name": "WHO Medical Device Information System (MeDevIS) — Global Classification Reference",
        },
        {
            "document_id": "IMDRF_SAMD_FRAMEWORK",
            "source_url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-131209-samd-key-definitions-140901.pdf",
            "regulation_name": "IMDRF — Software as a Medical Device (SaMD): Key Definitions and Framework",
        },
        {
            "document_id": "IMDRF_LABELLING_N18",
            "source_url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-121102-labelling-n18.pdf",
            "regulation_name": "IMDRF N18 FINAL — Principles of Labelling for Medical Devices and IVDs",
        },
        {
            "document_id": "JOHNER_AI_MEDICAL_DEVICE_GUIDELINE",
            "source_url": "https://raw.githubusercontent.com/johner-institut/ai-guideline/master/Guideline-AI-Medical-Devices_EN.md",
            "regulation_name": "Johner Institut — AI Medical Device Guideline (EU MDR + IVDR + AI Act 2024/1689)",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_monitored_docs(country: str) -> list[dict]:
    """
    Return all monitored documents for a country (ISO code, case-insensitive).
    Returns an empty list if the country is not in the registry.
    """
    return _REGISTRY.get(country.strip().upper(), [])


def get_monitored_doc(country: str, document_id: str) -> Optional[dict]:
    """
    Return a single monitored document by country and document_id.
    Returns None if not found.
    """
    for doc in get_monitored_docs(country):
        if doc["document_id"] == document_id:
            return doc
    return None


_NON_COUNTRY_SECTIONS = {"STANDARDS"}


def list_all_countries() -> list[str]:
    """Return the sorted list of monitored country codes (excludes pseudo-sections like STANDARDS)."""
    return sorted(k for k in _REGISTRY.keys() if k not in _NON_COUNTRY_SECTIONS)


def get_all_docs() -> list[dict]:
    """Return every document across all countries (flat list, excludes STANDARDS section)."""
    out = []
    for country, docs in _REGISTRY.items():
        if country in _NON_COUNTRY_SECTIONS:
            continue
        for d in docs:
            out.append({**d, "country": country})
    return out
