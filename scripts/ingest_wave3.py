"""Wave 3: FDA AI/ML, WHO, UK MHRA, Health Canada guidance docs."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

HEADERS = {"User-Agent": "Mozilla/5.0"}

DOCS = [
    # ── FDA AI/ML ─────────────────────────────────────────────────────────────
    ("US_FDA_AIML_PCCP_FINAL_2024",
     "FDA Final Guidance: Marketing Submission Recommendations for a PCCP for AI-Enabled Device Software Functions (December 2024)",
     "https://www.fda.gov/media/166704/download", "US"),

    ("US_FDA_AIML_DEVICE_SW_DRAFT_2025",
     "FDA Draft Guidance: Artificial Intelligence-Enabled Device Software Functions (January 2025)",
     "https://www.fda.gov/media/184856/download", "US"),

    ("US_FDA_PCCP_DRAFT_2024",
     "FDA Draft Guidance: Predetermined Change Control Plans for Medical Devices (August 2024)",
     "https://www.fda.gov/media/180978/download", "US"),

    ("US_FDA_AIML_ACTION_PLAN_2021",
     "FDA AI/ML-Based Software as a Medical Device Action Plan (January 2021)",
     "https://www.fda.gov/media/145022/download", "US"),

    ("US_FDA_GMLP_2021",
     "FDA Good Machine Learning Practice for Medical Device Development — Guiding Principles (October 2021)",
     "https://www.fda.gov/media/153486/download", "US"),

    # ── WHO ───────────────────────────────────────────────────────────────────
    ("WHO_GMRF_TRS1003_ANNEX4",
     "WHO TRS 1003 Annex 4 — Global Model Regulatory Framework for Medical Devices Including IVDs",
     "https://cdn.who.int/media/docs/default-source/medicines/norms-and-standards/guidelines/regulatory-standards/trs1003-annex4-who-model-regulatory-framework-medical-devices.pdf",
     "STANDARDS"),

    ("WHO_GMRF_2022_REVISION",
     "WHO Global Model Regulatory Framework for Medical Devices Including IVDs (2022 Revision Draft)",
     "https://cdn.who.int/media/docs/default-source/biologicals/bs-2022.2425_global-model-regulatory-framework-for-medical-devices-including-ivds_11-july-2022-dl_14-july.pdf",
     "STANDARDS"),

    # ── UK MHRA ───────────────────────────────────────────────────────────────
    ("UK_MHRA_AI_IMPACT_2024",
     "MHRA Impact of Artificial Intelligence on the Regulation of Medical Products (2024)",
     "https://assets.publishing.service.gov.uk/media/662fce1e9e82181baa98a988/MHRA_Impact-of-AI-on-the-regulation-of-medical-products.pdf",
     "UK"),

    ("UK_MHRA_STANDALONE_SOFTWARE_2023",
     "MHRA Guidance: Medical Device Stand-Alone Software Including Apps (Including IVDMDs) (2023)",
     "https://assets.publishing.service.gov.uk/media/64a7d22d7a4c230013bba33c/Medical_device_stand-alone_software_including_apps__including_IVDMDs_.pdf",
     "UK"),

    ("UK_MHRA_AI_REGULATION_REPORT_2022",
     "UK Regulatory Horizons Council — The Regulation of Artificial Intelligence as a Medical Device (November 2022)",
     "https://assets.publishing.service.gov.uk/media/6384bf98e90e0778a46ce99f/RHC_regulation_of_AI_as_a_Medical_Device_report.pdf",
     "UK"),

    # ── Health Canada ─────────────────────────────────────────────────────────
    ("CA_HC_MLMD_PREMARKET_GUIDANCE",
     "Health Canada Pre-Market Guidance for Machine Learning-Enabled Medical Devices",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/pre-market-guidance-machine-learning-enabled-medical-devices/pre-market-guidance-machine-learning-enabled-medical-devices.pdf",
     "CA"),

    ("CA_HC_CYBERSECURITY_GUIDANCE",
     "Health Canada Guidance Document: Pre-Market Requirements for Medical Device Cybersecurity",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/cybersecurity-guidance.pdf",
     "CA"),

    ("CA_HC_CLINICAL_EVIDENCE_REQUIREMENTS",
     "Health Canada Guidance on Clinical Evidence Requirements for Medical Devices",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/clinical-evidence-requirements-medical-devices/clinical-evidence-requirements-medical-devices.pdf",
     "CA"),

    ("CA_HC_MDEL_GUI0016",
     "Health Canada Guidance on Medical Device Establishment Licensing (MDEL) — GUI-0016",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/compliance-enforcement/establishment-licences/directives-guidance-documents-policies/guidance-medical-device-establishment-licensing-0016/pub-eng.pdf",
     "CA"),

    ("CA_HC_ITA_GUIDANCE",
     "Health Canada Guidance: Applications for Medical Device Investigational Testing Authorizations (ITA)",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/investigational-testing-authorizations-guidance/guidance-document.pdf",
     "CA"),

    ("CA_HC_CLASS3_IVD_GUIDANCE",
     "Health Canada Guidance: Class 3 In Vitro Diagnostic Devices — New and Amendment Applications",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/international-medical-device-regulators-forum/Class-3-IVD-eng.pdf",
     "CA"),
]


def extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


store = get_vector_store()
existing = set(m.get("document_id", "") for m in store.metadata)
print(f"[lazy-load] Checking {len(DOCS)} docs...\n")

total_new = 0
skipped = 0

for doc_id, title, url, country in DOCS:
    if doc_id in existing:
        print(f"  SKIP {doc_id}")
        skipped += 1
        continue

    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  FAIL {doc_id}: {e}")
        continue

    if not (r.content[:4] == b"%PDF" or "pdf" in r.headers.get("content-type", "")):
        print(f"  FAIL {doc_id}: not a PDF")
        continue

    text = extract_pdf_text(r.content)
    words = len(text.split())
    if words < 50:
        print(f"  SKIP {doc_id}: too short ({words} words)")
        continue

    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=title,
        document_id=doc_id,
        source_url=url,
        device_classes=["Class I", "Class IIa", "Class IIb", "Class III",
                        "Class II", "Class IV", "SaMD", "IVD", "AI/ML"],
    )
    n = embed_and_index_chunks(chunks, vector_store=store)
    total_new += n
    print(f"  +{n:3d}  {doc_id}  [{country}]", flush=True)

print(f"\nDone: {skipped} skipped, {total_new} new chunks added")
