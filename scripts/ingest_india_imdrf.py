"""Ingest India CDSCO guidance docs + IMDRF global standards."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── India CDSCO ──────────────────────────────────────────────────────────────
INDIA_DOCS = [
    ("IN_MDR_2017_FULL",
     "Medical Devices Rules 2017 (as amended 2022) — Full Consolidated Text",
     "https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2022/m_device/Medical%20Devices%20Rules,%202017.pdf",
     "IN"),

    ("IN_CDSCO_MD_FAQ_2024_ADD01",
     "CDSCO FAQ on Medical Devices Rules 2017 — Addendum No. 01 (April 2025)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Addendum-0152.pdf",
     "IN"),

    ("IN_CDSCO_MD_FAQ_2024_ADD03",
     "CDSCO FAQ on Medical Devices Rules 2017 — Addendum No. 03 (November 2025)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Addendum-03-to-FAQ-on-Medical-Devices-Rules-2017.pdf",
     "IN"),

    ("IN_CDSCO_MD_FAQ_2024",
     "CDSCO FAQ on Medical Devices Rules 2017 (2024 Edition)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/mdfaq24.pdf",
     "IN"),

    ("IN_CDSCO_FSC_GUIDANCE_2024",
     "CDSCO Guidance Document for Free Sale Certificate (FSC) for Licensed Medical Devices in India",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/fscguidancerev2024.pdf",
     "IN"),

    ("IN_CDSCO_SUBMISSION_FORMAT_REGISTRATION",
     "CDSCO Guidance on Common Submission Format for Registration of Notified Medical Devices",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Guidance2.pdf",
     "IN"),

    ("IN_CDSCO_SUBMISSION_FORMAT_MANUFACTURING",
     "CDSCO Guidance on Common Submission Format for Manufacturing of Notified Medical Devices (CLAA Scheme)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Guidance1.pdf",
     "IN"),

    ("IN_CDSCO_IVD_FAQ_2024",
     "CDSCO FAQ on In-Vitro Diagnostic (IVD) Medical Devices (2024)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/IVD/FAQs/FAQ-ivd2024.pdf",
     "IN"),

    ("IN_CDSCO_GROUPING_GUIDANCE",
     "CDSCO Guidance Document on Grouping of Medical Devices and IVD Medical Devices",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Guidelines_Grouping_of_MDandIVD.pdf",
     "IN"),

    ("IN_CDSCO_IVD_PER_2025",
     "CDSCO Guidance on Performance Evaluation Requirements (PER) for IVD Medical Devices (June 2025)",
     "https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-on-PER-updated-04-June-2025.pdf",
     "IN"),
]

# ── IMDRF Global Standards ────────────────────────────────────────────────────
IMDRF_DOCS = [
    ("IMDRF_N88_AIML_GMLP_2025",
     "IMDRF N88 (2025) — Good Machine Learning Practices for Medical Device Development Using AI/ML",
     "https://www.imdrf.org/sites/default/files/2025-01/IMDRF_AIML%20WG_GMLP_N88%20Final_0.pdf",
     "STANDARDS"),

    ("IMDRF_N47_ED2_2024",
     "IMDRF N47 Edition 2 (2024) — Essential Principles of Safety and Performance of Medical Devices and IVD Medical Devices",
     "https://www.imdrf.org/sites/default/files/2024-04/IMDRF%20GRRP%20WG%20N47%20(Edition%202).pdf",
     "STANDARDS"),

    ("IMDRF_N52_ED2_2024",
     "IMDRF N52 Edition 2 (2024) — Regulatory Requirements for Medical Devices: A Framework for Convergence",
     "https://www.imdrf.org/sites/default/files/2024-04/IMDRF%20GRRP%20WG%20N52%20(Edition%202).pdf",
     "STANDARDS"),

    ("IMDRF_N71_ED2_2024",
     "IMDRF N71 Edition 2 (2024) — Global Principles for Regulatory Convergence",
     "https://www.imdrf.org/sites/default/files/2024-04/IMDRF%20GRRP%20WG%20N71%20(Edition%202)_0.pdf",
     "STANDARDS"),

    ("IMDRF_N41_SAMD_CLINICAL_EVAL_2017",
     "IMDRF N41 (2017) — Software as a Medical Device (SaMD): Clinical Evaluation",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-170921-samd-n41-clinical-evaluation_1.pdf",
     "STANDARDS"),

    ("IMDRF_N23_SAMD_QMS_2015",
     "IMDRF N23 (2015) — Software as a Medical Device (SaMD): Application of Quality Management System",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-151002-samd-qms.pdf",
     "STANDARDS"),

    ("IMDRF_N60_CYBERSECURITY_2020",
     "IMDRF N60 (2020) — Principles and Practices for Medical Device Cybersecurity",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200318-pp-mdc-n60.pdf",
     "STANDARDS"),

    ("IMDRF_N70_LEGACY_CYBER_2023",
     "IMDRF N70 (2023) — Principles and Practices for the Cybersecurity of Legacy Medical Devices",
     "https://www.imdrf.org/sites/default/files/2023-04/IMDRF%20Principles%20and%20Practices%20of%20Cybersecurity%20for%20%20Legacy%20Medical%20Devices%20Final%20(N70)_0.pdf",
     "STANDARDS"),

    ("IMDRF_N58_PERSONALIZED_MD_2023",
     "IMDRF N58 Edition 2 (2023) — Personalized Medical Devices: Regulatory Pathways",
     "https://www.imdrf.org/sites/default/files/2023-09/IMDRF_PMD%20WG_N58%20FINAL_2023%20(Edition%202)_0.pdf",
     "STANDARDS"),

    ("IMDRF_N74_PERSONALIZED_MD_2023",
     "IMDRF N74 (2023) — Personalized Medical Devices: Clinical Evidence",
     "https://www.imdrf.org/sites/default/files/2023-04/IMDRF%20Personalised%20Medical%20Devices%20WG%20N74%20FINAL%20%202023.pdf",
     "STANDARDS"),

    ("IMDRF_N85_CDS_2024",
     "IMDRF N85 (2024) — Clinical Decision Support Software: Guidance",
     "https://www.imdrf.org/sites/default/files/2024-10/IMDRF%20CDS%20Guidance%20Doc_260724.pdf",
     "STANDARDS"),
]

ALL_DOCS = INDIA_DOCS + IMDRF_DOCS


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
print(f"[lazy-load] Checking {len(ALL_DOCS)} docs against store...\n")

total_new = 0
skipped = 0

for doc_id, title, url, country in ALL_DOCS:
    if doc_id in existing:
        print(f"  SKIP {doc_id}")
        skipped += 1
        continue

    try:
        r = requests.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
    except Exception as e:
        print(f"  FAIL {doc_id}: {e}")
        continue

    ct = r.headers.get("content-type", "")
    if "pdf" in ct or r.content[:4] == b"%PDF":
        text = extract_pdf_text(r.content)
    else:
        print(f"  FAIL {doc_id}: not a PDF (content-type={ct})")
        continue

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
        device_classes=["Class A", "Class B", "Class C", "Class D",
                        "Class I", "Class IIa", "Class IIb", "Class III",
                        "SaMD", "IVD"],
    )
    n = embed_and_index_chunks(chunks, vector_store=store)
    total_new += n
    print(f"  +{n:3d}  {doc_id}  [{country}]", flush=True)

print(f"\nDone: {skipped} skipped, {total_new} new chunks added")
