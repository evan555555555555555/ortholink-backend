"""Wave 4: Singapore HSA, New Zealand Medsafe, FDA additional,
Saudi SFDA, India CDSCO new, Australia TGA, ICH Q10, Switzerland, Brazil RDC 751."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

DOCS = [
    # ── Singapore HSA ─────────────────────────────────────────────────────────
    ("SG_HSA_GN15_PRODUCT_REGISTRATION",
     "HSA Guidance GN-15: Medical Device Product Registration (Rev 12, 2025)",
     "https://www.hsa.gov.sg/docs/default-source/hprg-mdb/guidance-documents-for-medical-devices/gn-15-r12-guidance-on-medical-device-product-registration-(2025-aug)-pub.pdf",
     "SG"),

    ("SG_HSA_GN22_CLASS_A",
     "HSA Guidance GN-22: Guidance for Dealers on Class A Medical Devices (Rev 7.3)",
     "https://www.hsa.gov.sg/docs/default-source/hprg-mdb/guidance-documents-for-medical-devices/gn-22-r7-3-guidance-for-dealers-on-class-a-medical-devices(19-oct-pub).pdf",
     "SG"),

    ("SG_HSA_GN13_RISK_CLASSIFICATION",
     "HSA Guidance GN-13: Risk Classification of General Medical Devices (Rev 2.1)",
     "https://www.hsa.gov.sg/docs/default-source/medical-devices/gn-13-r2-1-guidance-on-the-risk-classification-of-general-medical-devices-(18sep-pub).pdf",
     "SG"),

    ("SG_HSA_GN01_OVERVIEW",
     "HSA Guidance GN-01: Overview of Medical Device Regulation in Singapore",
     "https://www.hsa.gov.sg/docs/default-source/hprg-mdb/guidance-documents-for-medical-devices/gn-01-r9-overview-of-medical-device-regulation-in-singapore.pdf",
     "SG"),

    # ── New Zealand Medsafe ───────────────────────────────────────────────────
    ("NZ_MEDSAFE_WAND_USER_GUIDE",
     "Medsafe New Zealand: WAND User Guide v3.1 — Web Assisted Notification of Devices",
     "https://www.medsafe.govt.nz/regulatory/devicesnew/NZWandUserGuide.pdf",
     "NZ"),

    ("NZ_MEDSAFE_DEVICE_OVERVIEW",
     "Medsafe New Zealand: Overview of Medical Device Regulation",
     "https://www.medsafe.govt.nz/regulatory/devicesnew/guidance.asp",
     "NZ"),

    # ── FDA Additional ────────────────────────────────────────────────────────
    ("US_FDA_510K_PROGRAM_2014",
     "FDA Guidance: The 510(k) Program — Evaluating Substantial Equivalence in Premarket Notifications (2014)",
     "https://www.fda.gov/media/82395/download",
     "US"),

    ("US_FDA_SOFTWARE_CONTENT_PREMARKET",
     "FDA Guidance: Content of Premarket Submissions for Device Software Functions (2023)",
     "https://www.fda.gov/media/153781/download",
     "US"),

    ("US_FDA_UDI_FORM_CONTENT",
     "FDA Guidance: Unique Device Identification — Form and Content of the UDI",
     "https://www.fda.gov/media/99084/download",
     "US"),

    ("US_FDA_UDI_DIRECT_MARKING",
     "FDA Guidance: Unique Device Identification — Direct Marking of Devices",
     "https://www.fda.gov/media/84269/download",
     "US"),

    ("US_FDA_COMBINATION_PRODUCTS_PATHWAYS",
     "FDA Guidance: Principles of Premarket Pathways for Combination Products (2019)",
     "https://www.fda.gov/media/119958/download",
     "US"),

    ("US_FDA_COMBINATION_POSTMARKET",
     "FDA Guidance: Postmarketing Safety Reporting for Combination Products (2019)",
     "https://www.fda.gov/media/111788/download",
     "US"),

    ("US_FDA_SAFETY_PERFORMANCE_PATHWAY",
     "FDA Guidance: Safety and Performance Based Pathway for Devices (2019)",
     "https://www.fda.gov/media/112691/download",
     "US"),

    ("US_FDA_ABBREVIATED_510K",
     "FDA Guidance: The Abbreviated 510(k) Program",
     "https://www.fda.gov/media/72646/download",
     "US"),

    ("US_FDA_HUMAN_FACTORS_2016",
     "FDA Guidance: Applying Human Factors and Usability Engineering to Medical Devices (2016)",
     "https://www.fda.gov/media/80481/download",
     "US"),

    ("US_FDA_CYBERSECURITY_2023",
     "FDA Guidance: Cybersecurity in Medical Devices — Quality System Considerations and Content of Premarket Submissions (2023)",
     "https://www.fda.gov/media/119933/download",
     "US"),

    # ── Saudi Arabia SFDA Additional ──────────────────────────────────────────
    ("SA_SFDA_MDS_G5",
     "SFDA MDS-G5: Guidance on Requirements for Medical Devices Registration",
     "https://www.sfda.gov.sa/sites/default/files/2020-07/MDS-G5.pdf",
     "SA"),

    ("SA_SFDA_MDS_G027_DIGITAL_HEALTH",
     "SFDA MDS-G027: Guidance on Digital Health Products (2025)",
     "https://www.sfda.gov.sa/sites/default/files/2025-08/MDS-G027.pdf",
     "SA"),

    ("SA_SFDA_MDS_REQ1",
     "SFDA MDS-REQ 1: Requirements for Medical Devices Marketing Authorization",
     "https://www.sfda.gov.sa/sites/default/files/2021-12/REQ1En_0.pdf",
     "SA"),

    # ── India CDSCO Additional 2024-2025 ─────────────────────────────────────
    ("IN_CDSCO_IVDMD_STABILITY_2024",
     "CDSCO Guidance on Stability Studies of In-Vitro Diagnostic Medical Devices (2024)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Final-stability-guidance-2024.pdf",
     "IN"),

    ("IN_CDSCO_MD_SOFTWARE_DRAFT_2025",
     "CDSCO Draft Guidance on Medical Device Software (October 2025)",
     "https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/UploadPublic_NoticesFiles/Draft%20guidance%20document%20on%20Medical%20Device%20Software%2021%2010%202025.pdf",
     "IN"),

    ("IN_CDSCO_FAQ_ADDENDUM_02_2025",
     "CDSCO FAQ on Medical Devices Rules 2017 — Addendum No. 02 (July 2025)",
     "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Addendum02faqmd.pdf",
     "IN"),

    # ── Australia TGA Additional ──────────────────────────────────────────────
    ("AU_TGA_SOFTWARE_REGULATORY_CHANGES_2024",
     "TGA Regulatory Changes for Software-Based Medical Devices (June 2024)",
     "https://www.tga.gov.au/sites/default/files/2024-07/regulatory-changes-software-based-medical-devices.pdf",
     "AU"),

    # ── ICH Guidelines ────────────────────────────────────────────────────────
    ("ICH_Q10_PHARMA_QUALITY_SYSTEM",
     "ICH Q10 — Pharmaceutical Quality System (Official Guideline)",
     "https://database.ich.org/sites/default/files/Q10%20Guideline.pdf",
     "STANDARDS"),

    # ── Switzerland Swissmedic Additional ────────────────────────────────────
    ("CH_SWISSMEDIC_SOFTWARE_GUIDANCE",
     "Swissmedic Information Sheet: Medical Device Software (BW630_30_007e)",
     "https://www.swissmedic.ch/dam/swissmedic/en/dokumente/medizinprodukte/mep_urr/bw630_30_007d_mbmedizinprodukte-software.pdf.download.pdf/BW630_30_007e_MB%20Medical%20Device%20Software.pdf",
     "CH"),

    ("CH_SWISSMEDIC_PROCUREMENT",
     "Swissmedic: Procurement of Medical Devices in Health Institutions (MU600_00_006e)",
     "https://www.swissmedic.ch/dam/swissmedic/en/dokumente/medizinprodukte/mep_urr/mu600_00_006d_mb_beschaffung_mep.pdf.download.pdf/MU600_00_006e_MB_Procurement_of_medical_devices_in_health_institutions.pdf",
     "CH"),

    # ── Brazil ANVISA ─────────────────────────────────────────────────────────
    ("BR_ANVISA_RDC_751_2022",
     "ANVISA RDC No. 751/2022 — Technical Requirements for Medical Devices Registration (English)",
     "https://latinigroup.com.br/images/legis/RESOLUTION_RDC_No_751.pdf",
     "BR"),

    # ── Japan PMDA Additional ─────────────────────────────────────────────────
    ("JP_PMDA_DEVICE_OVERVIEW",
     "PMDA Overview of Medical Device Regulatory System in Japan",
     "https://www.pmda.go.jp/files/000245839.pdf",
     "JP"),

    ("JP_PMDA_AI_ML_GUIDANCE_2023",
     "MHLW/PMDA AI/ML-Based Medical Device Software Discussion Paper (2023)",
     "https://www.pmda.go.jp/files/000265049.pdf",
     "JP"),

    # ── EU Additional ─────────────────────────────────────────────────────────
    ("EU_MDCG_2021_6_BORDERLINE",
     "MDCG 2021-6 — Questions and Answers on Transitional Provisions under MDR and IVDR",
     "https://health.ec.europa.eu/document/download/d33a09c3-5dbc-4f79-91af-b28b5abe0c4c_en?filename=mdcg_2021-6_en.pdf",
     "EU"),

    ("EU_MDCG_2022_14_VIGILANCE_FAQ",
     "MDCG 2022-14 — Guidance on the EUDAMED Actors Module — FAQ on Registration",
     "https://health.ec.europa.eu/document/download/29af54fc-f13b-4fde-869e-a3e9d3d62dba_en?filename=mdcg_2022-14_en.pdf",
     "EU"),

    ("EU_MDCG_2021_22_IMPLANT_CARD",
     "MDCG 2021-22 — Guidance on Implant Card (Article 18 MDR)",
     "https://health.ec.europa.eu/document/download/8dbdfe94-dfae-4804-8d23-23d1f3534d2a_en?filename=mdcg_2021-22_en.pdf",
     "EU"),

    ("EU_MDCG_2019_9_EUDAMED",
     "MDCG 2019-9 rev 1 — Summary of Safety and Clinical Performance (SSCP)",
     "https://health.ec.europa.eu/document/download/e7e73f35-bff4-498d-aade-96f3f4a62e17_en?filename=mdcg_2019-9-rev1_en.pdf",
     "EU"),
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
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        print(f"  FAIL {doc_id}: {e}")
        continue

    ct = r.headers.get("content-type", "")
    if not (r.content[:4] == b"%PDF" or "pdf" in ct):
        print(f"  FAIL {doc_id}: not a PDF (ct={ct[:40]})")
        continue

    text = extract_pdf_text(r.content)
    words = len(text.split())
    if words < 50:
        print(f"  SKIP {doc_id}: too short ({words} words)")
        continue

    device_classes = ["Class A", "Class B", "Class C", "Class D",
                      "Class I", "Class IIa", "Class IIb", "Class III", "Class IV",
                      "SaMD", "IVD", "AI/ML", "combination product"]

    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=title,
        document_id=doc_id,
        source_url=url,
        device_classes=device_classes,
    )
    n = embed_and_index_chunks(chunks, vector_store=store)
    total_new += n
    print(f"  +{n:3d}  {doc_id}  [{country}]", flush=True)

print(f"\nDone: {skipped} skipped, {total_new} new chunks added")
