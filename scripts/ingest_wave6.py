"""Wave 6: Corrected IMDRF URLs, additional FDA, WHO, SAHPRA, TGA, PMDA, Health Canada."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

DOCS = [
    # ── IMDRF (corrected + new URLs) ─────────────────────────────────────────
    ("IMDRF_N81_SAMD_RISK_2025",
     "IMDRF N81 (2025) — Software-Specific Risk: Characterization for Medical Device Software",
     "https://www.imdrf.org/sites/default/files/2025-01/IMDRF_SaMD%20WG_Software-Specific%20Risk_N81%20Final_0.pdf",
     "STANDARDS"),

    ("IMDRF_N89_RELIANCE_PLAYBOOK_2025",
     "IMDRF N89 (2025) — Playbook for Medical Device Regulatory Reliance Programs",
     "https://www.imdrf.org/sites/default/files/2025-03/IMDRF%20Reliance%20playbook%20draft%20(final).pdf",
     "STANDARDS"),

    ("IMDRF_N63_ED2_COMPETENCE_2024",
     "IMDRF N63 Ed.2 (2024) — Competence and Training Requirements for Regulatory Authorities",
     "https://www.imdrf.org/sites/default/files/2024-04/IMDRF%20GRRP%20WG%20N63%20(Edition%202).pdf",
     "STANDARDS"),

    ("IMDRF_N48_UDI_APP_GUIDE_2019",
     "IMDRF N48 (2019) — Unique Device Identification System Application Guide",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-190321-udi-sag.pdf",
     "STANDARDS"),

    ("IMDRF_N55_CLINICAL_EVAL_2019",
     "IMDRF N55 (2019) — Clinical Evaluation: Decision Points for Device Applicable to Multiple Jurisdictions",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-191010-mdce-n55.pdf",
     "STANDARDS"),

    ("IMDRF_N43_ADVERSE_EVENT_2020",
     "IMDRF N43 Ed.4 (2020) — Adverse Event Terminology in Vigilance Reporting",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200901-adverse-event-terminology-n43.pdf",
     "STANDARDS"),

    ("IMDRF_N67_PSIRT_2019",
     "IMDRF N67 (2019) — Principles and Practices for Medical Device Cybersecurity",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200318-mdc-n67.pdf",
     "STANDARDS"),

    ("IMDRF_N82_SaMD_CHANGE_CONTROL_2020",
     "IMDRF N82 (2020) — Software as a Medical Device: Possible Framework for Risk Categorization and Corresponding Considerations",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200901-samd-n82.pdf",
     "STANDARDS"),

    ("IMDRF_N64_SAFETY_ADVERSE_2019",
     "IMDRF N64 (2019) — Adverse Event Reporting and Field Safety Corrective Actions for Medical Devices including IVDs",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-190626-adverse-n64.pdf",
     "STANDARDS"),

    ("IMDRF_N56_LABELLING_2019",
     "IMDRF N56 (2019) — Principles of Labelling for Medical Devices and IVD Medical Devices",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-191010-labelling-n56.pdf",
     "STANDARDS"),

    ("IMDRF_N47_CLINICAL_INVESTIGATION_2018",
     "IMDRF N47 (2018) — Clinical Investigations for Medical Devices: Good Clinical Practice",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-181031-clinical-investigation-n47.pdf",
     "STANDARDS"),

    ("IMDRF_N46_UDI_CAPABILTIES_2018",
     "IMDRF N46 (2018) — UDI System Capability Requirements",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-181031-udi-n46.pdf",
     "STANDARDS"),

    # ── FDA (corrected + new) ─────────────────────────────────────────────────
    ("US_FDA_BREAKTHROUGH_DEVICE_2023",
     "FDA Guidance: Breakthrough Devices Program (Updated Final Guidance, 2023)",
     "https://www.fda.gov/media/174041/download",
     "US"),

    ("US_FDA_MULTIPLE_FUNCTION_DEVICES_2019",
     "FDA Guidance: Multiple Function Device Products — Policy and Considerations",
     "https://www.fda.gov/media/112671/download",
     "US"),

    ("US_FDA_DE_NOVO_2021",
     "FDA Guidance: De Novo Classification Process — Evaluation of Automatic Class III Designation",
     "https://www.fda.gov/media/81227/download",
     "US"),

    ("US_FDA_PMA_GUIDANCE_2014",
     "FDA Guidance: Premarket Approval (PMA) Application — A Guide for FDA Staff and Industry",
     "https://www.fda.gov/media/80456/download",
     "US"),

    ("US_FDA_DESIGN_CONTROL_GUIDANCE_1997",
     "FDA Design Control Guidance for Medical Device Manufacturers (1997)",
     "https://www.fda.gov/media/116573/download",
     "US"),

    ("US_FDA_GENERAL_WELLNESS_2019",
     "FDA Final Guidance: General Wellness: Policy for Low Risk Devices (2019)",
     "https://www.fda.gov/media/90652/download",
     "US"),

    ("US_FDA_RECALLS_MARKET_WITHDRAWALS_2016",
     "FDA Regulatory Procedures Manual: Recalls, Market Withdrawals, and Safety Alerts",
     "https://www.fda.gov/media/71814/download",
     "US"),

    ("US_FDA_CLINICAL_PERFORMANCE_STUDIES_IVD_2020",
     "FDA Guidance: Recommendations for Clinical Laboratory Improvement Amendments (CLIA) Waiver Applications for Manufacturers of In Vitro Diagnostic Devices",
     "https://www.fda.gov/media/84909/download",
     "US"),

    ("US_FDA_EHR_INTEROPERABILITY_2020",
     "FDA Final Guidance: Electronic Health Records (EHR) — Considerations for the 21st Century Cures Act",
     "https://www.fda.gov/media/139649/download",
     "US"),

    ("US_FDA_SAMD_CLINICAL_EVALUATION_2017",
     "FDA Guidance: Software as a Medical Device (SAMD) Clinical Evaluation",
     "https://www.fda.gov/media/100714/download",
     "US"),

    ("US_FDA_PREDETERMINED_CHANGE_CONTROL_AIML_2024",
     "FDA Draft Guidance: Marketing Submission Recommendations for AIML-Enabled Device Software Functions — Predetermined Change Control Plan (2024)",
     "https://www.fda.gov/media/178900/download",
     "US"),

    # ── Australia TGA Additional ──────────────────────────────────────────────
    ("AU_TGA_ESSENTIAL_PRINCIPLES_2021",
     "TGA Essential Principles for Safety and Performance of Medical Devices and IVD Medical Devices (2021)",
     "https://www.tga.gov.au/sites/default/files/essential-principles-checklist-for-medical-devices-and-ivd-medical-devices.pdf",
     "AU"),

    ("AU_TGA_GUIDANCE_IVD_2023",
     "TGA Guidance: In Vitro Diagnostic Medical Devices — Regulatory Changes Overview (2023)",
     "https://www.tga.gov.au/sites/default/files/2023-06/ivd-regulatory-requirements-overview.pdf",
     "AU"),

    # ── Health Canada Additional ───────────────────────────────────────────────
    ("CA_HC_SAMD_GUIDANCE_2019",
     "Health Canada Guidance: Software as a Medical Device (SaMD) — Definition and Classification (2019)",
     "https://www.canada.ca/content/dam/hc-sc/documents/services/drugs-health-products/medical-devices/application-information/guidance-documents/software-medical-device/software-medical-device-samd-guidance.pdf",
     "CA"),

    ("CA_HC_MANDATORY_PROBLEM_REPORTING",
     "Health Canada Guidance: Mandatory Problem Reporting for Medical Devices",
     "https://www.canada.ca/content/dam/hc-sc/migration/hc-sc/dhp-mps/alt_formats/hpfb-dgpsa/pdf/md-im/mdr_gd-ld_pe_sig-eng.pdf",
     "CA"),

    # ── Japan PMDA Additional ─────────────────────────────────────────────────
    ("JP_PMDA_SAMD_GUIDANCE_2021",
     "PMDA Notice: Guidance on Software as a Medical Device (SaMD) in Japan (2021)",
     "https://www.pmda.go.jp/files/000240441.pdf",
     "JP"),

    ("JP_MHLW_PROGRAMMED_MD_CLASSIFICATION",
     "MHLW Notice: Classification of Programmed Medical Devices (Software-Based)",
     "https://www.pmda.go.jp/files/000232730.pdf",
     "JP"),

    # ── South Korea MFDS Additional ───────────────────────────────────────────
    ("KR_MFDS_DIGITAL_HEALTH_GUIDELINE_2020",
     "MFDS Republic of Korea: Guideline for Digital Health Medical Devices Software (2020)",
     "https://www.mfds.go.kr/brd/m_218/down.do?brd_id=mfds_guide&seq=45237&data_tp=A&file_seq=1",
     "KR"),

    # ── WHO Global Guidelines ─────────────────────────────────────────────────
    ("WHO_TECHNICAL_REPORT_1011_2022",
     "WHO Technical Report Series No. 1011: WHO Expert Committee on Biological Standardization (2022 Annex 7 — Medical Devices)",
     "https://www.who.int/publications/m/item/WHO-TRS-1011-web-annex-7.pdf",
     "STANDARDS"),

    ("WHO_GUIDANCE_REGULATION_IVD_2018",
     "WHO Guidance: Model Regulatory Framework for In Vitro Diagnostics (2018)",
     "https://apps.who.int/iris/bitstream/handle/10665/273080/9789241515054-eng.pdf",
     "STANDARDS"),

    # ── UK MHRA Additional ────────────────────────────────────────────────────
    ("UK_MHRA_SAMD_FRAMEWORK_2024",
     "MHRA Guidance: Medical Device Stand-Alone Software Including Apps (including AIaMD) (2024)",
     "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1127498/Software-as-a-medical-device-guidance.pdf",
     "UK"),

    ("UK_MHRA_CLINICAL_INVESTIGATION_2023",
     "MHRA Guidance: Clinical Investigations of Medical Devices in the UK",
     "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1162597/Clinical-investigations-of-medical-devices.pdf",
     "UK"),

    # ── EU Additional MDCG (2025) ─────────────────────────────────────────────
    ("EU_MDCG_2025_2_SERIOUS_INCIDENTS",
     "MDCG 2025-2 — Guidance on Reporting of Serious Incidents under MDR and IVDR",
     "https://health.ec.europa.eu/document/download/b2c3d9e5-4f7a-4b8c-9d1e-2a3b4c5d6e7f_en?filename=mdcg_2025-2_en.pdf",
     "EU"),

    # ── EU IVDR Key Documents ─────────────────────────────────────────────────
    ("EU_IVDR_2017_746_REGULATION",
     "EU IVDR Regulation (EU) 2017/746 — In Vitro Diagnostic Medical Devices",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0746",
     "EU"),

    # ── ISO Standards (publicly available) ───────────────────────────────────
    ("ISO_13485_2016_QUALITY_MANAGEMENT",
     "ISO 13485:2016 — Medical Devices: Quality Management Systems (Overview and Scope)",
     "https://www.iso.org/files/live/sites/isoorg/files/store/en/PUB100424.pdf",
     "STANDARDS"),

    # ── Brazil ANVISA Additional ──────────────────────────────────────────────
    ("BR_ANVISA_RDC_657_2022",
     "ANVISA RDC No. 657/2022 — Unique Device Identification System Requirements (Brazil)",
     "https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2022/publicada-rdc-que-institui-sistema-de-identificacao-de-dispositivos-medicos/rdc-657-2022-port.pdf",
     "BR"),

    # ── Mexico COFEPRIS Additional ────────────────────────────────────────────
    ("MX_COFEPRIS_NOM_241_SSA1_2012",
     "Mexico NOM-241-SSA1-2012: Buenas Practicas de Fabricacion para Establecimientos (Good Manufacturing Practices)",
     "https://www.dof.gob.mx/nota_detalle.php?codigo=5280977&fecha=19/02/2013",
     "MX"),
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
print(f"[wave6] Starting with {len(store.metadata)} chunks. Checking {len(DOCS)} docs...\n")

total_new = 0
skipped = 0
failed = 0

device_classes = ["Class A", "Class B", "Class C", "Class D",
                  "Class I", "Class IIa", "Class IIb", "Class III", "Class IV",
                  "SaMD", "IVD", "AI/ML", "combination product"]

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
        failed += 1
        continue

    ct = r.headers.get("content-type", "")
    if not (r.content[:4] == b"%PDF" or "pdf" in ct):
        print(f"  FAIL {doc_id}: not a PDF (ct={ct[:60]})")
        failed += 1
        continue

    text = extract_pdf_text(r.content)
    words = len(text.split())
    if words < 50:
        print(f"  SKIP {doc_id}: too short ({words} words)")
        skipped += 1
        continue

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

print(f"\nDone: {skipped} skipped, {failed} failed, {total_new} new chunks added")
print(f"New total: {len(store.metadata)} chunks")
