"""Ingest MDCG guidance documents (EU Medical Device Coordination Group)."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

HEADERS = {"User-Agent": "Mozilla/5.0"}

# All substantive MDCG PDF guidance documents
DOCS = [
    # --- Software / Digital Health / AI ---
    ("EU_MDCG_2025_6",  "MDCG 2025-6 — FAQ on MDR/IVDR and Artificial Intelligence Act",
     "https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en?filename=mdcg_2025-6_en.pdf"),
    ("EU_MDCG_2025_4",  "MDCG 2025-4 — Medical Device Software Apps on Online Platforms",
     "https://health.ec.europa.eu/document/download/ec9b0f40-7f82-43a7-b833-ebd45b772eae_en?filename=mdcg_2025-4_en.pdf"),
    ("EU_MDCG_2023_4",  "MDCG 2023-4 — Guidance on MDSW-Hardware Combinations",
     "https://health.ec.europa.eu/document/download/b2c4e715-f2b4-4d24-af60-056b5d41a72e_en?filename=md_mdcg_2023-4_software_en.pdf"),
    ("EU_MDCG_2020_1",  "MDCG 2020-1 — Clinical Evaluation and Performance Evaluation of Medical Device Software",
     "https://health.ec.europa.eu/document/download/19d9e24f-2808-4e00-bfeb-75892047407d_en?filename=md_mdcg_2020_1_guidance_clinic_eva_md_software_en.pdf"),
    ("EU_MDCG_2019_16", "MDCG 2019-16 rev.1 — Cybersecurity for Medical Devices",
     "https://health.ec.europa.eu/document/download/b23b362f-8a56-434c-922a-5b3ca4d0a7a1_en?filename=md_cybersecurity_en.pdf"),
    ("EU_MDCG_2019_11", "MDCG 2019-11 — Qualification and Classification of Software in Regulation (EU) 2017/745 and 2017/746",
     "https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en?filename=mdcg_2019_11_en.pdf"),
    # --- Clinical Evaluation ---
    ("EU_MDCG_2025_10", "MDCG 2025-10 — Guidance on Post-Market Surveillance",
     "https://health.ec.europa.eu/document/download/a9ad86b7-1b8e-4bae-beb4-48b2b3ed2f05_en?filename=mdcg_2025-10_en.pdf"),
    ("EU_MDCG_2024_10", "MDCG 2024-10 — Clinical Evaluation of Orphan Medical Devices",
     "https://health.ec.europa.eu/document/download/daa1fc59-9d2c-4e82-878e-d6fdf12ecd1a_en?filename=mdcg_2024-10_en.pdf"),
    ("EU_MDCG_2023_7",  "MDCG 2023-7 — Exemptions from Clinical Investigation Requirements under MDR Article 62(4)",
     "https://health.ec.europa.eu/document/download/1b5f9cc0-cea0-4459-921f-eaf4b4f80983_en?filename=mdcg_2023-7_en.pdf"),
    ("EU_MDCG_2021_28", "MDCG 2021-28 — Substantial Modification of a Clinical Investigation",
     "https://health.ec.europa.eu/document/download/ba8069a1-6881-4360-b52b-6cab048aee43_en?filename=mdcg_2021-28_en.pdf"),
    ("EU_MDCG_2021_8",  "MDCG 2021-8 — Clinical Investigation Application and Notification Documents",
     "https://health.ec.europa.eu/document/download/13265ec7-1776-41af-afb6-e0a64bc407b5_en?filename=mdcg_2021-8_en.pdf"),
    ("EU_MDCG_2021_6",  "MDCG 2021-6 rev.1 — Q&A Regarding Clinical Investigation under MDR",
     "https://health.ec.europa.eu/document/download/f124f630-389e-4c45-90dc-24ec0a707838_en?filename=mdcg_2021-6_en.pdf"),
    ("EU_MDCG_2020_10", "MDCG 2020-10/1 rev.1 — Guidance on Safety Reporting in Clinical Investigations",
     "https://health.ec.europa.eu/document/download/0537d335-7eed-4087-b65d-3c2bd8c72c1a_en?filename=md_mdcg_2020-10-1_guidance_safety_reporting_en.pdf"),
    ("EU_MDCG_2020_8",  "MDCG 2020-8 — PMCF Evaluation Report Template",
     "https://health.ec.europa.eu/document/download/11121036-696a-4589-a311-c5525bd84df3_en?filename=md_mdcg_2020_8_guidance_pmcf_evaluation_report_en.pdf"),
    ("EU_MDCG_2020_7",  "MDCG 2020-7 — Post-Market Clinical Follow-Up Plan Template",
     "https://health.ec.europa.eu/document/download/a5cdb303-c782-4010-8723-7d389af678f7_en?filename=md_mdcg_2020_7_guidance_pmcf_plan_template_en.pdf"),
    ("EU_MDCG_2020_6",  "MDCG 2020-6 — Sufficient Clinical Evidence for Legacy Devices",
     "https://health.ec.europa.eu/document/download/a6d29444-b5d5-4afb-8024-10be85256aa7_en?filename=md_mdcg_2020_6_guidance_sufficient_clinical_evidence_en.pdf"),
    ("EU_MDCG_2020_5",  "MDCG 2020-5 — Clinical Evaluation — Equivalence",
     "https://health.ec.europa.eu/document/download/575a0f79-e3a0-4a96-9ce0-930576c12aa2_en?filename=md_mdcg_2020_5_guidance_clinical_evaluation_equivalence_en.pdf"),
    ("EU_MDCG_2019_9",  "MDCG 2019-9 rev.1 — Summary of Safety and Clinical Performance",
     "https://health.ec.europa.eu/document/download/5f082b2f-8d51-495c-9ab9-985a9f39ece4_en?filename=md_mdcg_2019_9_sscp_en.pdf"),
    # --- Classification and Borderline ---
    ("EU_MDCG_2021_24", "MDCG 2021-24 — Guidance on Classification of Medical Devices",
     "https://health.ec.europa.eu/document/download/cbb19821-a517-4e13-bf87-fdc6ddd1782e_en?filename=mdcg_2021-24_en.pdf"),
    ("EU_MDCG_2022_5",  "MDCG 2022-5 rev.1 — Borderline Between Medical Devices and Medicinal Products",
     "https://health.ec.europa.eu/document/download/b5a27717-229f-4d7a-97b1-e1c7d819e579_en?filename=mdcg_2022-5_en.pdf"),
    ("EU_MDCG_BORDERLINE", "Manual on Borderline and Classification under MDR and IVDR v4",
     "https://health.ec.europa.eu/document/download/71a87df8-5ca1-4555-b453-b65bdf8de909_en?filename=md_borderline_manual_en.pdf"),
    # --- PMS / Vigilance ---
    ("EU_MDCG_2022_21", "MDCG 2022-21 — Guidance on Periodic Safety Update Report (PSUR)",
     "https://health.ec.europa.eu/document/download/a7df24c3-d4a3-4218-a8e0-726febfa01c2_en?filename=mdcg_2022-21_en.pdf"),
    ("EU_MDCG_2023_3",  "MDCG 2023-3 rev.2 — Q&A on Vigilance Terms and Concepts",
     "https://health.ec.europa.eu/document/download/af1433fd-ed64-4c53-abc7-612a7f16f976_en?filename=mdcg_2023-3_en.pdf"),
    # --- PRRC / Compliance ---
    ("EU_MDCG_2019_7",  "MDCG 2019-7 rev.1 — Guidance on Article 15 PRRC Requirements",
     "https://health.ec.europa.eu/document/download/463b4f08-44a2-4018-9957-488bf386fc3a_en?filename=md_mdcg_2019_7_guidance_art15_mdr_ivdr_en.pdf"),
    ("EU_MDCG_2019_15", "MDCG 2019-15 rev.1 — Guidance Notes for Manufacturers of Class I Devices",
     "https://health.ec.europa.eu/document/download/349a2d4c-ea2a-4279-861c-7c063bc077e4_en?filename=md_guidance-manufacturers_en.pdf"),
    ("EU_MDCG_2021_27", "MDCG 2021-27 rev.1 — Q&A on Articles 13 and 14 (Importers and Distributors)",
     "https://health.ec.europa.eu/document/download/82d9adbc-dbf0-40d4-93ed-ade673c8232a_en?filename=mdcg_2021-27_en.pdf"),
    ("EU_MDCG_2022_16", "MDCG 2022-16 — Guidance on Authorised Representatives",
     "https://health.ec.europa.eu/document/download/0a7613cb-6b9a-4396-a4c6-d2479e43e167_en?filename=mdcg_202216_en.pdf"),
    ("EU_MDCG_2021_26", "MDCG 2021-26 — Q&A on Repackaging and Relabelling Activities",
     "https://health.ec.europa.eu/document/download/4c7bd740-d7c3-4f7b-9a4d-249121a67b8a_en?filename=md_mdcg_2021_26_en.pdf"),
    # --- UDI ---
    ("EU_MDCG_2022_7",  "MDCG 2022-7 — Q&A on the Unique Device Identifier System",
     "https://health.ec.europa.eu/document/download/b5429d14-25a9-4cfc-b059-355388f03e05_en?filename=mdcg_2022-7_en.pdf"),
    ("EU_MDCG_2021_19", "MDCG 2021-19 — Integration of the UDI in a QMS",
     "https://health.ec.europa.eu/document/download/64443e95-e1be-4ffd-80f7-9c6a1b0b99b6_en?filename=md_2021-19_en.pdf"),
    ("EU_MDCG_2018_1",  "MDCG 2018-1 rev.4 — Basic UDI-DI and Changes",
     "https://health.ec.europa.eu/document/download/cb1bf6e5-3972-4b3a-82d9-c5946738b2a5_en?filename=md_mdcg_2018-1_guidance_udi-di_en.pdf"),
    # --- IVD ---
    ("EU_MDCG_2022_2",  "MDCG 2022-2 — Guidance on General Principles of Clinical Evidence for IVD Medical Devices",
     "https://health.ec.europa.eu/document/download/f373538f-939c-472f-9536-436b6ddac085_en?filename=mdcg_2022-2_en.pdf"),
    ("EU_MDCG_2022_8",  "MDCG 2022-8 — Application of IVDR to Legacy Devices",
     "https://health.ec.europa.eu/document/download/76f9983e-164c-45f1-b2b9-c9e5050cefe9_en?filename=mdcg_2022-8_en.pdf"),
    ("EU_MDCG_2024_11", "MDCG 2024-11 — Qualification of IVD Medical Devices",
     "https://health.ec.europa.eu/document/download/12b92152-371f-404d-a865-93800cd5cdca_en?filename=mdcg_2024-11_en.pdf"),
    ("EU_MDCG_2020_16", "MDCG 2020-16 rev.4 — Classification Rules for IVD Medical Devices",
     "https://health.ec.europa.eu/document/download/12f9756a-1e0d-4aed-9783-d948553f1705_en?filename=md_mdcg_2020_guidance_classification_ivd-md_en.pdf"),
    # --- Notified Bodies ---
    ("EU_MDCG_2019_6",  "MDCG 2019-6 rev.5 — Q&A on Notified Bodies Requirements under MDR/IVDR",
     "https://health.ec.europa.eu/document/download/9c9c532f-013a-477c-9378-0a9e714e5549_en?filename=md_mdcg_qa_requirements_notified_bodies_en.pdf"),
    ("EU_MDCG_2022_13", "MDCG 2022-13 rev.1 — Designation and Notification of Conformity Assessment Bodies",
     "https://health.ec.europa.eu/document/download/27f91dc2-b5bc-44f9-a975-5024ce3ea556_en?filename=mdcg_2022-13_en.pdf"),
    ("EU_MDCG_2024_12", "MDCG 2024-12 — CAPA Plan Assessment Guidance for Notified Bodies",
     "https://health.ec.europa.eu/document/download/080c6aed-4f09-4a8a-b052-3414275945db_en?filename=mdcg_2024-12_en.pdf"),
    ("EU_MDCG_2021_5",  "MDCG 2021-5 rev.1 — Standardisation in the Field of Medical Devices",
     "https://health.ec.europa.eu/document/download/59ac4cb0-f187-4ca2-814d-82c42cde5408_en?filename=md_mdcg_2021_5_en.pdf"),
    ("EU_MDCG_2020_14", "MDCG 2020-14 — MDSAP Audit Reports in a Surveillance Context",
     "https://health.ec.europa.eu/document/download/44dc96aa-e517-4af1-855b-f7fcb4b699c9_en?filename=md_2020-14-guidance-mdsap_en.pdf"),
    # --- Transitional Provisions ---
    ("EU_MDCG_2021_25", "MDCG 2021-25 rev.1 — Application of MDR to Legacy Devices",
     "https://health.ec.europa.eu/document/download/cbb11a6e-f0f3-4e30-af5e-990f9ef68bc1_en?filename=md_mdcg_2021_25_en.pdf"),
    ("EU_MDCG_2022_18", "MDCG 2022-18 — Article 97 Legacy Devices Position Paper",
     "https://health.ec.europa.eu/document/download/ccfb20a8-82d4-4216-8d11-4ce127be431e_en?filename=mdcg_2022-18_en_1.pdf"),
    ("EU_MDCG_2020_3",  "MDCG 2020-3 rev.1 — Significant Changes under Article 120",
     "https://health.ec.europa.eu/document/download/800e8e87-d4eb-4cc5-b5ad-07a9146d7c90_en?filename=mdcg_2020-3_en_1.pdf"),
    # --- Annex XVI ---
    ("EU_MDCG_2023_6",  "MDCG 2023-6 — Guidance on Demonstration of Equivalence for Annex XVI Products",
     "https://health.ec.europa.eu/document/download/15a33521-87f1-4939-92a1-ef23f2b09c6c_en?filename=mdcg_2023-6_en.pdf"),
    ("EU_MDCG_2023_5",  "MDCG 2023-5 — Guidance on Qualification and Classification of Annex XVI Products",
     "https://health.ec.europa.eu/document/download/ea4acf26-979a-4dbb-92ff-8d1d804da51a_en?filename=mdcg_2023-5_en.pdf"),
    # --- In-house / Article 5(5) ---
    ("EU_MDCG_2023_1",  "MDCG 2023-1 — Health Institution Exemption under Article 5(5) MDR/IVDR",
     "https://health.ec.europa.eu/document/download/05b15d55-1bcf-4e17-99c4-15c706325847_en?filename=mdcg_2023-1_en.pdf"),
    # --- Custom-Made ---
    ("EU_MDCG_2021_3",  "MDCG 2021-3 — Q&A on Custom-Made Devices",
     "https://health.ec.europa.eu/document/download/385d7e20-d8b5-49d0-abd7-8daf269bf1b8_en?filename=mdcg_2021-3_en.pdf"),
    # --- Breakthrough Devices ---
    ("EU_MDCG_2025_9",  "MDCG 2025-9 — Guidance on Breakthrough Devices",
     "https://health.ec.europa.eu/document/download/edca94c7-62ab-4dd5-8539-2b347bd14809_en?filename=mdcg_2025-9.pdf"),
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
print(f"Vector store: {len(store.metadata)} chunks, {len(existing)} unique docs\n")

total = 0
skipped = 0

for doc_id, title, url in DOCS:
    if doc_id in existing:
        print(f"  SKIP {doc_id} (already indexed)")
        skipped += 1
        continue

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  FAIL {doc_id}: {e}")
        continue

    text = extract_pdf_text(r.content)
    words = len(text.split())
    if words < 50:
        print(f"  SKIP {doc_id}: too short ({words} words)")
        continue

    chunks = chunk_regulatory_text(
        text=text,
        country="EU",
        regulation_name=title,
        document_id=doc_id,
        source_url=url,
        device_classes=["Class I", "Class IIa", "Class IIb", "Class III", "SaMD", "IVD"],
    )
    n = embed_and_index_chunks(chunks, vector_store=store)
    total += n
    print(f"  +{n:3d}  {doc_id}", flush=True)

print(f"\nDone: {skipped} skipped, {total} new chunks added")
