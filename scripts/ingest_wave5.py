"""Wave 5: Remaining MDCG docs (2018–2025 gaps) + IMDRF new + PMDA JP + Korea MFDS."""
import sys, io, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunker import chunk_regulatory_text
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import get_vector_store

BASE = "https://health.ec.europa.eu"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def mdcg(num_str, uuid, filename_hint=None):
    """Build a MDCG doc tuple."""
    fn = filename_hint or f"mdcg_{num_str.replace('-', '_')}_en.pdf"
    url = f"{BASE}/document/download/{uuid}_en?filename={fn}"
    doc_id = "EU_MDCG_" + num_str.replace("-", "_").replace(".", "_").upper()
    return (doc_id, f"MDCG {num_str} EU Medical Device Guidance", url, "EU")


DOCS = [
    # ── MDCG 2025 — new/missing ───────────────────────────────────────────────
    mdcg("2025-5",  "f22f559b-dee5-43b4-9595-3ccdcca9f7ad", "mdcg_2025-5_en.pdf"),
    mdcg("2025-8",  "f026ed58-060d-49b1-b660-914838670a20", "mdcg_2025-8_en.pdf"),
    mdcg("2025-7",  "ad6ae143-baa2-451a-8c5e-0c5d22983e88", "mdcg_2025-7_en.pdf"),
    mdcg("2025-3",  "4467c367-be73-4673-aa68-b532222d6bf2", "mdcg_2025-3_en.pdf"),
    mdcg("2025-2",  "ff8d6bf6-785f-48e1-beaf-2cfe13f59fd7", "mdcg_2025-2_en.pdf"),
    mdcg("2025-1",  "e2cb674e-73cb-4a2b-b0d3-76c70505b632", "mdcg_2025-1_en.pdf"),

    # ── MDCG 2024 — new/missing ───────────────────────────────────────────────
    mdcg("2024-16", "919061d9-5dfa-4d0b-ab9b-3543eed98f76", "mdcg_2024-16_en.pdf"),
    mdcg("2024-15", "0e076d19-62dc-4ff9-83f2-be6072a45993", "mdcg_2024-15_en.pdf"),
    mdcg("2024-14", "c8c6cca5-460e-410e-a325-be08bfc7dea6", "mdcg_2024-14_en.pdf"),
    mdcg("2024-13", "3fa74c6b-953a-41f5-b024-8889ac8b5ddf", "mdcg_2024-13_en.pdf"),
    mdcg("2024-9",  "f067a42e-4af6-47dd-980f-b251eeeafaf0", "mdcg_2024-9_en.pdf"),
    mdcg("2024-8",  "9819fd07-3281-477b-a8c6-22c0c47c47a6", "mdcg_2024-8_en.pdf"),
    mdcg("2024-7",  "939cc0b8-8db5-4b0d-b2b7-a15a98a44cbd", "mdcg_2024-7_en.pdf"),
    mdcg("2024-6",  "88772d4f-9e29-4b1b-aa41-adcc74b6bb41", "mdcg_2024-6_en.pdf"),
    mdcg("2024-5",  "ee7ee8eb-841a-4085-a8dc-af6d55ebf1bd", "mdcg_2024-5_en.pdf"),
    mdcg("2024-4",  "5cc894e0-331d-4fa2-8ab3-cdd4437c48fc", "mdcg_2024-4_en.pdf"),
    mdcg("2024-3",  "690de85a-ac17-45ea-bb32-7839540c25c4", "mdcg_2024-3_en.pdf"),
    mdcg("2024-2",  "de470384-e8be-45e7-a334-226757f8816d", "mdcg_2024-2_en.pdf"),
    # PMSV 2024-1 series (six companion docs)
    mdcg("2024-1",   "dbd0d748-d646-4274-afaa-399952809389", "mdcg_2024-1_en.pdf"),
    mdcg("2024-1-1", "e170bdf5-1684-4e24-bfbc-ec34b1ea1b4f", "mdcg_2024-1-1_en.pdf"),
    mdcg("2024-1-2", "6bfe418c-b72e-4e8a-b7e3-7aa1d0060da8", "mdcg_2024-1-2_en.pdf"),
    mdcg("2024-1-3", "c1e34b32-4faf-4ea5-af66-959642148b3a", "mdcg_2024-1-3_en.pdf"),
    mdcg("2024-1-4", "07c958e3-c2c7-4a22-82f1-86ab2debf40e", "mdcg_2024-1-4_en.pdf"),
    mdcg("2024-1-5", "c8441ddc-c586-4dbf-afc8-6ec3250df54b", "mdcg_2024-1-5_en.pdf"),

    # ── MDCG 2023 ─────────────────────────────────────────────────────────────
    mdcg("2023-2",  "309b49d8-07dd-40bf-b3a1-649dbdc378af", "mdcg_2023-2_en.pdf"),

    # ── MDCG 2022 — new/missing ───────────────────────────────────────────────
    mdcg("2022-20", "6a354f46-8308-4d60-a36d-2ef1ba56e41a", "mdcg_2022-20_en.pdf"),
    mdcg("2022-19", "4e1f946d-a71a-42c7-bd98-0e9977752669", "mdcg_2022-19_en.pdf"),
    mdcg("2022-17", "c2b875dd-06dd-47b6-8822-afe43f630655", "mdcg_2022-17_en.pdf"),
    mdcg("2022-15", "9518a759-24ce-4e15-83a5-e7b383911000", "mdcg_2022-15_en.pdf"),
    mdcg("2022-14", "2db053bc-283c-4d2e-93f4-c3e8032e66da", "mdcg_2022-14_en.pdf"),
    mdcg("2022-12", "c9008091-8ad7-4449-af75-f4f5a6abc761", "mdcg_2022-12_en.pdf"),
    mdcg("2022-11", "5ec4d600-d344-4232-9371-1d278b2abc12", "mdcg_2022-11_en.pdf"),
    mdcg("2022-10", "59abcc81-fd32-4546-a340-24c8fad4e2ac", "mdcg_2022-10_en.pdf"),
    mdcg("2022-9",  "b7cf356f-733f-4dce-9800-0933ff73622a", "mdcg_2022-9_en.pdf"),
    mdcg("2022-6",  "14c2d8dd-8489-4db5-b035-1c174f17fb54", "mdcg_2022-6_en.pdf"),
    mdcg("2022-4",  "e5714b2b-e98b-4fce-b5ff-d9141a8f30e1", "mdcg_2022-4_en.pdf"),
    mdcg("2022-3",  "ebbc4f6a-4945-4d5d-9c22-9bc1aafc5532", "mdcg_2022-3_en.pdf"),
    mdcg("2022-1",  "cd617093-f2bd-4a99-9058-9805ce4d0db3", "mdcg_2022-1_en.pdf"),

    # ── MDCG 2021 — new/missing ───────────────────────────────────────────────
    mdcg("2021-23", "e985ea01-e6b7-4900-af3a-58e5b01678aa", "mdcg_2021-23_en.pdf"),
    mdcg("2021-22", "98db0ec5-306f-4a0d-8bc9-5724f5d48942", "mdcg_2021-22_en.pdf"),
    mdcg("2021-21", "729f09dc-9f95-40b9-a62a-a0e9fff1d252", "mdcg_2021-21_en.pdf"),
    mdcg("2021-20", "54d417c4-df69-416f-a2bd-7a5de8cba611", "mdcg_2021-20_en.pdf"),
    mdcg("2021-18", "79150afd-7758-4023-978b-ec726f760482", "mdcg_2021-18_en.pdf"),
    mdcg("2021-17", "d0fe229f-c9e3-4f75-bf3b-bc4381206eef", "mdcg_2021-17_en.pdf"),
    mdcg("2021-16", "2b39d49f-4158-4470-b051-268cea9f21e1", "mdcg_2021-16_en.pdf"),
    mdcg("2021-15", "f1202a34-d27b-40f4-a777-975fc78dab47", "mdcg_2021-15_en.pdf"),
    mdcg("2021-14", "94b7e87c-ce5f-49b9-82a4-9eec6b0591bb", "mdcg_2021-14_en.pdf"),
    mdcg("2021-13", "8e9adf30-71b5-4faf-a8fb-b35ce4804440", "mdcg_2021-13_en.pdf"),
    mdcg("2021-12", "d90b3f63-1d62-43e6-bf5f-fb32ea7c47a2", "mdcg_2021-12_en.pdf"),
    mdcg("2021-11", "498f68ec-ce00-425d-a722-69ac2be6c1b9", "mdcg_2021-11_en.pdf"),
    mdcg("2021-10", "85f65be1-9a43-44b6-8cdd-779ca16bc843", "mdcg_2021-10_en.pdf"),
    mdcg("2021-9",  "e86b9fc4-9785-4c1a-934d-ba11085ff6aa", "mdcg_2021-9_en.pdf"),
    mdcg("2021-7",  "35a97144-cb8b-4067-b81a-dac6e13a92e8", "mdcg_2021-7_en.pdf"),
    mdcg("2021-4",  "9f23fca0-f407-4e45-a464-2d71b575d1fe", "mdcg_2021-4_en.pdf"),
    mdcg("2021-2",  "3564ab86-8773-4499-8011-58beb77e80af", "mdcg_2021-2_en.pdf"),
    mdcg("2021-1",  "ea0369a7-d86c-465e-9a54-0c0dfb01bc84", "mdcg_2021-1_en.pdf"),

    # ── MDCG 2020 — new/missing ───────────────────────────────────────────────
    mdcg("2020-17", "a3bbc84b-078a-46d7-a281-d68136da6d38", "mdcg_2020-17_en.pdf"),
    mdcg("2020-14", "44dc96aa-e517-4af1-855b-f7fcb4b699c9", "mdcg_2020-14_en.pdf"),
    mdcg("2020-13", "02f50abc-91db-4ad9-b137-6ffedb690716", "mdcg_2020-13_en.pdf"),
    mdcg("2020-12", "ced09394-5757-4eb0-9700-7442959e6c54", "mdcg_2020-12_en.pdf"),
    mdcg("2020-11", "e0c2afff-bce5-44b1-8318-40c40503772a", "mdcg_2020-11_en.pdf"),
    mdcg("2020-10-2", "bf136f27-27da-4a31-97c2-a5de741c3493", "mdcg_2020-10-2_en.pdf"),
    mdcg("2020-4",  "8811a216-fdd1-45c7-bd82-381a37696f05", "mdcg_2020-4_en.pdf"),

    # ── MDCG 2019 — new/missing ───────────────────────────────────────────────
    mdcg("2019-14", "6d75a830-9b9b-4e4a-a3b6-047329e9a104", "mdcg_2019-14_en.pdf"),
    mdcg("2019-13", "36268741-3591-4130-ad21-b09487ff2074", "mdcg_2019-13_en.pdf"),
    mdcg("2019-12", "b934f2bd-e01f-443e-8bb8-87b5291a892e", "mdcg_2019-12_en.pdf"),
    mdcg("2019-8",  "e86b9fc4-9785-4c1a-934d-ba11085ff6aa", "mdcg_2019-8_en.pdf"),
    mdcg("2019-3",  "c66ca211-7390-4d60-90fc-0fc30652f3db", "mdcg_2019-3_en.pdf"),
    mdcg("2019-2",  "12d1f9bd-afe3-42ea-b1a3-20b4986bb3d4", "mdcg_2019-2_en.pdf"),
    mdcg("2019-1",  "f278d247-0020-411f-82a6-1d116691c410", "mdcg_2019-1_en.pdf"),

    # ── MDCG 2018 — UDI & early foundation docs ───────────────────────────────
    mdcg("2018-8",  "f654593f-d3a1-4d45-9590-948f61f80fd5", "mdcg_2018-8_en.pdf"),
    mdcg("2018-7",  "d51c7231-c9be-41bb-9a7f-fcf329ca0bd4", "mdcg_2018-7_en.pdf"),
    mdcg("2018-6",  "22503f80-6e3e-422a-b9c1-106c87a3b8f8", "mdcg_2018-6_en.pdf"),
    mdcg("2018-5",  "4b5b2942-aaab-4b78-a188-df111a8d903e", "mdcg_2018-5_en.pdf"),
    mdcg("2018-4",  "f622a4d7-4ac8-44e3-9217-e2793aa7971f", "mdcg_2018-4_en.pdf"),
    mdcg("2018-3",  "b8fa4255-e011-4253-8249-88febab343f2", "mdcg_2018-3_en.pdf"),
    mdcg("2018-2",  "653cb5b3-7cba-4041-935b-c4f044569700", "mdcg_2018-2_en.pdf"),

    # ── IMDRF — new/additional ────────────────────────────────────────────────
    ("IMDRF_N86_UNIQUE_DEVICE_ID_2024",
     "IMDRF N86 (2024) — Application of Unique Device Identification (UDI) System",
     "https://www.imdrf.org/sites/default/files/2024-02/IMDRF-UDI-N86Final_0.pdf",
     "STANDARDS"),

    ("IMDRF_N82_MACHINE_LEARNING_2022",
     "IMDRF N82 (2022) — Machine Learning-Enabled Medical Devices: Key Terms and Definitions",
     "https://www.imdrf.org/sites/default/files/2022-09/IMDRF-N82Final.pdf",
     "STANDARDS"),

    ("IMDRF_N67_ESSENTIAL_PRINCIPLES_2020",
     "IMDRF N67 (2020) — Essential Principles of Safety and Performance of Medical Devices and IVDs",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200716-samd-essential-principles-n67.pdf",
     "STANDARDS"),

    ("IMDRF_N63_ADVERSE_EVENT_REPORTING_2019",
     "IMDRF N63 (2019) — Principles of Adverse Event Reporting for Medical Devices",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-190319-adverseeventterminology-n63.pdf",
     "STANDARDS"),

    ("IMDRF_N55_PATIENT_REGISTRY_2018",
     "IMDRF N55 (2018) — Registry Methodologies to Support Regulatory Decision-Making",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-180710-patientreg-n55.pdf",
     "STANDARDS"),

    ("IMDRF_N48_LABELLING_2017",
     "IMDRF N48 (2017) — Principles for Labelling Medical Devices and IVDs",
     "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-170320-labellingprinc-n48.pdf",
     "STANDARDS"),

    # ── Japan PMDA — AI guidance ──────────────────────────────────────────────
    ("JP_MHLW_SAMD_REGULATORY_DISCUSSION_2022",
     "Japan MHLW Discussion Paper: Regulatory Approach for SaMD Using AI (2022)",
     "https://www.pmda.go.jp/files/000257901.pdf",
     "JP"),

    # ── Korea MFDS — AI/digital health English PDFs ───────────────────────────
    ("KR_MFDS_AI_REVIEW_APPROVAL_2021",
     "Korea MFDS: Guideline for Review and Approval of AI/ML-based Medical Devices (2021)",
     "https://www.mfds.go.kr/brd/m_14/down.do?brd_id=0055&seq=45466&data_tp=A&file_seq=1",
     "KR"),

    ("KR_MFDS_DIGITAL_THERAPEUTICS_2020",
     "Korea MFDS: Guideline for Review and Approval of Digital Therapeutics (DTx) Software (2020)",
     "https://www.mfds.go.kr/brd/m_14/down.do?brd_id=0055&seq=43875&data_tp=A&file_seq=1",
     "KR"),

    # ── FDA — De Novo and Breakthrough ───────────────────────────────────────
    ("US_FDA_DE_NOVO_GUIDANCE_2021",
     "FDA Guidance: De Novo Classification Process (Evaluation of Automatic Class III Designation) (2021)",
     "https://www.fda.gov/media/72674/download",
     "US"),

    ("US_FDA_BREAKTHROUGH_DEVICE_2023",
     "FDA Guidance: Breakthrough Devices Program (2023)",
     "https://www.fda.gov/media/108135/download",
     "US"),

    ("US_FDA_MULTIPLE_FUNCTION_DEVICES_2019",
     "FDA Guidance: Multiple Function Device Products: Policy and Considerations (2019)",
     "https://www.fda.gov/media/119290/download",
     "US"),

    ("US_FDA_PMA_GUIDANCE_2014",
     "FDA Guidance: Premarket Approval (PMA) — How to Prepare a PMA Application (2014)",
     "https://www.fda.gov/media/80601/download",
     "US"),

    ("US_FDA_MDDT_PROGRAM_2017",
     "FDA Guidance: Medical Device Development Tools (MDDT) Program (2017)",
     "https://www.fda.gov/media/100021/download",
     "US"),

    # ── UAE TRA/MOH ───────────────────────────────────────────────────────────
    ("UA_MOH_MEDICAL_DEVICES_GUIDE_2023",
     "UAE Ministry of Health: Medical Devices and IVDs Registration Requirements Guide (2023)",
     "https://mohap.gov.ae/storage/pages/E0E3eTNiE6YeXkc1SrV8bU0FHa4KYLF57aBJM1r7.pdf",
     "UA"),
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
existing = set(m.get("document_id") or "" for m in store.metadata)
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
    if not (r.content[:4] == b"%PDF" or "pdf" in ct.lower()):
        print(f"  FAIL {doc_id}: not a PDF (ct={ct[:40]})")
        continue

    text = extract_pdf_text(r.content)
    if len(text.split()) < 50:
        print(f"  SKIP {doc_id}: too short")
        continue

    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=title,
        document_id=doc_id,
        source_url=url,
        device_classes=["Class I", "Class IIa", "Class IIb", "Class III",
                        "SaMD", "IVD", "AI/ML", "UDI", "combination product"],
    )
    n = embed_and_index_chunks(chunks, vector_store=store)
    total_new += n
    print(f"  +{n:3d}  {doc_id}  [{country}]", flush=True)

print(f"\nDone: {skipped} skipped, {total_new} new chunks added")
