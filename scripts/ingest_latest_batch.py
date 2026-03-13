#!/usr/bin/env python3
"""
OrthoLink — Latest Batch Ingestion (March 2026)

Ingests newly downloaded regulatory files that are not yet in the FAISS store.
Files include AU TGA Act volumes, JP enforcement regs, MX NOM-241 SIDOF,
RU Decree 1684 2024, RU Roszdravnadzor EN, SA MDS-G5/REQ9, CN NMPA new rules.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ingest_batch")

# New documents to ingest — skipped automatically if document_id already in store
NEW_DOCS = [
    # AU — TGA Act 1989 volumes (the parent Act, not yet in store)
    {
        "country": "AU",
        "document_id": "AU_TGA_ACT_1989_VOL1",
        "regulation_name": "Therapeutic Goods Act 1989 — Volume 1 (Australia)",
        "local_file": "data/raw/AU/au_tga_act_1989_vol1.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "AU",
        "document_id": "AU_TGA_ACT_1989_VOL2",
        "regulation_name": "Therapeutic Goods Act 1989 — Volume 2 (Australia)",
        "local_file": "data/raw/AU/au_tga_act_1989_vol2.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # JP — PMD Act Enforcement Regulation (English translation)
    {
        "country": "JP",
        "document_id": "JP_PMD_ENFORCEMENT_REG",
        "regulation_name": "PMD Act Enforcement Regulation — English Translation (Japan)",
        "local_file": "data/raw/JP/jp_pmd_enforcement_regulation_en.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # MX — NOM-241 from SIDOF portal (alternate official source)
    {
        "country": "MX",
        "document_id": "MX_NOM241_SSA1_2021_SIDOF",
        "regulation_name": "NOM-241-SSA1-2021 Good Manufacturing Practices — SIDOF (Mexico)",
        "local_file": "data/raw/MX/mx_nom241_ssa1_2021_sidof.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # RU — Decree 1684 (2024 updated registration rules)
    {
        "country": "RU",
        "document_id": "RU_DECREE_1684_2024",
        "regulation_name": "Russia Government Decree 1684 (2024) — Updated Medical Device Registration Rules",
        "local_file": "data/raw/RU/ru_decree_1684_2024_new_registration_rules.html",
        "device_classes": ["1", "2a", "2b", "3"],
        "language": "ru",
    },
    # RU — Roszdravnadzor registration guidance in English
    {
        "country": "RU",
        "document_id": "RU_ROSZDRAV_REGISTRATION_EN",
        "regulation_name": "Roszdravnadzor Medical Device Registration Guidance — English",
        "local_file": "data/raw/RU/ru_roszdravnadzor_registration_en.html",
        "device_classes": ["1", "2a", "2b", "3"],
        "language": "en",
    },
    # SA — SFDA MDS-G5 Marketing Authorization Guidance
    {
        "country": "SA",
        "document_id": "SA_SFDA_MDS_G5",
        "regulation_name": "SFDA MDS-G5 — Marketing Authorization Guidance (Saudi Arabia)",
        "local_file": "data/raw/SA/sa_sfda_mds_g5_marketing_authorization.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # SA — SFDA MDS-REQ9 Licensing Requirements
    {
        "country": "SA",
        "document_id": "SA_SFDA_MDS_REQ9",
        "regulation_name": "SFDA MDS-REQ9 — Medical Device Licensing Requirements (Saudi Arabia)",
        "local_file": "data/raw/SA/sa_sfda_mds_req9_licensing.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # CN — NMPA Order 739 new rules (2021 revision)
    {
        "country": "CN",
        "document_id": "CN_NMPA_ORDER739_NEW_RULES",
        "regulation_name": "NMPA Order 739 — Revised Medical Device Supervision Regulations (2021)",
        "local_file": "data/raw/CN/cn_nmpa_order739_new_rules.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # CN — State Council Order 739 summary
    {
        "country": "CN",
        "document_id": "CN_STATE_COUNCIL_ORDER739",
        "regulation_name": "State Council Order 739 — Medical Device Supervision Summary (China)",
        "local_file": "data/raw/CN/cn_state_council_order739_summary.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
]


def main():
    # Step 1: Load existing document IDs
    metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "embeddings", "metadata.json"
    )
    existing_ids: set[str] = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        for m in meta:
            did = m.get("document_id")
            if did:
                existing_ids.add(did)
        logger.info(f"Store: {len(meta)} chunks, {len(existing_ids)} unique document IDs")

    # Step 2: Filter to only new
    to_ingest = [d for d in NEW_DOCS if d["document_id"] not in existing_ids]

    if not to_ingest:
        logger.info("All documents already in store. Nothing to ingest.")
        return

    logger.info(f"Will ingest {len(to_ingest)} new documents:")
    for d in to_ingest:
        logger.info(f"  {d['country']}/{d['document_id']}")

    # Step 3: Ingest each document
    from scripts.ingest_country import ingest

    results = []
    for i, doc in enumerate(to_ingest, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{i}/{len(to_ingest)}] {doc['country']} — {doc['regulation_name']}")
        logger.info(f"{'=' * 60}")

        try:
            result = ingest(
                country=doc["country"],
                regulation_name=doc["regulation_name"],
                source_url="",
                file_path=doc["local_file"],
                device_classes=doc["device_classes"],
                language=doc["language"],
                document_id=doc["document_id"],
                skip_translation=(doc["language"] == "en"),
                skip_audit=False,
            )
            logger.info(f"  OK: {result.get('chunks_embedded', '?')} chunks embedded")
            results.append({"status": "success", **doc, **result})
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"status": "failed", "error": str(e), **doc})

        if i < len(to_ingest):
            time.sleep(1)

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH INGESTION SUMMARY")
    print(f"{'=' * 60}")

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    total_chunks = sum(r.get("chunks_embedded", 0) for r in success)
    print(f"\nSucceeded: {len(success)} documents, {total_chunks} total new chunks")
    for r in success:
        print(f"  {r['country']}/{r['document_id']}: {r.get('chunks_embedded', '?')} chunks")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['country']}/{r['document_id']}: {r.get('error', '?')}")

    # Final store counts
    from app.tools.vector_store import get_vector_store
    from collections import Counter
    store = get_vector_store()
    per_country = Counter(m.country.upper() for m in store.metadata if m.is_active)
    countries = sorted(per_country.keys())
    print(f"\nFinal store: {len(store.metadata)} total chunks")
    for c in countries:
        print(f"  {c}: {per_country[c]} chunks")


if __name__ == "__main__":
    main()
