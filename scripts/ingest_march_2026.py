#!/usr/bin/env python3
"""
OrthoLink — March 2026 Batch Ingestion

Ingests newly scraped government regulatory documents:
- US: 21 CFR 803 (MDR/Adverse Events), 801 (Labeling), 806 (Corrections/Removals)
- UK: UK MDR 2002 (SI 2002/618) full legislation
- CA: Canada MDR SOR/98-282
- AU: TGA Essential Principles Checklist, Conformity Assessment
- IN: CDSCO Medical Device portal (comprehensive)
- STANDARDS: ISO 13485:2016 overview, IEC 62304 overview, ISO 14971 overview
- JP: PMDA Q-sub guidance
- SA: SFDA classification guide (comprehensive portal)
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
logger = logging.getLogger("ingest_march_2026")

NEW_DOCS = [
    # US — 21 CFR 803 (Medical Device Reporting — Adverse Events)
    {
        "country": "US",
        "document_id": "US-21CFR-803",
        "regulation_name": "21 CFR Part 803 — Medical Device Reporting (MDR/Adverse Events)",
        "local_file": "data/raw/US/21cfr803_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # US — 21 CFR 801 (Medical Device Labeling)
    {
        "country": "US",
        "document_id": "US-21CFR-801",
        "regulation_name": "21 CFR Part 801 — Medical Device Labeling Requirements",
        "local_file": "data/raw/US/21cfr801_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # US — 21 CFR 806 (Corrections and Removals)
    {
        "country": "US",
        "document_id": "US-21CFR-806",
        "regulation_name": "21 CFR Part 806 — Medical Device Corrections and Removals",
        "local_file": "data/raw/US/21cfr806_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # UK — Full UK MDR 2002 legislation text
    {
        "country": "UK",
        "document_id": "UK_MDR2002_FULL_LEGISLATION",
        "regulation_name": "UK Medical Devices Regulations 2002 (SI 2002/618) — Full Text",
        "local_file": "data/raw/UK/uk_legislation_mdr2002.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # Canada — Medical Devices Regulations SOR/98-282
    {
        "country": "CA",
        "document_id": "CA_MDR_SOR98_282_FULL",
        "regulation_name": "Canada Medical Devices Regulations SOR/98-282 — Full Text",
        "local_file": "data/raw/CA/ca_mdr_sor98_282_full.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # Australia — TGA Essential Principles Checklist
    {
        "country": "AU",
        "document_id": "AU_TGA_ESSENTIAL_PRINCIPLES",
        "regulation_name": "TGA Essential Principles Checklist for Medical Devices",
        "local_file": "data/raw/AU/au_tga_essential_principles.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # Australia — TGA Conformity Assessment Procedures
    {
        "country": "AU",
        "document_id": "AU_TGA_CONFORMITY_ASSESSMENT",
        "regulation_name": "TGA Conformity Assessment Procedures for Medical Devices",
        "local_file": "data/raw/AU/au_tga_conformity_assessment.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # India — CDSCO Medical Device comprehensive portal
    {
        "country": "IN",
        "document_id": "IN_CDSCO_MD_COMPREHENSIVE",
        "regulation_name": "CDSCO Medical Device & Diagnostics — Comprehensive Portal",
        "local_file": "data/raw/IN/india_cdsco_class_b_application.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # STANDARDS — ISO 13485:2016
    {
        "country": "STANDARDS",
        "document_id": "ISO_13485_2016",
        "regulation_name": "ISO 13485:2016 — Medical Devices Quality Management Systems",
        "local_file": "data/raw/STANDARDS/iso_13485_2016_overview.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # STANDARDS — ISO 14971:2019
    {
        "country": "STANDARDS",
        "document_id": "ISO_14971_2019",
        "regulation_name": "ISO 14971:2019 — Medical Devices Risk Management",
        "local_file": "data/raw/STANDARDS/iso_14971_2019_overview.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # STANDARDS — IEC 62304:2006
    {
        "country": "STANDARDS",
        "document_id": "IEC_62304_2006",
        "regulation_name": "IEC 62304:2006 — Medical Device Software Lifecycle Processes",
        "local_file": "data/raw/STANDARDS/iec_62304_2006_overview.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # Japan — PMDA Q-sub equivalent guidance
    {
        "country": "JP",
        "document_id": "JP_PMDA_CONSULTATION_GUIDANCE",
        "regulation_name": "PMDA — Pre-Approval Consultation (Q-sub Equivalent) Guidance",
        "local_file": "data/raw/JP/jp_pmda_q_sub_guidance.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # Saudi Arabia — SFDA comprehensive classification guide
    {
        "country": "SA",
        "document_id": "SA_SFDA_CLASSIFICATION_PORTAL",
        "regulation_name": "SFDA Medical Devices Classification and Regulatory Portal",
        "local_file": "data/raw/SA/sa_sfda_classification_guide.html",
        "device_classes": ["A", "B", "C", "D"],
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

    # Step 3: Ingest
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
                skip_translation=True,
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
    print("MARCH 2026 INGESTION SUMMARY")
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
