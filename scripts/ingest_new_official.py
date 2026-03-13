#!/usr/bin/env python3
"""
OrthoLink — Targeted Ingestion of NEW Official Government Documents

Ingests only documents with document_ids NOT already in the FAISS store.
Skips documents that will fail validation (< 500 words after HTML stripping).
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
logger = logging.getLogger("ingest_new")


# Documents to ingest — ONLY those with NEW document_ids (not in store)
NEW_DOCS = [
    # US — 8 new CFR parts from eCFR (official US gov database)
    {
        "country": "US",
        "document_id": "US-21CFR-807",
        "regulation_name": "21 CFR Part 807 — Establishment Registration and 510(k)",
        "local_file": "data/raw/US/21cfr807_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-812",
        "regulation_name": "21 CFR Part 812 — Investigational Device Exemptions (IDE)",
        "local_file": "data/raw/US/21cfr812_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-814",
        "regulation_name": "21 CFR Part 814 — Premarket Approval (PMA)",
        "local_file": "data/raw/US/21cfr814_ecfr.html",
        "device_classes": ["III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-860",
        "regulation_name": "21 CFR Part 860 — Medical Device Classification Procedures",
        "local_file": "data/raw/US/21cfr860_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-830",
        "regulation_name": "21 CFR Part 830 — Unique Device Identification (UDI)",
        "local_file": "data/raw/US/21cfr830_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-822",
        "regulation_name": "21 CFR Part 822 — Postmarket Surveillance",
        "local_file": "data/raw/US/21cfr822_ecfr.html",
        "device_classes": ["II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-810",
        "regulation_name": "21 CFR Part 810 — Medical Device Recall Authority",
        "local_file": "data/raw/US/21cfr810_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-821",
        "regulation_name": "21 CFR Part 821 — Medical Device Tracking",
        "local_file": "data/raw/US/21cfr821_ecfr.html",
        "device_classes": ["II", "III"],
        "language": "en",
    },
    # EU — IVDR from EUR-Lex
    {
        "country": "EU",
        "document_id": "EU_IVDR_2017_746",
        "regulation_name": "EU IVDR 2017/746 — In Vitro Diagnostic Regulation",
        "local_file": "data/raw/EU/eu_ivdr_2017_746_eurlex.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # UK — MHRA registration guidance
    {
        "country": "UK",
        "document_id": "UK_MHRA_REGISTER_MFR",
        "regulation_name": "MHRA — Register as a Manufacturer to Sell Medical Devices",
        "local_file": "data/raw/UK/uk_mhra_register_manufacturer.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # CA — Food and Drugs Act
    {
        "country": "CA",
        "document_id": "CA_FOOD_DRUGS_ACT",
        "regulation_name": "Canada Food and Drugs Act",
        "local_file": "data/raw/CA/ca_food_drugs_act_justice.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # IN — CDSCO portal
    {
        "country": "IN",
        "document_id": "IN_CDSCO_MD_PORTAL",
        "regulation_name": "CDSCO Medical Device Regulations — Official Portal",
        "local_file": "data/raw/IN/india_cdsco_md_portal.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # UA — 3 official resolutions from Verkhovna Rada
    {
        "country": "UA",
        "document_id": "UA-RES-753-OFFICIAL",
        "regulation_name": "Resolution 753 — Technical Regulation on Medical Devices (Official Rada)",
        "local_file": "data/raw/UA/ua_resolution_753_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    {
        "country": "UA",
        "document_id": "UA-RES-754-OFFICIAL",
        "regulation_name": "Resolution 754 — Conformity Assessment (Official Rada)",
        "local_file": "data/raw/UA/ua_resolution_754_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    {
        "country": "UA",
        "document_id": "UA-RES-755-OFFICIAL",
        "regulation_name": "Resolution 755 — Market Surveillance (Official Rada)",
        "local_file": "data/raw/UA/ua_resolution_755_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    # RU — Official Decree from garant.ru
    {
        "country": "RU",
        "document_id": "RU_DECREE_1416_OFFICIAL",
        "regulation_name": "Russia Government Decree 1416 — Medical Device Registration (Official)",
        "local_file": "data/raw/RU/ru_decree_1416_garant.html",
        "device_classes": ["1", "2a", "2b", "3"],
        "language": "ru",
    },
]


def main():
    # Step 1: Check which doc IDs are already in the store
    metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "embeddings", "metadata.json"
    )
    existing_ids = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        for m in meta:
            did = m.get("document_id", "")
            if did:
                existing_ids.add(did)
        logger.info(f"Found {len(existing_ids)} existing document IDs in store")

    # Step 2: Filter to only truly new docs
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

        # Rate limit pause between ingestions
        if i < len(to_ingest):
            time.sleep(1)

    # Summary
    print(f"\n{'=' * 60}")
    print("INGESTION SUMMARY")
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

    # Final counts
    from app.tools.vector_store import get_vector_store
    store = get_vector_store()
    countries = sorted(store.get_countries())
    from collections import Counter
    per_country = Counter(m.country.upper() for m in store.metadata if m.is_active)
    print(f"\nFinal store: {len(store.metadata)} total chunks, {len(countries)} countries")
    for c in countries:
        print(f"  {c}: {per_country.get(c, 0)} chunks")


if __name__ == "__main__":
    main()
