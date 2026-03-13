#!/usr/bin/env python3
"""
OrthoLink — Weekly Refresh Ingestion (March 2026)

New high-value documents scraped and ready to ingest:
- EU: openregulatory/templates — 91 MDR templates (GSPR, CER, PMS, CAPA, IEC 62304, etc.)
- CA: SOR/98-282 full text — massive upgrade over current 92 CA chunks
- AU: TGA Act 1989 compiled
- JP: PMDA approved devices overview
- BR: ANVISA medical devices portal
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
logger = logging.getLogger("ingest_weekly_refresh")

NEW_DOCS = [
    # EU — openregulatory/templates (91 MIT-licensed MDR/IVDR templates)
    {
        "country": "EU",
        "document_id": "EU_OPENREG_TEMPLATES",
        "regulation_name": "openregulatory/templates — EU MDR/IVDR SOPs, Checklists & Templates (91 docs)",
        "local_file": "data/raw/EU/openreg_mdr_templates_complete.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # CA — Medical Devices Regulations SOR/98-282 full text (32k words)
    {
        "country": "CA",
        "document_id": "CA_MDR_SOR98_282_V2_FULL",
        "regulation_name": "Canada Medical Devices Regulations SOR/98-282 — Full Legislative Text (2026)",
        "local_file": "data/raw/CA/ca_mdr_full_text.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # AU — Therapeutic Goods Act 1989 compiled
    {
        "country": "AU",
        "document_id": "AU_TGA_ACT_1989_COMPILED",
        "regulation_name": "Australia Therapeutic Goods Act 1989 — Compiled Text",
        "local_file": "data/raw/AU/au_tga_act_1989_compiled.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # JP — PMDA approved devices overview
    {
        "country": "JP",
        "document_id": "JP_PMDA_APPROVED_DEVICES",
        "regulation_name": "PMDA — Approved Medical Devices Overview & Classification",
        "local_file": "data/raw/JP/jp_pmda_approved_devices.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # BR — ANVISA medical devices portal
    {
        "country": "BR",
        "document_id": "BR_ANVISA_MD_PORTAL",
        "regulation_name": "ANVISA — Medical Devices Regulatory Portal (Brazil)",
        "local_file": "data/raw/BR/br_anvisa_md_portal.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
]


def main():
    base_dir = Path(__file__).parent.parent

    # Step 1: Load existing document IDs
    metadata_path = base_dir / "data" / "embeddings" / "metadata.json"
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        for m in meta:
            did = m.get("document_id")
            if did:
                existing_ids.add(did)
        logger.info(f"Store: {len(meta)} chunks, {len(existing_ids)} unique document IDs")

    # Step 2: Filter to only new docs, verify local files exist
    to_ingest = []
    for d in NEW_DOCS:
        if d["document_id"] in existing_ids:
            logger.info(f"  SKIP (already in store): {d['document_id']}")
            continue
        local_path = base_dir / d["local_file"]
        if not local_path.exists():
            logger.warning(f"  SKIP (file not found): {d['local_file']}")
            continue
        size_kb = local_path.stat().st_size // 1024
        logger.info(f"  QUEUE: {d['document_id']} ({size_kb} KB)")
        to_ingest.append(d)

    if not to_ingest:
        logger.info("All documents already in store or files missing. Nothing to ingest.")
        return

    logger.info(f"\nWill ingest {len(to_ingest)} documents:")
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
            chunks = result.get("chunks_embedded", "?")
            logger.info(f"  OK: {chunks} chunks embedded")
            results.append({"status": "success", **doc, **result})
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"status": "failed", "error": str(e), **doc})

        if i < len(to_ingest):
            time.sleep(2)

    # Summary
    print(f"\n{'=' * 60}")
    print("WEEKLY REFRESH INGESTION SUMMARY")
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

    # Final store breakdown
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
