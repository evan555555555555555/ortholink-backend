#!/usr/bin/env python3
"""
OrthoLink — Priority Accuracy Ingestion (March 2026)

Fetches and indexes the 8 critical missing regulatory documents needed
to reach 90%+ accuracy before first customer handoff.

Critical gaps being closed:
  1. EU Reg 2024/1860  — MDR transition extension (wrong deadlines without this)
  2. FDA QMSR 2026     — 21 CFR Part 820 (Feb 2026 rewrite, replaces old QSR)
  3. FDA ISO 14971 guidance — substantive FDA risk management guidance (free)
  4. FDA CDS guidance 2026  — Clinical Decision Support Software (Jan 2026)
  5. Health Canada Class II-IV guidance — thin CA coverage
  6. CDSCO India AI/ML guidance — 2023 SaMD guidance document
  7. WHO MeDevIS        — global device classification reference
  8. STANDARDS gap      — FDA QS guidance cross-references ISO 13485/14971/62304

All documents are official government sources or FDA/WHO/Health Canada
public guidance — no paywalled ISO standard text reproduced.

Run from backend/:
    source .venv/bin/activate
    python scripts/ingest_priority_accuracy_docs.py
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
logger = logging.getLogger("ingest_priority")

# ─────────────────────────────────────────────────────────────────────────────
# Priority documents — all public government sources
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_DOCS = [
    # ── 1. EU Regulation 2024/1860 ──────────────────────────────────────────
    # MDR transition timeline extension (Regulation (EU) 2024/1860).
    # Without this, RAA will cite WRONG compliance deadlines for legacy devices.
    # Effective: 23 Oct 2024. Extends transitional periods through 31 Dec 2028.
    {
        "country": "EU",
        "document_id": "EU_MDR_2024_1860",
        "regulation_name": "EU Regulation 2024/1860 — MDR Transitional Period Extension",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1860",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
        "notes": "Critical: corrects MDR deadline answers. Extends to Dec 2028 for legacy devices.",
    },
    # ── 2. FDA QMSR 21 CFR Part 820 (Feb 2026 current text) ─────────────────
    # Quality Management System Regulation — replaced QSR on Feb 2, 2026.
    # Aligns 21 CFR 820 with ISO 13485. Old QSR (32 chunks) is outdated.
    {
        "country": "US",
        "document_id": "US_FDA_QMSR_2026",
        "regulation_name": "FDA 21 CFR Part 820 — Quality Management System Regulation (QMSR, Feb 2026)",
        "source_url": "https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820",
        "device_classes": ["I", "II", "III"],
        "language": "en",
        "notes": "Feb 2026 QMSR — replaces old QSR. Harmonised with ISO 13485:2016.",
    },
    # ── 3. FDA Guidance: Factors to Consider When Making Benefit-Risk Determinations ──
    # FDA's substantive guidance on ISO 14971 risk management framework.
    # Fills ISO_14971_2019 gap (currently 1 chunk) without reproducing paywall content.
    {
        "country": "US",
        "document_id": "US_FDA_BENEFIT_RISK_GUIDANCE",
        "regulation_name": "FDA Guidance — Factors to Consider When Making Benefit-Risk Determinations in Medical Device Premarket Approval",
        "source_url": "https://www.fda.gov/media/99769/download",
        "device_classes": ["II", "III"],
        "language": "en",
        "notes": "FDA risk framework guidance — complements ISO 14971 indexing.",
    },
    # ── 4. FDA QMSR Preamble + ISO 13485 Comparison ─────────────────────────
    # FDA's QMSR final rule preamble — includes detailed ISO 13485 comparison tables.
    # Already scraped as fda_qmsreg_preamble.html, but NOT yet indexed.
    {
        "country": "US",
        "document_id": "US_FDA_QMSR_PREAMBLE",
        "regulation_name": "FDA QMSR Final Rule Preamble — ISO 13485 Alignment Analysis (2022)",
        "source_url": "",  # local file
        "local_file": "data/raw/US/fda_qmsreg_preamble.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
        "notes": "Contains FDA-ISO 13485 mapping tables — fills STANDARDS gap.",
    },
    # ── 5. FDA CDS Guidance Jan 2026 ────────────────────────────────────────
    # Clinical Decision Support Software guidance (Jan 2026 update).
    # Critical for AI/SaMD queries across all 15 markets.
    {
        "country": "US",
        "document_id": "US_FDA_CDS_GUIDANCE_2026",
        "regulation_name": "FDA Guidance — Clinical Decision Support Software (Jan 2026)",
        "source_url": "https://www.fda.gov/media/211464/download",
        "device_classes": ["I", "II", "III"],
        "language": "en",
        "notes": "Updated CDS/SaMD guidance — required for AI device classification queries.",
    },
    # ── 6. Health Canada Medical Device Licensing Guidance ───────────────────
    # Guidance on Class II, III, IV device licensing requirements.
    # CA coverage is thin (137 chunks). This is the primary licensing reference.
    {
        "country": "CA",
        "document_id": "CA_HEALTH_CANADA_CLASS_IIIIV_GUIDANCE",
        "regulation_name": "Health Canada — Guidance on Medical Device Licensing for Class II, III, IV Devices",
        "source_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents/guidance-document-medical-device-licensing.html",
        "device_classes": ["II", "III", "IV"],
        "language": "en",
        "notes": "Primary licensing guidance for CA. Fills thin CA coverage.",
    },
    # ── 7. Health Canada Medical Devices Bureau — Overview ───────────────────
    {
        "country": "CA",
        "document_id": "CA_HEALTH_CANADA_MDB_OVERVIEW",
        "regulation_name": "Health Canada Medical Devices Bureau — Regulatory Overview and Guidance Index",
        "source_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/guidance-documents.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
        "notes": "Guidance document index — CA coverage boost.",
    },
    # ── 8. CDSCO India — AI/ML-Based SaMD Guidance ──────────────────────────
    # India's 2023 guidance on AI/ML-based Software as Medical Device.
    # Fills gap for IN SaMD queries.
    {
        "country": "IN",
        "document_id": "IN_CDSCO_AIML_SAMD_GUIDANCE",
        "regulation_name": "CDSCO India — Guidance on AI/ML-Based Software as a Medical Device (SaMD) 2023",
        "source_url": "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Medical-Device-Diagnostics/Guidance-document-on-artificial-intelligence-machine-learning-based-medical-devices-FINAL-CLEAN.pdf",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
        "notes": "India AI/SaMD guidance — critical for IN AI device queries.",
    },
    # ── 9. WHO MeDevIS — Global Medical Device Classification Reference ───────
    # WHO Medical Device Information System — global classification framework.
    # Referenced in FAISS queries across multiple countries.
    {
        "country": "STANDARDS",
        "document_id": "WHO_MEDEVIS_CLASSIFICATION",
        "regulation_name": "WHO Medical Device Information System (MeDevIS) — Global Classification Reference",
        "source_url": "https://www.who.int/teams/health-product-and-policy-standards/assistive-and-medical-technology/medical-devices/medevis",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
        "notes": "WHO global framework — fills STANDARDS gap.",
    },
    # ── 10. IMDRF SaMD Framework ─────────────────────────────────────────────
    # International Medical Device Regulators Forum — SaMD guidance.
    # Referenced by FDA, EU, AU, CA, JP regulators for software classification.
    {
        "country": "STANDARDS",
        "document_id": "IMDRF_SAMD_FRAMEWORK",
        "regulation_name": "IMDRF — Software as a Medical Device (SaMD): Key Definitions and Framework",
        "source_url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-131209-samd-key-definitions-140901.pdf",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
        "notes": "Core IMDRF SaMD framework — referenced by 10+ country regulators.",
    },
    # ── 11. IMDRF Principles of Labelling ────────────────────────────────────
    {
        "country": "STANDARDS",
        "document_id": "IMDRF_LABELLING_N18",
        "regulation_name": "IMDRF N18 FINAL — Principles of Labelling for Medical Devices and IVDs",
        "source_url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-121102-labelling-n18.pdf",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
        "notes": "Global labelling standard — referenced across all 15 markets.",
    },
    # ── 12. EU MDR Annex I — General Safety and Performance Requirements ──────
    # EU MDR Annex I full text scraped separately.
    # The MDR 2017/745 is indexed (1260 chunks) but verify Annex I coverage.
    {
        "country": "EU",
        "document_id": "EU_MDR_MEDDEV_271_REV4",
        "regulation_name": "EU MEDDEV 2.7.1 Rev 4 — Clinical Evaluation: Guide for Manufacturers and Notified Bodies",
        "source_url": "",  # local file
        "local_file": "data/raw/EU/eu_meddev_2_7_1_rev4.html",
        "device_classes": ["IIa", "IIb", "III"],
        "language": "en",
        "notes": "CER guidance — critical for EU clinical evaluation queries.",
    },
    # ── 13. EU MDCG 2021-24 PMCF ─────────────────────────────────────────────
    {
        "country": "EU",
        "document_id": "EU_MDCG_2021_24_PMCF",
        "regulation_name": "EU MDCG 2021-24 — Guidance on Post-Market Clinical Follow-up (PMCF)",
        "source_url": "",  # local file
        "local_file": "data/raw/EU/eu_mdcg_2021_24_pmcf.html",
        "device_classes": ["IIa", "IIb", "III"],
        "language": "en",
        "notes": "PMCF plan guidance — required for EU PMS queries.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    base_dir = Path(__file__).parent.parent
    metadata_path = base_dir / "data" / "embeddings" / "metadata.json"

    # Step 1: Load existing document IDs
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        for m in meta:
            did = m.get("document_id")
            if did:
                existing_ids.add(did)
        logger.info("Store: %d chunks, %d unique document IDs", len(meta), len(existing_ids))

    # Step 2: Filter to only new documents
    to_ingest = [d for d in PRIORITY_DOCS if d["document_id"] not in existing_ids]

    if not to_ingest:
        logger.info("All priority documents already in store. Nothing to ingest.")
        _print_coverage_summary()
        return

    logger.info("Will ingest %d new priority documents:", len(to_ingest))
    for d in to_ingest:
        logger.info("  [%s] %s — %s", d["country"], d["document_id"], d["regulation_name"][:60])

    # Step 3: Ingest each document
    from scripts.ingest_country import ingest

    results = []
    for i, doc in enumerate(to_ingest, 1):
        logger.info("")
        logger.info("=" * 65)
        logger.info("[%d/%d] %s — %s", i, len(to_ingest), doc["country"], doc["regulation_name"])
        logger.info("=" * 65)

        # Resolve file path relative to base_dir
        file_path = ""
        if doc.get("local_file"):
            resolved = str(base_dir / doc["local_file"])
            if os.path.exists(resolved):
                file_path = resolved
                logger.info("Using local file: %s", file_path)
            else:
                logger.warning("Local file not found: %s — will try source_url", resolved)

        source_url = doc.get("source_url", "")

        if not file_path and not source_url:
            logger.error("  SKIPPED: no file_path or source_url for %s", doc["document_id"])
            results.append({"status": "skipped", "reason": "no source", **doc})
            continue

        try:
            result = ingest(
                country=doc["country"],
                regulation_name=doc["regulation_name"],
                source_url=source_url if not file_path else "",
                file_path=file_path,
                device_classes=doc.get("device_classes", []),
                language=doc.get("language", "en"),
                document_id=doc["document_id"],
                skip_translation=True,
                skip_audit=False,
            )
            chunks = result.get("chunks_embedded", 0)
            logger.info("  OK: %d chunks embedded", chunks)
            results.append({"status": "success", **doc, **result})
        except Exception as e:
            logger.error("  FAILED: %s", e, exc_info=True)
            results.append({"status": "failed", "error": str(e), **doc})

        # Brief pause to avoid rate-limiting eCFR/EUR-Lex
        if i < len(to_ingest):
            time.sleep(2)

    # Step 4: Summary
    _print_results(results)
    _print_coverage_summary()


def _print_results(results: list[dict]) -> None:
    print()
    print("=" * 65)
    print("PRIORITY ACCURACY INGESTION SUMMARY")
    print("=" * 65)

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    total_chunks = sum(r.get("chunks_embedded", 0) for r in success)
    print(f"\n✓ Succeeded: {len(success)} documents, {total_chunks} new chunks")
    for r in success:
        print(f"  [{r['country']}] {r['document_id']}: {r.get('chunks_embedded', '?')} chunks")

    if skipped:
        print(f"\n[WARN] Skipped: {len(skipped)}")
        for r in skipped:
            print(f"  [{r['country']}] {r['document_id']}: {r.get('reason', '?')}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for r in failed:
            print(f"  [{r['country']}] {r['document_id']}: {r.get('error', '?')[:120]}")


def _print_coverage_summary() -> None:
    try:
        from app.tools.vector_store import get_vector_store
        from collections import Counter

        store = get_vector_store()
        per_country = Counter(
            m.country.upper() for m in store.metadata if getattr(m, "is_active", True)
        )
        total = sum(per_country.values())
        countries = sorted(per_country.keys())

        print()
        print("=" * 65)
        print(f"FINAL VECTOR STORE: {total} total chunks")
        print("=" * 65)
        for c in countries:
            n = per_country[c]
            bar = "█" * min(40, n // 100)
            print(f"  {c:12s} {n:6d}  {bar}")

        # Confidence tiers
        print()
        print("CONFIDENCE TIERS (post-ingestion):")
        tiers = {
            "HIGH (≥80%)": [],
            "MEDIUM (60–79%)": [],
            "LOW (<60%)": [],
        }
        thresholds = {"US": 90, "AU": 87, "JP": 85, "EU": 90, "KR": 80, "UK": 80,
                      "IN": 75, "UA": 75, "MX": 72, "SA": 72, "CA": 0, "CN": 0,
                      "CH": 0, "BR": 0, "RU": 0, "STANDARDS": 0}
        for c in countries:
            n = per_country[c]
            # Simple heuristic: >2000 chunks = high, >500 = medium, else low
            if n >= 2000:
                tiers["HIGH (≥80%)"].append(f"{c}:{n}")
            elif n >= 400:
                tiers["MEDIUM (60–79%)"].append(f"{c}:{n}")
            else:
                tiers["LOW (<60%)"].append(f"{c}:{n}")

        for tier, items in tiers.items():
            if items:
                print(f"  {tier}: {', '.join(items)}")

    except Exception as e:
        logger.warning("Coverage summary failed: %s", e)


if __name__ == "__main__":
    main()
