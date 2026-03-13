#!/usr/bin/env python3
"""
OrthoLink — Master Ingestion: Ingest ALL 15 Countries

This script orchestrates the full ingestion pipeline for every country in scope.
It attempts scraping from official government URLs first, then falls back to
local files in data/raw/{COUNTRY}/.

Usage:
    poetry run python scripts/ingest_all_countries.py
    poetry run python scripts/ingest_all_countries.py --country EU --country UK
    poetry run python scripts/ingest_all_countries.py --skip-existing
    poetry run python scripts/ingest_all_countries.py --file-only   # only ingest from local files
    poetry run python scripts/ingest_all_countries.py --dry-run     # show what would be ingested

Pipeline per document:
    1. Acquire (scrape URL or load file)
    2. Translate (if non-English, via gpt-4o)
    3. Validate (scraper hardening — no garbage)
    4. Chunk   (Article/Section boundary splitting)
    5. Embed   (text-embedding-3-large → FAISS)
    6. CFS     (Certificate of Free Sale chunk per country)
    7. Audit   (sample 10 chunks, verify quality)
"""

import argparse
import glob
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
logger = logging.getLogger("ingest_all")


# ─────────────────────────────────────────────────────────────────────────────
# Per-country ingestion registry
# Each entry: country, document_id, regulation_name, source_url, device_classes,
#             language, local_file_glob (fallback pattern in data/raw/{COUNTRY}/)
# ─────────────────────────────────────────────────────────────────────────────

INGESTION_PLAN = [
    # ══════════════════════════════════════════════════════════════════════════
    # US — FDA (Source: eCFR — official US government regulation database)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "US",
        "document_id": "US-21CFR-807",
        "regulation_name": "21 CFR Part 807 — Establishment Registration and 510(k)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr807_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-812",
        "regulation_name": "21 CFR Part 812 — Investigational Device Exemptions (IDE)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr812_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-814",
        "regulation_name": "21 CFR Part 814 — Premarket Approval (PMA)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr814_ecfr.html",
        "device_classes": ["III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-820",
        "regulation_name": "21 CFR Part 820 — Quality System Regulation (QSR/CGMP)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr820_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-801",
        "regulation_name": "21 CFR Part 801 — Labeling",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr801_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-803",
        "regulation_name": "21 CFR Part 803 — Medical Device Reporting (MDR)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr803_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-806",
        "regulation_name": "21 CFR Part 806 — Reports of Corrections and Removals",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr806_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-860",
        "regulation_name": "21 CFR Part 860 — Medical Device Classification Procedures",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr860_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-830",
        "regulation_name": "21 CFR Part 830 — Unique Device Identification (UDI)",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr830_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-822",
        "regulation_name": "21 CFR Part 822 — Postmarket Surveillance",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr822_ecfr.html",
        "device_classes": ["II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-810",
        "regulation_name": "21 CFR Part 810 — Medical Device Recall Authority",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr810_ecfr.html",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US-21CFR-821",
        "regulation_name": "21 CFR Part 821 — Medical Device Tracking",
        "source_url": "",
        "local_file_glob": "data/raw/US/21cfr821_ecfr.html",
        "device_classes": ["II", "III"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # EU — EUR-Lex (Official Journal of the European Union)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "EU",
        "document_id": "EU_MDR_2017_745",
        "regulation_name": "EU MDR 2017/745",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0745",
        "local_file_glob": "data/raw/EU/eu_mdr_2017_745_eurlex.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "EU",
        "document_id": "EU_IVDR_2017_746",
        "regulation_name": "EU IVDR 2017/746 — In Vitro Diagnostic Regulation",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0746",
        "local_file_glob": "data/raw/EU/eu_ivdr_2017_746_eurlex.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # UK — legislation.gov.uk + MHRA (gov.uk)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "UK",
        "document_id": "UK_MHRA_MDR2002",
        "regulation_name": "UK Medical Devices Regulations 2002",
        "source_url": "https://www.legislation.gov.uk/uksi/2002/618/made/data.htm",
        "local_file_glob": "data/raw/UK/uk_mdr_2002_618_legislation.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "UK",
        "document_id": "UK_MHRA_REGISTER_MFR",
        "regulation_name": "MHRA — Register as a Manufacturer to Sell Medical Devices",
        "source_url": "https://www.gov.uk/guidance/register-as-a-manufacturer-to-sell-medical-devices",
        "local_file_glob": "data/raw/UK/uk_mhra_register_manufacturer.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # CA — Justice Laws Website (Official Canadian legislation)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "CA",
        "document_id": "CA_MDR_SOR98_282",
        "regulation_name": "Canada Medical Devices Regulations SOR/98-282",
        "source_url": "https://laws-lois.justice.gc.ca/eng/regulations/SOR-98-282/FullText.html",
        "local_file_glob": "data/raw/CA/ca_mdr_sor98_282_justice.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    {
        "country": "CA",
        "document_id": "CA_FOOD_DRUGS_ACT",
        "regulation_name": "Canada Food and Drugs Act",
        "source_url": "https://laws-lois.justice.gc.ca/eng/acts/f-27/FullText.html",
        "local_file_glob": "data/raw/CA/ca_food_drugs_act_justice.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # AU — Federal Register of Legislation + TGA
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "AU",
        "document_id": "AU_TGA_ACT_1989",
        "regulation_name": "Therapeutic Goods Act 1989 (Australia)",
        "source_url": "https://www.legislation.gov.au/C2017C00225/latest/text",
        "local_file_glob": "data/raw/AU/au_therapeutic_goods_act_1989.html",
        "device_classes": ["I", "IIa", "IIb", "III", "AIMD"],
        "language": "en",
    },
    {
        "country": "AU",
        "document_id": "AU_TGA_ARTG",
        "regulation_name": "Therapeutic Goods (Medical Devices) Regulations 2002",
        "source_url": "",
        "local_file_glob": "data/raw/AU/tga_md_regulations.*",
        "device_classes": ["I", "IIa", "IIb", "III", "AIMD"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # CH — Fedlex (Official Swiss legislation)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "CH",
        "document_id": "CH_SWISSMEDIC_MDO",
        "regulation_name": "Swiss Medical Devices Ordinance (MedDO)",
        "source_url": "https://www.fedlex.admin.ch/eli/cc/2020/552/en",
        "local_file_glob": "data/raw/CH/ch_meddo_fedlex.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # JP — PMDA (Official Japanese regulator)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "JP",
        "document_id": "JP_PMDA_REGULATORY",
        "regulation_name": "PMDA Regulatory Information — Medical Device Review",
        "source_url": "https://www.pmda.go.jp/english/review-services/regulatory-info/0002.html",
        "local_file_glob": "data/raw/JP/jp_pmda_regulatory_info.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    {
        "country": "JP",
        "document_id": "JP_PMDA_REVIEWS",
        "regulation_name": "PMDA Device Reviews — Medical Device Approval Process",
        "source_url": "https://www.pmda.go.jp/english/review-services/reviews/0002.html",
        "local_file_glob": "data/raw/JP/jp_pmda_device_reviews.html",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    {
        "country": "JP",
        "document_id": "JP_PMDA_PMDACT",
        "regulation_name": "Japan PMD Act",
        "source_url": "",
        "local_file_glob": "data/raw/JP/pmd_act_en.*",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # IN — CDSCO (Central Drugs Standard Control Organisation)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "IN",
        "document_id": "IN_CDSCO_MD_PORTAL",
        "regulation_name": "CDSCO Medical Device Regulations — Official Portal",
        "source_url": "https://cdsco.gov.in/opencms/opencms/en/Medical-Device-Diagnostics/Medical-Device-Diagnostics/",
        "local_file_glob": "data/raw/IN/india_cdsco_md_portal.html",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # CN — NMPA (National Medical Products Administration)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "CN",
        "document_id": "CN_NMPA_ORDER739",
        "regulation_name": "NMPA Order 739 — Medical Device Registration",
        "source_url": "https://www.nmpa.gov.cn/xxgk/fgwj/xzfg/20210831152835170.html",
        "local_file_glob": "data/raw/CN/nmpa_md_regulations_en.*",
        "device_classes": ["I", "II", "III"],
        "language": "zh",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # BR — ANVISA (Agencia Nacional de Vigilancia Sanitaria)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "BR",
        "document_id": "BR_ANVISA_RDC751",
        "regulation_name": "ANVISA RDC 751/2022 — Medical Device Registration",
        "source_url": "https://www.in.gov.br/en/web/dou/-/resolucao-rdc-n-751-de-15-de-setembro-de-2022-430647999",
        "local_file_glob": "data/raw/BR/anvisa_rdc751_en.*",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "pt",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # KR — MFDS (Ministry of Food and Drug Safety)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "KR",
        "document_id": "KR_MFDS_MDA",
        "regulation_name": "South Korea Medical Device Act",
        "source_url": "https://www.mfds.go.kr/eng/brd/m_15/view.do?seq=73047",
        "local_file_glob": "data/raw/KR/mfds_medical_device_act.*",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # UA — Verkhovna Rada (Official Ukrainian legislation database)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "UA",
        "document_id": "UA-RES-753-OFFICIAL",
        "regulation_name": "Resolution 753 — Technical Regulation on Medical Devices (Official)",
        "source_url": "https://zakon.rada.gov.ua/laws/show/753-2020-%D0%BF",
        "local_file_glob": "data/raw/UA/ua_resolution_753_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    {
        "country": "UA",
        "document_id": "UA-RES-754-OFFICIAL",
        "regulation_name": "Resolution 754 — Conformity Assessment (Official)",
        "source_url": "https://zakon.rada.gov.ua/laws/show/754-2020-%D0%BF",
        "local_file_glob": "data/raw/UA/ua_resolution_754_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    {
        "country": "UA",
        "document_id": "UA-RES-755-OFFICIAL",
        "regulation_name": "Resolution 755 — Market Surveillance (Official)",
        "source_url": "https://zakon.rada.gov.ua/laws/show/755-2020-%D0%BF",
        "local_file_glob": "data/raw/UA/ua_resolution_755_rada.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "uk",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # MX — COFEPRIS (Official Mexican health regulator)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "MX",
        "document_id": "MX_COFEPRIS_DEVICES",
        "regulation_name": "COFEPRIS — Medical Device Regulation",
        "source_url": "https://www.gob.mx/cofepris/acciones-y-programas/dispositivos-medicos",
        "local_file_glob": "data/raw/MX/cofepris.*",
        "device_classes": ["I", "II", "III"],
        "language": "es",
    },
    {
        "country": "MX",
        "document_id": "MX_COFEPRIS_NOM241",
        "regulation_name": "COFEPRIS NOM-241-SSA1 Medical Device Regulation",
        "source_url": "",
        "local_file_glob": "data/raw/MX/mx_nom241_en.*",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # RU — Garant.ru (Official Russian legal database)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "RU",
        "document_id": "RU_DECREE_1416_OFFICIAL",
        "regulation_name": "Russia Government Decree 1416 — Medical Device Registration (Official)",
        "source_url": "https://base.garant.ru/70843838/",
        "local_file_glob": "data/raw/RU/ru_decree_1416_garant.html",
        "device_classes": ["1", "2a", "2b", "3"],
        "language": "ru",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # SA — SFDA (Saudi Food and Drug Authority)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "country": "SA",
        "document_id": "SA_SFDA_REGULATIONS",
        "regulation_name": "SFDA Regulations — Medical Devices (Official Portal)",
        "source_url": "https://www.sfda.gov.sa/en/regulations",
        "local_file_glob": "data/raw/SA/sa_sfda_regulations_page.html",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "SA",
        "document_id": "SA_SFDA_MDIR",
        "regulation_name": "SFDA Medical Device Implementing Regulations MDIR",
        "source_url": "",
        "local_file_glob": "data/raw/SA/sfda_mdir_en.*",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
]


def _find_local_file(glob_pattern: str) -> str | None:
    """Find first matching file for a glob pattern."""
    matches = sorted(glob.glob(glob_pattern))
    return matches[0] if matches else None


def _get_existing_countries() -> set[str]:
    """Return set of countries already in the FAISS vector store."""
    from app.tools.vector_store import get_vector_store
    store = get_vector_store()
    return {m.country.upper() for m in store.metadata if hasattr(m, "country") and m.country}


def _get_existing_doc_ids() -> set[str]:
    """Return set of document_ids already in the FAISS vector store."""
    from app.tools.vector_store import get_vector_store
    store = get_vector_store()
    ids = set()
    for m in store.metadata:
        did = getattr(m, "document_id", None) or ""
        if did:
            ids.add(did)
    return ids


def ingest_one(entry: dict, dry_run: bool = False, file_only: bool = False) -> dict:
    """
    Ingest a single document. Returns result dict.
    Tries source_url first, then local file fallback.
    """
    country = entry["country"]
    doc_id = entry["document_id"]
    reg_name = entry["regulation_name"]
    source_url = entry["source_url"]
    local_glob = entry.get("local_file_glob", "")
    device_classes = entry.get("device_classes", [])
    language = entry.get("language", "en")

    # Determine source
    local_file = _find_local_file(local_glob) if local_glob else None

    if file_only and not local_file:
        return {
            "country": country,
            "document_id": doc_id,
            "status": "skipped",
            "reason": f"No local file matching {local_glob}",
        }

    source_desc = ""
    use_url = False
    use_file = False

    if local_file:
        source_desc = f"file: {local_file}"
        use_file = True
    elif source_url and not file_only:
        source_desc = f"url: {source_url}"
        use_url = True
    else:
        return {
            "country": country,
            "document_id": doc_id,
            "status": "skipped",
            "reason": "No source URL and no local file",
        }

    if dry_run:
        return {
            "country": country,
            "document_id": doc_id,
            "status": "dry_run",
            "source": source_desc,
        }

    # Import and run the ingestion pipeline
    from scripts.ingest_country import ingest

    try:
        result = ingest(
            country=country,
            regulation_name=reg_name,
            source_url=source_url if use_url else "",
            file_path=local_file if use_file else "",
            device_classes=device_classes,
            language=language,
            document_id=doc_id,
            skip_translation=(language == "en"),
            skip_audit=False,
        )
        return {
            "country": country,
            "document_id": doc_id,
            "status": "success",
            "source": source_desc,
            **result,
        }
    except Exception as e:
        logger.error(f"Failed to ingest {country}/{doc_id}: {e}")

        # If URL failed, try local file fallback
        if use_url and local_file:
            logger.info(f"Retrying {country}/{doc_id} from local file: {local_file}")
            try:
                result = ingest(
                    country=country,
                    regulation_name=reg_name,
                    source_url="",
                    file_path=local_file,
                    device_classes=device_classes,
                    language=language,
                    document_id=doc_id,
                    skip_translation=(language == "en"),
                    skip_audit=False,
                )
                return {
                    "country": country,
                    "document_id": doc_id,
                    "status": "success_fallback",
                    "source": f"file: {local_file} (fallback)",
                    **result,
                }
            except Exception as e2:
                logger.error(f"Fallback also failed for {country}/{doc_id}: {e2}")

        return {
            "country": country,
            "document_id": doc_id,
            "status": "failed",
            "source": source_desc,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Ingest all 15 countries into FAISS")
    parser.add_argument("--country", action="append", help="Specific country code(s)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip documents already in the vector store")
    parser.add_argument("--file-only", action="store_true",
                        help="Only ingest from local files (no URL scraping)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be ingested without doing it")
    args = parser.parse_args()

    target_countries = set(c.upper() for c in args.country) if args.country else None
    existing_docs = _get_existing_doc_ids() if args.skip_existing else set()

    plan = INGESTION_PLAN
    if target_countries:
        plan = [e for e in plan if e["country"] in target_countries]

    if args.skip_existing:
        plan = [e for e in plan if e["document_id"] not in existing_docs]

    if not plan:
        logger.info("Nothing to ingest. All specified documents are already in the store.")
        return

    logger.info(f"{'DRY RUN: ' if args.dry_run else ''}Ingesting {len(plan)} documents across "
                f"{len(set(e['country'] for e in plan))} countries")

    results = []
    for i, entry in enumerate(plan, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(plan)}] {entry['country']} — {entry['regulation_name']}")
        logger.info(f"{'='*60}")

        result = ingest_one(entry, dry_run=args.dry_run, file_only=args.file_only)
        results.append(result)
        logger.info(f"  → {result['status']}")

        # Rate limit pause between ingestions (embedding API)
        if result["status"] in ("success", "success_fallback") and i < len(plan):
            time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")

    success = [r for r in results if r["status"] in ("success", "success_fallback")]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    dry_runs = [r for r in results if r["status"] == "dry_run"]

    if dry_runs:
        print(f"\nWould ingest {len(dry_runs)} documents:")
        for r in dry_runs:
            print(f"  {r['country']}/{r['document_id']} from {r.get('source', '?')}")
        return

    print(f"\nSucceeded: {len(success)}")
    for r in success:
        chunks = r.get("chunks_embedded", "?")
        print(f"  [OK] {r['country']}/{r['document_id']} -- {chunks} chunks from {r.get('source', '?')}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  [FAIL] {r['country']}/{r['document_id']} -- {r.get('error', '?')}")

    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        for r in skipped:
            print(f"  ⏭️  {r['country']}/{r['document_id']} — {r.get('reason', '?')}")

    # Final brain status
    if success:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        countries = sorted(store.get_countries())
        print(f"\nFinal brain status: {len(store.metadata)} chunks, {len(countries)} countries")
        print(f"Countries: {', '.join(countries)}")
        missing = {"US", "EU", "UK", "UA", "IN", "CA", "AU", "JP", "CN", "BR", "KR", "CH", "MX", "RU", "SA"} - set(countries)
        if missing:
            print(f"[WARN] Still missing: {', '.join(sorted(missing))}")
        else:
            print("[OK] All 15 countries covered!")


if __name__ == "__main__":
    main()
