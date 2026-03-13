#!/usr/bin/env python3
"""
OrthoLink Ukraine Ingestion (A2)
Scrape Resolutions 753, 754, 755 from Rada → validate → translate → chunk → embed → CFS → audit.
Run: poetry run python scripts/ingest_ukraine.py
Exits 1 if chunk_audit --country UA --sample 10 has any failure.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path

_backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_backend_root))
os.chdir(_backend_root)
# Ensure config is re-read from backend/.env
from app.core.config import get_settings
get_settings.cache_clear()

from app.ingestion.chunker import Chunk, chunk_regulatory_text
# CFS synthetic seeding removed — all chunks must come from real regulatory sources.
from app.ingestion.chunk_audit import audit_chunks
from app.ingestion.embedder import embed_and_index_chunks
from app.ingestion.scraper import load_from_file, scrape_url, validate_scraped_text
from app.ingestion.translator import translate_ukrainian_regulatory_to_english
from app.tools.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ingest_ukraine")

UA_RESOLUTIONS = [
    ("753", "https://zakon.rada.gov.ua/laws/show/753-2013-%D0%BF", "Resolution 753: Technical Regulation on Medical Devices"),
    ("754", "https://zakon.rada.gov.ua/laws/show/754-2013-%D0%BF", "Resolution 754: Conformity Assessment"),
    ("755", "https://zakon.rada.gov.ua/laws/show/755-2013-%D0%BF", "Resolution 755: Market Surveillance"),
]
COUNTRY = "UA"
DEVICE_CLASSES_UA = ["IIa", "IIb", "III"]


def _enrich_ua_chunk(chunk: Chunk, document_id: str, document_title: str) -> Chunk:
    """Set document_id and section_path; ensure chunk_hash."""
    section_path = f"{document_title} > {chunk.article}"
    if chunk.clause:
        section_path += f" > Clause {chunk.clause}"
    clean_text = chunk.text.strip()
    chunk_hash = hashlib.sha256(clean_text.encode()).hexdigest()
    return Chunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        parent_text=chunk.parent_text,
        country=chunk.country,
        regulation_name=document_title,
        article=chunk.article,
        clause=chunk.clause,
        device_classes=chunk.device_classes,
        source_url=chunk.source_url,
        language="en",
        original_language="uk",
        chunk_hash=chunk_hash,
        document_id=document_id,
        section_path=section_path,
    )


DOCUMENT_TITLES = {
    "753": "Resolution 753: Technical Regulation on Medical Devices",
    "754": "Resolution 754: Conformity Assessment",
    "755": "Resolution 755: Market Surveillance",
}


def _acquire_text(resolution_num: str, url: str, document_title: str, file_dir: str | None) -> str:
    """Get raw Ukrainian text: scrape URL or load from file_dir/{num}.html."""
    if file_dir:
        path = Path(file_dir) / f"{resolution_num}.html"
        if path.exists():
            logger.info(f"Loading {document_title} from {path}")
            result = load_from_file(str(path), extract_rada_main=True)
            if result.success and result.text.strip():
                return result.text
            logger.warning(f"File {path} produced no text, trying scrape")
    logger.info(f"Scraping {document_title}: {url}")
    result = scrape_url(url)
    if not result.success:
        raise RuntimeError(f"Scrape failed for {url}: {result.error}")
    return result.text


def ingest_one_from_file(
    file_path: str,
    document_id: str,
    language: str = "uk",
    regulation_name: str | None = None,
) -> int:
    """Load one PDF/file, translate (if uk), validate, chunk, embed. Returns chunk count."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Derive resolution number and title from document_id (e.g. UA-RES-753)
    res_num = document_id.replace("UA-RES-", "").strip() if document_id.startswith("UA-RES-") else path.stem
    document_title = regulation_name or DOCUMENT_TITLES.get(res_num, f"Resolution {res_num}")

    logger.info(f"Loading {document_title} from {file_path}")
    result = load_from_file(str(path))
    if not result.success:
        raise RuntimeError(f"Load failed: {result.error}")
    raw_text = result.text
    if not raw_text or len(raw_text.strip()) < 100:
        raise RuntimeError(f"File produced too little text ({len(raw_text.strip())} chars)")

    if language and language != "en":
        logger.info(f"Translating {document_title} (gpt-4o)...")
        trans = translate_ukrainian_regulatory_to_english(raw_text)
        if not trans.get("success"):
            raise RuntimeError(f"Translation failed: {trans.get('error')}")
        raw_text = trans["translated_text"]

    validation = validate_scraped_text(raw_text, COUNTRY, file_path)
    if not validation.passed:
        raise RuntimeError(f"Validation failed: {validation.warnings}")

    chunks = chunk_regulatory_text(
        text=raw_text,
        country=COUNTRY,
        regulation_name=document_title,
        device_classes=DEVICE_CLASSES_UA,
        source_url=file_path,
        language="en",
        original_language=language if language != "en" else None,
        document_id=document_id,
    )
    enriched = [_enrich_ua_chunk(c, document_id, document_title) for c in chunks]
    store = get_vector_store()
    n = embed_and_index_chunks(enriched, vector_store=store)
    store.save()
    logger.info(f"Indexed {n} chunks for {document_id}")
    return n


def ingest_one_resolution(resolution_num: str, url: str, document_title: str, file_dir: str | None = None) -> int:
    """Scrape or load, validate, translate, chunk, return count of chunks created."""
    raw_text = _acquire_text(resolution_num, url, document_title, file_dir)
    logger.info(f"Translating {document_title} (gpt-4o)...")
    trans = translate_ukrainian_regulatory_to_english(raw_text)
    if not trans.get("success"):
        raise RuntimeError(f"Translation failed for {url}: {trans.get('error')}")
    en_text = trans["translated_text"]

    validation = validate_scraped_text(en_text, COUNTRY, url)
    if not validation.passed:
        logger.warning(f"Validation failed for {url}: {validation.warnings}; aborting this source.")
        raise RuntimeError(f"Validation failed: {validation.warnings}")

    chunks = chunk_regulatory_text(
        text=en_text,
        country=COUNTRY,
        regulation_name=document_title,
        device_classes=DEVICE_CLASSES_UA,
        source_url=url,
        language="en",
        original_language="uk",
    )
    document_id = f"UA-RES-{resolution_num}"
    enriched = [_enrich_ua_chunk(c, document_id, document_title) for c in chunks]
    store = get_vector_store()
    n = embed_and_index_chunks(enriched, vector_store=store)
    store.save()
    logger.info(f"Indexed {n} chunks for {document_id}")
    return n


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Ukraine Resolutions 753, 754, 755")
    parser.add_argument("--file", default="", help="Single PDF/file to ingest (e.g. ../ukraine-regs/resolution-753.pdf)")
    parser.add_argument("--document-id", default="", help="e.g. UA-RES-753")
    parser.add_argument("--language", default="uk", help="Source language (uk for Ukrainian)")
    parser.add_argument("--regulation-name", default="", help="Optional title (default from document-id)")
    parser.add_argument("--file-dir", default=os.environ.get("UA_HTML_DIR", ""), help="Directory with 753.html, 754.html, 755.html")
    parser.add_argument("--skip-audit", action="store_true", help="Do not run chunk_audit at end (use when ingesting one file at a time)")
    args = parser.parse_args()

    if args.file.strip():
        # Single-file mode: ingest one PDF, then seed CFS, no audit (user runs chunk_audit separately)
        try:
            ingest_one_from_file(
                file_path=args.file.strip(),
                document_id=args.document_id.strip() or "UA-RES-753",
                language=args.language.strip() or "uk",
                regulation_name=args.regulation_name.strip() or None,
            )
            get_vector_store().save()
        except Exception as e:
            logger.exception(e)
            return 1
        return 0

    file_dir = args.file_dir.strip() or None
    store = get_vector_store()
    total = 0
    for num, url, title in UA_RESOLUTIONS:
        try:
            n = ingest_one_resolution(num, url, title, file_dir=file_dir)
            total += n
        except Exception as e:
            logger.exception(e)
            return 1

    store.save()

    if args.skip_audit:
        return 0

    report = audit_chunks(COUNTRY, sample_size=10)
    logger.info(f"Audit UA: {report.passed}/{report.sampled} passed (total {report.total_chunks} chunks)")
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        logger.info(f"  [{status}] {r.chunk_id} | {r.regulation_name} | {r.article} | {r.issues or []}")

    if report.failed > 0:
        logger.error(f"Chunk audit failed: {report.failed} samples failed. Do not proceed to A3.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
