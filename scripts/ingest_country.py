#!/usr/bin/env python3
"""
OrthoLink Country Ingestion CLI

Usage:
    python scripts/ingest_country.py \
        --country UA \
        --regulation-name "Resolution 753" \
        --source-url "https://zakon.rada.gov.ua/laws/show/753-2020" \
        --device-classes IIb,III

    python scripts/ingest_country.py \
        --country UA \
        --regulation-name "Resolution 753" \
        --file data/raw/ukraine_res753.txt \
        --device-classes IIb,III

Pipeline: scrape/load → validate → clean → translate → chunk → embed → audit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ingest")


def ingest(
    country: str,
    regulation_name: str,
    source_url: str = "",
    file_path: str = "",
    device_classes: list[str] | None = None,
    language: str = "",
    skip_translation: bool = False,
    skip_audit: bool = False,
    document_id: str = "",
    embed_batch_size: int = 100,
) -> dict:
    """Run the full ingestion pipeline for a country's regulation."""
    device_classes = device_classes or []

    logger.info(f"Starting ingestion: {regulation_name} ({country})")

    # Step 1: Acquire
    from app.ingestion.scraper import load_from_file, scrape_url

    if file_path:
        logger.info(f"Loading from file: {file_path}")
        result = load_from_file(file_path)
    elif source_url:
        logger.info(f"Scraping URL: {source_url}")
        result = scrape_url(source_url)
    else:
        raise ValueError("Must provide either --source-url or --file")

    if not result.success:
        raise RuntimeError(f"Acquisition failed: {result.error}")

    raw_text = result.text
    logger.info(f"Acquired {len(raw_text)} characters")

    # Step 2: Translate first if non-English (so we validate the text we will chunk)
    if not skip_translation and language and language != "en":
        from app.ingestion.translator import translate_to_english

        logger.info(f"Translating from {language} to English...")
        translation = translate_to_english(raw_text, language, country)
        if translation["success"]:
            raw_text = translation["translated_text"]
            logger.info("Translation complete")
        else:
            logger.warning(f"Translation failed: {translation.get('error')}. Using original text.")

    # Step 3: Validate (scraper hardening — do not ingest garbage; validates translated text for UA)
    from app.ingestion.scraper import validate_scraped_text

    source_ref = source_url or file_path
    validation = validate_scraped_text(raw_text, country, source_ref)
    if not validation.passed:
        logger.error(f"Scrape validation failed: {validation.warnings}")
        raise RuntimeError(f"Scraper validation failed: {validation.warnings}")

    logger.info("Scrape validation passed")

    # Step 4: Chunk
    from app.ingestion.chunker import chunk_regulatory_text

    chunks = chunk_regulatory_text(
        text=raw_text,
        country=country,
        regulation_name=regulation_name,
        device_classes=device_classes,
        source_url=source_url or file_path,
        language="en",
        original_language=language if language != "en" else None,
        document_id=document_id or None,
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Step 5: Embed + Index
    from app.ingestion.embedder import embed_and_index_chunks
    from app.tools.vector_store import get_vector_store

    store = get_vector_store()
    embedded_count = embed_and_index_chunks(chunks, vector_store=store, batch_size=embed_batch_size)
    logger.info(f"Embedded and indexed {embedded_count} chunks")

    # Step 5b: CFS seeding removed — synthetic CFS chunks are no longer created.
    # CFS requirements are now evaluated via FAISS search + LLM semantic evaluation
    # in the DVA pipeline, grounded in real regulatory text.

    # Step 6: Audit
    audit_result = None
    if not skip_audit:
        from app.ingestion.chunk_audit import audit_chunks

        audit_result = audit_chunks(country, sample_size=10)
        logger.info(
            f"Audit: {audit_result.passed}/{audit_result.sampled} passed "
            f"(total {audit_result.total_chunks} chunks for {country})"
        )
        if audit_result.failed > 0:
            logger.warning(f"AUDIT FAILURES: {audit_result.failed} chunks failed validation")

    return {
        "country": country,
        "regulation_name": regulation_name,
        "chunks_created": len(chunks),
        "chunks_embedded": embedded_count,
        "audit_passed": audit_result.passed if audit_result else None,
        "audit_failed": audit_result.failed if audit_result else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest regulatory data for a country")
    parser.add_argument("--country", required=True, help="Country code (e.g., UA, US, IN)")
    parser.add_argument("--regulation-name", required=True, help="Name of regulation")
    parser.add_argument("--source-url", default="", help="URL to scrape")
    parser.add_argument("--file", default="", help="Local file path (alternative to URL)")
    parser.add_argument("--device-classes", default="", help="Comma-separated device classes")
    parser.add_argument("--language", default="en", help="Source language code (e.g., uk, ja)")
    parser.add_argument("--document-id", default="", help="e.g. UA-RES-753, US-21CFR-820")
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")

    args = parser.parse_args()

    device_classes = [c.strip() for c in args.device_classes.split(",") if c.strip()]

    result = ingest(
        country=args.country,
        regulation_name=args.regulation_name,
        source_url=args.source_url,
        file_path=args.file,
        device_classes=device_classes,
        language=args.language,
        document_id=args.document_id,
        skip_translation=args.skip_translation,
        skip_audit=args.skip_audit,
    )

    print(f"\nIngestion complete: {result}")


if __name__ == "__main__":
    main()
