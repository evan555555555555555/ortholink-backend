#!/usr/bin/env python3
"""
OrthoLink — Priority Accuracy Ingestion v2 (PDF + HTML direct extraction)

Handles documents that the standard scraper cannot process:
  - PDFs:  uses pdfplumber / fitz (PyMuPDF) for text extraction
  - HTML:  uses httpx + BeautifulSoup with browser headers + generous timeouts

Bypasses validate_scraped_text() (designed for scraped HTML noise, not clean PDF extracts).
Directly calls chunk_regulatory_text() → embed_and_index_chunks().

Documents:
  1. FDA benefit-risk / ISO 14971 guidance (PDF)
  2. FDA CDS guidance Jan 2026 (PDF — tries multiple FDA URLs)
  3. Health Canada Class II-IV licensing guidance (HTML)
  4. CDSCO India AI/ML SaMD guidance (PDF)
  5. IMDRF SaMD Key Definitions (PDF)
  6. IMDRF N18 Labelling Principles (PDF)
  7. EU MEDDEV 2.7.1 Rev 4 Clinical Evaluation (PDF)
  8. EU MDCG 2021-24 PMCF guidance (PDF)
  9. WHO Essential Medicines and Medical Devices (HTML)

Run from backend/:
    source .venv/bin/activate
    python scripts/ingest_priority_v2.py
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ingest_v2")

# ─────────────────────────────────────────────────────────────────────────────
# Priority documents v2
# ─────────────────────────────────────────────────────────────────────────────

DOCS = [
    {
        "country": "US",
        "document_id": "US_FDA_BENEFIT_RISK_GUIDANCE",
        "regulation_name": "FDA Guidance — Factors to Consider When Making Benefit-Risk Determinations in PMA and De Novo Requests",
        "url": "https://www.fda.gov/media/99769/download",
        "url_type": "pdf",
        "device_classes": ["II", "III"],
        "language": "en",
    },
    {
        "country": "US",
        "document_id": "US_FDA_CDS_GUIDANCE_2024",
        "regulation_name": "FDA Guidance — Clinical Decision Support Software (September 2022)",
        "url": "https://www.fda.gov/media/109618/download",
        "url_type": "pdf",
        "device_classes": ["I", "II", "III"],
        "language": "en",
    },
    {
        "country": "CA",
        "document_id": "CA_HEALTH_CANADA_CLASS_IIIIV_GUIDANCE",
        "regulation_name": "Health Canada — Guidance on Medical Device Licensing (Class II, III, IV)",
        "url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents/guidance-document-medical-device-licensing.html",
        "url_type": "html",
        "device_classes": ["II", "III", "IV"],
        "language": "en",
    },
    {
        "country": "IN",
        "document_id": "IN_CDSCO_AIML_SAMD_GUIDANCE",
        "regulation_name": "CDSCO India — Guidance on AI/ML-Based Software as a Medical Device (SaMD) 2023",
        "url": "https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/Medical-Device-Diagnostics/Guidance-document-on-artificial-intelligence-machine-learning-based-medical-devices-FINAL-CLEAN.pdf",
        "url_type": "pdf",
        "device_classes": ["A", "B", "C", "D"],
        "language": "en",
    },
    {
        "country": "STANDARDS",
        "document_id": "IMDRF_SAMD_FRAMEWORK",
        "regulation_name": "IMDRF — Software as a Medical Device (SaMD): Key Definitions and Framework",
        "url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-131209-samd-key-definitions-140901.pdf",
        "url_type": "pdf",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "STANDARDS",
        "document_id": "IMDRF_LABELLING_N18",
        "regulation_name": "IMDRF N18 FINAL — Principles of Labelling for Medical Devices and IVDs",
        "url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-121102-labelling-n18.pdf",
        "url_type": "pdf",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "EU",
        "document_id": "EU_MEDDEV_271_REV4",
        "regulation_name": "EU MEDDEV 2.7.1 Rev 4 — Clinical Evaluation: Guide for Manufacturers and Notified Bodies",
        "url": "https://ec.europa.eu/docsroom/documents/17522/attachments/1/translations/en/renditions/native",
        "url_type": "pdf",
        "device_classes": ["IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "EU",
        "document_id": "EU_MDCG_2021_24_PMCF",
        "regulation_name": "EU MDCG 2021-24 — Guidance on Post-Market Clinical Follow-up Studies (PMCF)",
        "url": "https://health.ec.europa.eu/system/files/2021-11/mdcg_2021-24_en_0.pdf",
        "url_type": "pdf",
        "device_classes": ["IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "STANDARDS",
        "document_id": "IMDRF_RISK_MANAGEMENT_N47",
        "regulation_name": "IMDRF N47 — Application of Risk Management Principles and Activities for Connected Medical Devices",
        "url": "https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-200318-n47-risk-management-connected-medical-devices.pdf",
        "url_type": "pdf",
        "device_classes": ["I", "IIa", "IIb", "III"],
        "language": "en",
    },
    {
        "country": "CA",
        "document_id": "CA_HEALTH_CANADA_SOR98_AMEND",
        "regulation_name": "Health Canada — Medical Devices Regulations SOR/98-282 Consolidated (Canada.ca)",
        "url": "https://laws-lois.justice.gc.ca/PDF/SOR-98-282.pdf",
        "url_type": "pdf",
        "device_classes": ["I", "II", "III", "IV"],
        "language": "en",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_pdf_text(url: str, timeout: int = 60) -> Optional[str]:
    """Download a PDF and extract full text using pdfplumber (falls back to fitz)."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
    }
    logger.info("Downloading PDF: %s", url)
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    if "html" in content_type:
        # Not a real PDF — maybe a redirect page
        logger.warning("Got HTML instead of PDF from %s", url)
        return None

    pdf_bytes = resp.content
    logger.info("Downloaded %d bytes", len(pdf_bytes))

    # Try pdfplumber first
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        text = "\n\n".join(text_parts)
        if len(text) > 200:
            logger.info("pdfplumber extracted %d chars from %d pages", len(text), len(pdf.pages))
            return text
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)

    # Fallback: fitz (PyMuPDF)
    try:
        import fitz  # noqa
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        text = "\n\n".join(text_parts)
        if len(text) > 200:
            logger.info("fitz extracted %d chars", len(text))
            return text
    except Exception as e:
        logger.warning("fitz failed: %s", e)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_html_text(url: str, timeout: int = 45) -> Optional[str]:
    """Fetch an HTML page and extract clean text using BeautifulSoup."""
    import httpx
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }
    logger.info("Fetching HTML: %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav, header, footer, scripts
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # Prefer main content areas
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find(class_="main-content")
        or soup.find(class_="page-content")
        or soup.body
    )

    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    logger.info("Extracted %d chars from HTML", len(text))
    return text if len(text) > 300 else None


# ─────────────────────────────────────────────────────────────────────────────
# Core ingest (no scraper validation)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_direct(
    text: str,
    country: str,
    regulation_name: str,
    document_id: str,
    source_url: str,
    device_classes: list[str],
    language: str = "en",
) -> dict:
    """
    Ingest pre-extracted text directly into FAISS.
    Bypasses scraper and validator — use only for trusted PDF/HTML extracts.
    """
    from app.ingestion.chunker import chunk_regulatory_text
    from app.ingestion.embedder import embed_and_index_chunks
    from app.tools.vector_store import get_vector_store

    logger.info("Chunking %d chars for %s/%s", len(text), country, document_id)
    chunks = chunk_regulatory_text(
        text=text,
        country=country,
        regulation_name=regulation_name,
        device_classes=device_classes,
        source_url=source_url,
        language=language,
        original_language=None,
        document_id=document_id,
    )
    logger.info("Created %d chunks", len(chunks))
    if not chunks:
        raise ValueError("chunker produced 0 chunks — text may be too short")

    store = get_vector_store()
    embedded = embed_and_index_chunks(chunks, vector_store=store)
    logger.info("Embedded %d chunks", embedded)

    # CFS synthetic seeding removed — all chunks must come from real regulatory sources.

    return {
        "country": country,
        "document_id": document_id,
        "regulation_name": regulation_name,
        "chunks_created": len(chunks),
        "chunks_embedded": embedded,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent
    metadata_path = base_dir / "data" / "embeddings" / "metadata.json"

    # Load existing doc IDs
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        for m in meta:
            did = m.get("document_id")
            if did:
                existing_ids.add(did)
        logger.info("Store: %d chunks, %d unique doc IDs", len(meta), len(existing_ids))

    to_ingest = [d for d in DOCS if d["document_id"] not in existing_ids]
    if not to_ingest:
        logger.info("All v2 docs already in store.")
        _print_coverage()
        return

    logger.info("Will ingest %d documents:", len(to_ingest))
    for d in to_ingest:
        logger.info("  [%s] %s", d["country"], d["document_id"])

    results = []
    for i, doc in enumerate(to_ingest, 1):
        logger.info("")
        logger.info("=" * 65)
        logger.info("[%d/%d] %s — %s", i, len(to_ingest), doc["country"], doc["document_id"])
        logger.info("=" * 65)

        try:
            # 1. Extract text
            if doc["url_type"] == "pdf":
                text = _fetch_pdf_text(doc["url"])
            else:
                text = _fetch_html_text(doc["url"])

            if not text or len(text.split()) < 200:
                raise ValueError(
                    f"Insufficient text extracted: {len(text.split()) if text else 0} words"
                )

            # 2. Ingest directly
            result = ingest_direct(
                text=text,
                country=doc["country"],
                regulation_name=doc["regulation_name"],
                document_id=doc["document_id"],
                source_url=doc["url"],
                device_classes=doc.get("device_classes", []),
                language=doc.get("language", "en"),
            )
            logger.info("  ✓ %d chunks embedded", result["chunks_embedded"])
            results.append({"status": "success", **result})

        except Exception as e:
            logger.error("  ✗ FAILED: %s", e, exc_info=False)
            results.append({
                "status": "failed",
                "document_id": doc["document_id"],
                "country": doc["country"],
                "error": str(e),
            })

        if i < len(to_ingest):
            time.sleep(1)

    # Summary
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_new = sum(r.get("chunks_embedded", 0) for r in success)

    print()
    print("=" * 65)
    print("PRIORITY v2 INGESTION SUMMARY")
    print("=" * 65)
    print(f"\n✓ Succeeded: {len(success)} documents, {total_new} new chunks")
    for r in success:
        print(f"  [{r['country']}] {r['document_id']}: {r.get('chunks_embedded', '?')} chunks")
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for r in failed:
            print(f"  [{r['country']}] {r['document_id']}: {r.get('error', '?')[:100]}")

    _print_coverage()


def _print_coverage():
    try:
        from app.tools.vector_store import get_vector_store
        from collections import Counter

        store = get_vector_store()
        per_country = Counter(
            m.country.upper() for m in store.metadata if getattr(m, "is_active", True)
        )
        total = sum(per_country.values())
        print()
        print(f"VECTOR STORE: {total} total chunks")
        for c in sorted(per_country):
            print(f"  {c:12s}: {per_country[c]:6d}")
    except Exception as e:
        logger.warning("Coverage summary: %s", e)


if __name__ == "__main__":
    main()
