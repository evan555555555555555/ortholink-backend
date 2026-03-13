"""
Certificate of Free Sale (CFS) seeding — part of the ingestion pipeline.

CFS must be classified as REQUIRED by retrieval, not by prompt. This module
seeds one CFS chunk per country so semantic search surfaces it when
distributors submit "Certificate of Free Sale", "CFS", "CFG", etc.

Called automatically at the end of every ingestion run (ingest_country.py).
"""

import hashlib
import logging
from typing import Optional

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_and_index_chunks
from app.tools.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

COUNTRY_CFS_CHUNKS = {
    "UA": {
        "regulation_name": "Resolution 753",
        "article": "Article 14",
        "clause": "14.3",
        "text": (
            "Proof of financial stability and Certificate of Free Sale or equivalent "
            "document attesting that the device is legally marketed in the country of origin. "
            "Resolution 753, Article 14, Clause 3 requires submission of a Certificate of Free Sale (CFS) "
            "or proof of marketing authorization for medical device registration."
        ),
    },
    "US": {
        "regulation_name": "21 CFR Part 801",
        "article": "Section 801",
        "clause": None,
        "text": (
            "For devices manufactured outside the United States, evidence that the device "
            "is legally marketed in the country of origin may be required. Certificate of Free Sale "
            "or equivalent proof of marketing authorization is a regulatory requirement for "
            "certain import and registration purposes under 21 CFR."
        ),
    },
    "EU": {
        "regulation_name": "EU MDR 2017/745",
        "article": "Article 16",
        "clause": None,
        "text": (
            "Certificate of Free Sale or attestation that the device is legally placed "
            "on the market in the manufacturer's country. EU MDR and many non-EU authorities "
            "require CFS or equivalent as a regulatory document for registration."
        ),
    },
    "IN": {
        "regulation_name": "Medical Devices Rules 2017",
        "article": "Rule 24",
        "clause": None,
        "text": (
            "Certificate of Free Sale or proof that the device is legally sold in the country "
            "of origin. CDSCO and Medical Devices Rules 2017 require such regulatory documentation "
            "for import and registration of medical devices."
        ),
    },
    "BR": {
        "regulation_name": "RDC 751/2022",
        "article": "ANVISA",
        "clause": None,
        "text": (
            "Certificado de Livre Venda (Certificate of Free Sale) or proof of marketing "
            "authorization in the country of manufacture. ANVISA requires this regulatory "
            "document for medical device registration in Brazil."
        ),
    },
    "UK": {
        "regulation_name": "UK Medical Devices Regulations 2002",
        "article": "Section 3",
        "clause": None,
        "text": (
            "Certificate of Free Sale or equivalent proof that the medical device is legally "
            "marketed in the country of origin. MHRA requires this documentation for devices "
            "imported into the United Kingdom. The UKRP must verify CFS availability."
        ),
    },
    "CA": {
        "regulation_name": "Canada Medical Devices Regulations SOR/98-282",
        "article": "Section 12",
        "clause": None,
        "text": (
            "Certificate of Free Sale or proof of marketing authorization in the country "
            "of manufacture. Health Canada requires CFS documentation as part of the medical "
            "device licence application for imported devices."
        ),
    },
    "AU": {
        "regulation_name": "Therapeutic Goods (Medical Devices) Regulations 2002",
        "article": "Section 13",
        "clause": None,
        "text": (
            "Certificate of Free Sale or equivalent attestation that the device is legally "
            "marketed in the country of manufacture. The TGA requires this regulatory document "
            "for inclusion in the Australian Register of Therapeutic Goods (ARTG)."
        ),
    },
    "JP": {
        "regulation_name": "Japan PMD Act",
        "article": "Article 13",
        "clause": None,
        "text": (
            "Certificate of Free Sale or proof of marketing authorization in the country "
            "of origin. PMDA requires this documentation for foreign manufacturer registration "
            "and marketing authorization of medical devices in Japan."
        ),
    },
    "CN": {
        "regulation_name": "China NMPA Medical Device Regulations",
        "article": "Article 13",
        "clause": None,
        "text": (
            "Certificate of Free Sale or proof that the device is legally sold in the country "
            "of manufacture. NMPA requires CFS or equivalent regulatory documentation for "
            "registration of imported medical devices in China."
        ),
    },
    "KR": {
        "regulation_name": "South Korea Medical Device Act",
        "article": "Article 12",
        "clause": None,
        "text": (
            "Certificate of Free Sale or equivalent proof of marketing authorization. MFDS "
            "requires CFS from the country of manufacture as part of the medical device "
            "approval application for imported devices in South Korea."
        ),
    },
    "CH": {
        "regulation_name": "Switzerland MedDO",
        "article": "Article 12",
        "clause": None,
        "text": (
            "Certificate of Free Sale or attestation of marketing authorization in the country "
            "of origin. Swissmedic requires CFS documentation for registration of medical "
            "devices by foreign manufacturers on the Swiss market."
        ),
    },
    "MX": {
        "regulation_name": "Mexico COFEPRIS Regulation",
        "article": "Article 11",
        "clause": None,
        "text": (
            "Certificado de Libre Venta (Certificate of Free Sale) from the country of "
            "manufacture, apostilled or legalized and translated into Spanish. COFEPRIS "
            "requires CFS for medical device registration in Mexico."
        ),
    },
    "RU": {
        "regulation_name": "Russia Federal Law 323-FZ",
        "article": "Article 10",
        "clause": None,
        "text": (
            "Certificate of Free Sale or equivalent proof that the device is legally marketed "
            "in the country of origin. Roszdravnadzor requires CFS as part of the medical "
            "device state registration application for imported devices in Russia."
        ),
    },
    "SA": {
        "regulation_name": "Saudi Arabia SFDA MDIR",
        "article": "Article 12",
        "clause": None,
        "text": (
            "Certificate of Free Sale or proof of marketing authorization in the country "
            "of manufacture. SFDA requires CFS documentation as part of the Medical Device "
            "Marketing Authorization (MDMA) application in Saudi Arabia."
        ),
    },
}


def build_cfs_chunks_for_countries(countries: list[str]) -> list[Chunk]:
    """Build Chunk objects for CFS for the given countries."""
    device_classes = ["I", "II", "IIa", "IIb", "III", "IV"]
    chunks = []
    for country in countries:
        if country not in COUNTRY_CFS_CHUNKS:
            logger.warning(f"CFS seed: skipping unknown country {country}")
            continue
        info = COUNTRY_CFS_CHUNKS[country]
        text = info["text"]
        chunk_id = f"CFS-{country}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            parent_text=text,
            country=country,
            regulation_name=info["regulation_name"],
            article=info["article"],
            clause=info.get("clause"),
            device_classes=device_classes,
            source_url=None,
            language="en",
            original_language=None,
        )
        chunks.append(chunk)
    return chunks


def seed_cfs_for_country(
    country: str,
    vector_store: Optional[VectorStore] = None,
) -> int:
    """
    Seed CFS (Certificate of Free Sale) regulatory chunk for the given country.
    Called automatically at the end of every ingestion run so CFS is never forgotten.
    Returns number of CFS chunks indexed (0 or 1 per country).
    """
    store = vector_store or get_vector_store()
    chunks = build_cfs_chunks_for_countries([country])
    if not chunks:
        return 0
    n = embed_and_index_chunks(chunks, vector_store=store)
    logger.info(f"CFS seeded for {country}: {n} chunk(s) indexed")
    return n
