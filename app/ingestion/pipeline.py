"""
OrthoLink Ingestion Pipeline
Orchestrates end-to-end ingestion from all global regulatory sources into FAISS.

Sources:
  - 8 registry adapters (GUDID, EUDAMED, Swissdamed, ANVISA, ARTG, MDALL, SUGAM, GMDN)
  - 5 enforcement scrapers (FDA WL, TGA, EUDAMED FSN, HC Incidents, Market Surveillance)
  - Existing monitored_docs scraper (RAA source documents)

Stages per source:
  1. Fetch raw data (parallel via asyncio.gather)
  2. Normalize (via DocumentNormalizer)
  3. Chunk with legal hierarchy (Article/Section/Clause — HC-3)
  4. Embed with text-embedding-3-large (HC-1)
  5. Upsert to FAISS with metadata
  6. Log to audit_logger

Pipeline entry points:
  run_full_ingestion()              — all sources
  run_country_ingestion(country)    — one country's monitored docs + registry
  run_enforcement_ingestion()       — all enforcement scrapers
  run_registry_ingestion(countries) — registry adapters only

All methods return an IngestionReport with per-source results and totals.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional adapter / scraper imports — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

try:
    from app.adapters.gudid_adapter import GUDIDAdapter
except ImportError:
    GUDIDAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.eudamed_adapter import EUDAMEDAdapter
except ImportError:
    EUDAMEDAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.swissdamed_adapter import SwissdamedAdapter
except ImportError:
    SwissdamedAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.anvisa_adapter import ANVISAAdapter
except ImportError:
    ANVISAAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.artg_adapter import ARTGAdapter
except ImportError:
    ARTGAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.mdall_adapter import MDALLAdapter
except ImportError:
    MDALLAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.sugam_adapter import SUGAMAdapter
except ImportError:
    SUGAMAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.gmdn_adapter import GMDNAdapter
except ImportError:
    GMDNAdapter = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.fda_warning_letters import FDAWarningLettersScraper
except ImportError:
    FDAWarningLettersScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.tga_alerts import TGASafetyAlertsScraper
except ImportError:
    TGASafetyAlertsScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.eudamed_fsn import EUDAMEDFSNScraper
except ImportError:
    EUDAMEDFSNScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.hc_incidents import HCIncidentsScraper
except ImportError:
    HCIncidentsScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.market_surveillance import MarketSurveillanceScraper
except ImportError:
    MarketSurveillanceScraper = None  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class SourceResult(BaseModel):
    """Per-source ingestion result."""
    source_name: str
    country: str
    documents_fetched: int = 0
    chunks_created: int = 0
    errors: list[str] = Field(default_factory=list)
    status: str = "ok"  # ok | partial | failed | skipped


class IngestionReport(BaseModel):
    """Top-level report returned by all pipeline entry points."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_documents: int = 0
    total_chunks: int = 0
    sources_processed: int = 0
    source_results: list[SourceResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry adapter → country mapping
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY_MAP: dict[str, tuple[str, Any]] = {}


def _build_registry_map() -> dict[str, tuple[str, Any]]:
    m: dict[str, tuple[str, Any]] = {}
    if GUDIDAdapter is not None:
        m["US"] = ("GUDID", GUDIDAdapter)
    if EUDAMEDAdapter is not None:
        m["EU"] = ("EUDAMED", EUDAMEDAdapter)
    if SwissdamedAdapter is not None:
        m["CH"] = ("Swissdamed", SwissdamedAdapter)
    if ANVISAAdapter is not None:
        m["BR"] = ("ANVISA", ANVISAAdapter)
    if ARTGAdapter is not None:
        m["AU"] = ("ARTG", ARTGAdapter)
    if MDALLAdapter is not None:
        m["CA"] = ("MDALL", MDALLAdapter)
    if SUGAMAdapter is not None:
        m["IN"] = ("SUGAM", SUGAMAdapter)
    return m


_REGISTRY_MAP = _build_registry_map()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_audit(event: str, details: dict[str, Any]) -> None:
    """Write to audit_logger if available; fall back to structured INFO log."""
    try:
        from app.services.audit_logger import log_event
        log_event(event=event, details=details)
    except ImportError:
        logger.info("INGESTION_AUDIT | event=%s | %s", event, details)
    except Exception as exc:
        logger.warning("audit_logger.log_event failed: %s", exc)


def _adapter_fetch_text(adapter: Any, country: str, batch_size: int) -> tuple[list[str], list[dict]]:
    """
    Call adapter.get_documents() or adapter.search(query='', limit=batch_size).
    Returns (texts, metadatas).  Never raises.
    """
    texts: list[str] = []
    metas: list[dict] = []
    try:
        if hasattr(adapter, "get_documents"):
            docs = adapter.get_documents(limit=batch_size)
        elif hasattr(adapter, "search"):
            docs = adapter.search(query="", limit=batch_size)
        else:
            return texts, metas

        for doc in (docs or []):
            if hasattr(doc, "model_dump"):
                d = doc.model_dump()
            elif isinstance(doc, dict):
                d = doc
            else:
                d = {"text": str(doc)}

            text = d.get("text") or d.get("description") or d.get("body") or str(d)
            if text and len(text.strip()) > 50:
                texts.append(text)
                metas.append({"country": country, "source": "registry", **d})
    except Exception as exc:
        logger.warning("Adapter fetch failed for %s: %s", country, exc)
    return texts, metas


def _scraper_fetch_text(scraper_cls: Any, source_name: str, country: str) -> tuple[list[str], list[dict]]:
    """
    Instantiate scraper_cls, call get_recent(days=30) or get_all().
    Returns (texts, metadatas).  Never raises.
    """
    texts: list[str] = []
    metas: list[dict] = []

    if scraper_cls is None:
        return texts, metas

    try:
        scraper = scraper_cls()
        fn = getattr(scraper, "get_recent", None) or getattr(scraper, "get_all", None)
        if fn is None:
            return texts, metas

        try:
            docs = fn(days=30)
        except TypeError:
            docs = fn()

        for doc in (docs or []):
            if hasattr(doc, "model_dump"):
                d = doc.model_dump()
            elif isinstance(doc, dict):
                d = doc
            else:
                d = {"text": str(doc)}

            # Prefer a structured text representation
            text_parts = []
            for key in ("summary", "description", "text", "body", "content", "detail"):
                v = d.get(key)
                if v and isinstance(v, str) and len(v.strip()) > 10:
                    text_parts.append(v.strip())
            text = "\n\n".join(text_parts) if text_parts else str(d)

            if len(text.strip()) > 50:
                texts.append(text)
                metas.append({"country": country, "source": source_name, **d})
    except Exception as exc:
        logger.warning("Scraper fetch failed (%s): %s", source_name, exc)

    return texts, metas


def _ingest_texts(
    texts: list[str],
    metas: list[dict],
    source_name: str,
    country: str,
    regulation_name: str,
) -> SourceResult:
    """
    Given a list of raw texts and their metadata:
    1. Normalize via DocumentNormalizer
    2. Chunk with chunker (HC-3 hierarchy)
    3. Embed + upsert to FAISS (HC-1)
    Returns a SourceResult.
    """
    errors: list[str] = []
    chunks_created = 0

    if not texts:
        return SourceResult(
            source_name=source_name,
            country=country,
            documents_fetched=0,
            chunks_created=0,
            status="skipped",
        )

    # Step 1 — Normalize
    try:
        from app.ingestion.normalizer import DocumentNormalizer
        normalizer = DocumentNormalizer()
    except ImportError:
        normalizer = None

    all_chunks = []

    for i, (text, meta) in enumerate(zip(texts, metas)):
        try:
            # Normalize if available
            if normalizer is not None:
                try:
                    normalized = normalizer.normalize(
                        raw=text,
                        source_type=meta.get("source", "plain"),
                        metadata=meta,
                    )
                    body = normalized.body if hasattr(normalized, "body") else text
                except Exception as norm_exc:
                    logger.debug("Normalizer failed for item %d of %s: %s", i, source_name, norm_exc)
                    body = text
            else:
                body = text

            # Step 2 — Chunk
            from app.ingestion.chunker import chunk_regulatory_text
            doc_id = meta.get("document_id") or meta.get("action_id") or f"{source_name}-{i}"
            source_url = meta.get("source_url") or meta.get("url")

            item_chunks = chunk_regulatory_text(
                text=body,
                country=country,
                regulation_name=regulation_name,
                source_url=source_url,
                document_id=doc_id,
            )
            all_chunks.extend(item_chunks)
        except Exception as chunk_exc:
            errors.append(f"Chunking error for {source_name} item {i}: {chunk_exc}")
            logger.warning("Chunking error for %s item %d: %s", source_name, i, chunk_exc)

    # Step 3 — Embed + upsert
    if all_chunks:
        try:
            from app.ingestion.embedder import embed_and_index_chunks
            embedded = embed_and_index_chunks(all_chunks)
            chunks_created += embedded
        except Exception as embed_exc:
            errors.append(f"Embedding error for {source_name}: {embed_exc}")
            logger.error("Embedding failed for %s: %s", source_name, embed_exc, exc_info=True)

    status = "ok" if not errors else ("partial" if chunks_created > 0 else "failed")

    return SourceResult(
        source_name=source_name,
        country=country,
        documents_fetched=len(texts),
        chunks_created=chunks_created,
        errors=errors[:10],  # Cap errors in report
        status=status,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates ingestion from all global regulatory sources into FAISS.

    Sources:
    - 8 registry adapters (GUDID, EUDAMED, Swissdamed, ANVISA, ARTG, MDALL, SUGAM, GMDN)
    - 5 enforcement scrapers (FDA WL, TGA, EUDAMED FSN, HC Incidents, Market Surveillance)
    - Existing monitored_docs scraper

    Stages per source:
    1. Fetch raw data (parallel via asyncio.gather)
    2. Normalize (via DocumentNormalizer)
    3. Chunk with legal hierarchy (HC-3)
    4. Embed with text-embedding-3-large (HC-1)
    5. Upsert to FAISS with metadata
    6. Log to audit_logger
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    # ── Status ────────────────────────────────────────────────────────────────

    def get_ingestion_status(self) -> dict[str, Any]:
        """
        Return current pipeline status: available adapters, scrapers, and
        the latest FAISS index stats.
        """
        available_registries = {
            cc: name for cc, (name, _) in _REGISTRY_MAP.items()
        }

        available_scrapers: dict[str, bool] = {
            "FDA Warning Letters": FDAWarningLettersScraper is not None,
            "TGA Safety Alerts": TGASafetyAlertsScraper is not None,
            "EUDAMED FSN": EUDAMEDFSNScraper is not None,
            "Health Canada Incidents": HCIncidentsScraper is not None,
            "Market Surveillance": MarketSurveillanceScraper is not None,
        }

        faiss_stats: dict[str, Any] = {}
        try:
            from app.tools.vector_store import get_vector_store
            vs = get_vector_store()
            faiss_stats = {
                "total_chunks": len(vs.metadata),
                "active_chunks": sum(1 for m in vs.metadata if getattr(m, "is_active", True)),
                "countries": sorted({m.country for m in vs.metadata if hasattr(m, "country")}),
            }
        except Exception as exc:
            faiss_stats = {"error": str(exc)}

        return {
            "available_registries": available_registries,
            "available_scrapers": available_scrapers,
            "faiss": faiss_stats,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Monitored docs ingestion ──────────────────────────────────────────────

    async def _ingest_monitored_docs_country(self, country: str) -> SourceResult:
        """
        Fetch all monitored docs for a country via the existing scraper,
        then normalize → chunk → embed.
        """
        errors: list[str] = []
        chunks_created = 0
        docs_fetched = 0

        try:
            from app.ingestion.monitored_docs import get_monitored_docs
            from app.ingestion.scraper import scrape_url, validate_scraped_text

            docs = get_monitored_docs(country)
            if not docs:
                return SourceResult(
                    source_name=f"MonitoredDocs:{country}",
                    country=country,
                    status="skipped",
                )

            for doc in docs:
                try:
                    result = await asyncio.to_thread(scrape_url, doc["source_url"])
                    if not result.success or not result.text:
                        errors.append(f"Scrape failed: {doc['document_id']} — {result.error}")
                        continue

                    validation = validate_scraped_text(result.text, country, doc["source_url"])
                    if not validation.passed:
                        logger.warning(
                            "Monitored doc %s failed validation: %s",
                            doc["document_id"], validation.warnings,
                        )

                    source_result = await asyncio.to_thread(
                        _ingest_texts,
                        [result.text],
                        [{
                            "country": country,
                            "source": "monitored_docs",
                            "document_id": doc["document_id"],
                            "source_url": doc["source_url"],
                        }],
                        f"MonitoredDocs:{country}",
                        country,
                        doc["regulation_name"],
                    )
                    docs_fetched += 1
                    chunks_created += source_result.chunks_created
                    errors.extend(source_result.errors)
                except Exception as exc:
                    errors.append(f"Error ingesting {doc.get('document_id', '?')}: {exc}")
                    logger.warning("MonitoredDocs ingest error for %s/%s: %s", country, doc.get("document_id"), exc)

        except Exception as exc:
            errors.append(f"MonitoredDocs country sweep failed: {exc}")
            logger.error("MonitoredDocs country sweep failed (%s): %s", country, exc, exc_info=True)

        status = "ok" if not errors else ("partial" if chunks_created > 0 else "failed")
        return SourceResult(
            source_name=f"MonitoredDocs:{country}",
            country=country,
            documents_fetched=docs_fetched,
            chunks_created=chunks_created,
            errors=errors[:10],
            status=status,
        )

    # ── Registry ingestion ────────────────────────────────────────────────────

    async def _ingest_registry(self, country: str) -> SourceResult:
        """Ingest from a single country's registry adapter."""
        entry = _REGISTRY_MAP.get(country.upper())
        if entry is None:
            return SourceResult(
                source_name=f"Registry:{country}",
                country=country,
                status="skipped",
                errors=[f"No registry adapter available for {country}"],
            )

        registry_name, adapter_cls = entry
        try:
            adapter = adapter_cls()
            texts, metas = await asyncio.to_thread(
                _adapter_fetch_text, adapter, country, self.batch_size
            )
            return await asyncio.to_thread(
                _ingest_texts,
                texts,
                metas,
                registry_name,
                country,
                f"{registry_name} Device Registry",
            )
        except Exception as exc:
            logger.error("Registry ingestion failed for %s: %s", country, exc, exc_info=True)
            return SourceResult(
                source_name=registry_name,
                country=country,
                status="failed",
                errors=[str(exc)],
            )

    # ── Enforcement scraper ingestion ─────────────────────────────────────────

    async def _ingest_scraper(
        self,
        scraper_cls: Any,
        source_name: str,
        country: str,
        regulation_name: str,
    ) -> SourceResult:
        """Ingest from a single enforcement scraper."""
        if scraper_cls is None:
            return SourceResult(
                source_name=source_name,
                country=country,
                status="skipped",
                errors=[f"{source_name} scraper not installed"],
            )
        try:
            texts, metas = await asyncio.to_thread(
                _scraper_fetch_text, scraper_cls, source_name, country
            )
            return await asyncio.to_thread(
                _ingest_texts, texts, metas, source_name, country, regulation_name
            )
        except Exception as exc:
            logger.error("Scraper ingestion failed (%s): %s", source_name, exc, exc_info=True)
            return SourceResult(
                source_name=source_name,
                country=country,
                status="failed",
                errors=[str(exc)],
            )

    # ── Public entry points ───────────────────────────────────────────────────

    async def run_full_ingestion(self) -> IngestionReport:
        """
        Run full ingestion from all sources:
        - All monitored regulatory documents (15 countries)
        - All registry adapters
        - All enforcement scrapers
        """
        start = time.monotonic()
        logger.info("IngestionPipeline: starting full ingestion run")

        tasks: list[asyncio.Task] = []

        # Monitored docs for all countries
        try:
            from app.ingestion.monitored_docs import list_all_countries
            all_countries = list_all_countries()
        except Exception:
            all_countries = ["US", "EU", "UK", "UA", "IN", "CA", "AU", "JP",
                             "CN", "BR", "KR", "CH", "MX", "RU", "SA"]

        for country in all_countries:
            tasks.append(asyncio.create_task(
                self._ingest_monitored_docs_country(country)
            ))

        # Registry adapters
        for country in _REGISTRY_MAP:
            tasks.append(asyncio.create_task(self._ingest_registry(country)))

        # Enforcement scrapers
        enforcement_sources = [
            (FDAWarningLettersScraper, "FDA Warning Letters", "US",
             "FDA Warning Letters — Enforcement Actions"),
            (TGASafetyAlertsScraper, "TGA Safety Alerts", "AU",
             "TGA Safety Alerts — Therapeutic Goods Administration"),
            (EUDAMEDFSNScraper, "EUDAMED FSN", "EU",
             "EUDAMED Field Safety Notices"),
            (HCIncidentsScraper, "HC Incidents", "CA",
             "Health Canada Medical Device Incident Reports"),
            (MarketSurveillanceScraper, "Market Surveillance", "EU",
             "EUDAMED Market Surveillance Actions"),
        ]
        for scraper_cls, source_name, country, reg_name in enforcement_sources:
            tasks.append(asyncio.create_task(
                self._ingest_scraper(scraper_cls, source_name, country, reg_name)
            ))

        results: list[SourceResult] = list(await asyncio.gather(*tasks))

        return self._build_report(results, time.monotonic() - start)

    async def run_country_ingestion(self, country: str) -> IngestionReport:
        """
        Run ingestion for a single country:
        - Monitored regulatory documents for that country
        - Registry adapter for that country (if available)
        """
        start = time.monotonic()
        country = country.strip().upper()
        logger.info("IngestionPipeline: country ingestion for %s", country)

        tasks = [
            asyncio.create_task(self._ingest_monitored_docs_country(country)),
            asyncio.create_task(self._ingest_registry(country)),
        ]
        results: list[SourceResult] = list(await asyncio.gather(*tasks))
        return self._build_report(results, time.monotonic() - start)

    async def run_enforcement_ingestion(self) -> IngestionReport:
        """
        Run ingestion from all enforcement scrapers only.
        """
        start = time.monotonic()
        logger.info("IngestionPipeline: enforcement ingestion run")

        enforcement_sources = [
            (FDAWarningLettersScraper, "FDA Warning Letters", "US",
             "FDA Warning Letters — Enforcement Actions"),
            (TGASafetyAlertsScraper, "TGA Safety Alerts", "AU",
             "TGA Safety Alerts — Therapeutic Goods Administration"),
            (EUDAMEDFSNScraper, "EUDAMED FSN", "EU",
             "EUDAMED Field Safety Notices"),
            (HCIncidentsScraper, "HC Incidents", "CA",
             "Health Canada Medical Device Incident Reports"),
            (MarketSurveillanceScraper, "Market Surveillance", "EU",
             "EUDAMED Market Surveillance Actions"),
        ]

        tasks = [
            asyncio.create_task(
                self._ingest_scraper(scraper_cls, source_name, country, reg_name)
            )
            for scraper_cls, source_name, country, reg_name in enforcement_sources
        ]
        results: list[SourceResult] = list(await asyncio.gather(*tasks))
        return self._build_report(results, time.monotonic() - start)

    async def run_registry_ingestion(
        self,
        countries: Optional[list[str]] = None,
    ) -> IngestionReport:
        """
        Run ingestion from registry adapters only.
        If countries is None, ingests from all available adapters.
        """
        start = time.monotonic()
        target = [c.upper() for c in countries] if countries else list(_REGISTRY_MAP.keys())
        logger.info("IngestionPipeline: registry ingestion for %s", target)

        tasks = [
            asyncio.create_task(self._ingest_registry(country))
            for country in target
        ]
        results: list[SourceResult] = list(await asyncio.gather(*tasks))
        return self._build_report(results, time.monotonic() - start)

    # ── Report builder ────────────────────────────────────────────────────────

    def _build_report(
        self,
        results: list[SourceResult],
        duration: float,
    ) -> IngestionReport:
        """Aggregate per-source SourceResults into a top-level IngestionReport."""
        total_docs = sum(r.documents_fetched for r in results)
        total_chunks = sum(r.chunks_created for r in results)
        sources_processed = sum(1 for r in results if r.status not in ("skipped",))
        all_errors = [e for r in results for e in r.errors]

        report = IngestionReport(
            total_documents=total_docs,
            total_chunks=total_chunks,
            sources_processed=sources_processed,
            source_results=results,
            errors=all_errors[:50],
            duration_seconds=round(duration, 2),
        )

        _log_audit("ingestion_complete", {
            "run_id": report.run_id,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "sources_processed": sources_processed,
            "error_count": len(all_errors),
            "duration_seconds": report.duration_seconds,
        })

        logger.info(
            "IngestionPipeline complete: run_id=%s docs=%d chunks=%d sources=%d errors=%d duration=%.1fs",
            report.run_id, total_docs, total_chunks, sources_processed,
            len(all_errors), duration,
        )

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (used by scheduler)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: Optional[IngestionPipeline] = None


def get_pipeline(batch_size: int = 100) -> IngestionPipeline:
    """Return the module-level IngestionPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(batch_size=batch_size)
    return _pipeline


def run_priority_ingestion_sync() -> None:
    """
    Synchronous entry point for APScheduler cron.
    Runs enforcement + monitored docs for priority countries (US, EU, UK, AU).
    Does NOT block the calling thread beyond its own execution.
    """
    import asyncio as _asyncio

    async def _main():
        pipeline = get_pipeline()
        # Priority countries
        priority = ["US", "EU", "UK", "AU"]
        try:
            docs_task = pipeline.run_country_ingestion  # type: ignore[assignment]
            tasks = [pipeline._ingest_monitored_docs_country(c) for c in priority]
            tasks += [pipeline._ingest_registry(c) for c in priority]
            enf_task = pipeline.run_enforcement_ingestion()
            results_docs = await _asyncio.gather(*tasks)
            result_enf = await enf_task
            logger.info(
                "Priority ingestion complete: docs=%d chunks=%d enf_chunks=%d",
                sum(r.documents_fetched for r in results_docs),
                sum(r.chunks_created for r in results_docs),
                result_enf.total_chunks,
            )
        except Exception as exc:
            logger.error("Priority ingestion cron failed: %s", exc, exc_info=True)

    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            # APScheduler runs in a thread; create a new event loop
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_asyncio.run, _main())
                future.result(timeout=3600)
        else:
            loop.run_until_complete(_main())
    except Exception as exc:
        logger.error("run_priority_ingestion_sync failed: %s", exc, exc_info=True)
