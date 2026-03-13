"""
OrthoLink Vector Store Tool
FAISS search + metadata filtering.
HC-5: Country isolation enforced — no cross-contamination between countries.

FAISS METADATA FILTERING STRATEGY (PRD §6.2):
----------------------------------------------
FAISS does not natively support metadata filters. We use strategy (b): PRE-FILTER.
- Single global FAISS index (one index for all countries) for simplicity.
- On every search: we query FAISS for top_k * 5 (oversample), then POST-FILTER
  in Python by metadata: country (MANDATORY), is_active, device_class.
- Result: Ukraine queries return ZERO FDA chunks; FDA queries return ZERO Ukraine
  chunks. Country isolation is enforced in the loop below (chunk.country.upper() != country.upper() → skip).
- Alternative (not used): separate FAISS index per country would allow smaller
  per-query index but requires N index files and more complex loading.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.config import get_settings
from app.tools.embeddings import embed_text
from app.tools.metadata_db import MetadataDB, json_to_sqlite, DB_FILENAME

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for an embedded regulatory chunk."""

    def __init__(
        self,
        chunk_id: str,
        country: str,
        regulation_name: str,
        article: str,
        clause: Optional[str] = None,
        device_classes: Optional[list[str]] = None,
        text: str = "",
        parent_text: str = "",
        source_url: Optional[str] = None,
        language: str = "en",
        original_language: Optional[str] = None,
        is_active: bool = True,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        chunk_hash: Optional[str] = None,
        document_id: Optional[str] = None,
        section_path: Optional[str] = None,
        regulatory_status: Optional[str] = None,
        superseded_by: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.country = country
        self.regulation_name = regulation_name
        self.article = article
        self.clause = clause
        self.device_classes = device_classes or []
        self.text = text
        self.parent_text = parent_text
        self.source_url = source_url
        self.language = language
        self.original_language = original_language
        self.is_active = is_active
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.chunk_hash = chunk_hash
        self.document_id = document_id
        self.section_path = section_path
        # Fix-2: Regulatory lifecycle status for revoked/superseded law tracking
        # Values: None (active/current), "REVOKED", "SUPERSEDED", "AMENDED"
        self.regulatory_status: Optional[str] = regulatory_status
        # If regulatory_status == "REVOKED"/"SUPERSEDED", cite the replacement law
        self.superseded_by: Optional[str] = superseded_by

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "country": self.country,
            "regulation_name": self.regulation_name,
            "article": self.article,
            "clause": self.clause,
            "device_classes": self.device_classes,
            "text": self.text,
            "parent_text": self.parent_text,
            "source_url": self.source_url,
            "language": self.language,
            "original_language": self.original_language,
            "is_active": self.is_active,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "chunk_hash": self.chunk_hash,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "regulatory_status": self.regulatory_status,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        # Backward compatibility: omit keys not in constructor
        allowed = {"chunk_id", "country", "regulation_name", "article", "clause",
                   "device_classes", "text", "parent_text", "source_url", "language",
                   "original_language", "is_active", "valid_from", "valid_to", "chunk_hash",
                   "document_id", "section_path",
                   "regulatory_status", "superseded_by"}  # Fix-2 fields
        kwargs = {k: v for k, v in data.items() if k in allowed}
        return cls(**kwargs)


def _dynamic_top_k(country: str, base_k: int, device_class: Optional[str] = None) -> int:
    """
    Dynamic k-NN retrieval boundary for expansive regulatory frameworks.

    Countries with large regulatory corpora need higher top_k to avoid
    pulling generic chunks instead of specific schedule/annex content.
    EU: 100+ articles/annexes; US: 21 CFR Part 820 + QMSR.
    IN: MDR 2017 + 16 CDSCO guidance docs (3700+ chunks).
    AU: TGA Act + multiple guidance docs (4500+ chunks).
    JP: PMDA Act + notifications (2900+ chunks).
    KR: MFDS regulations (2200+ chunks).
    """
    c = (country or "").strip().upper()
    # Tier 1: Largest regulatory corpora — need widest retrieval
    if c == "US":
        return max(base_k, 150)
    if c == "EU":
        return max(base_k, 100)
    # Tier 2: Large corpora (>2000 chunks) — need broad retrieval
    # to find specific schedules (IN: Schedule 4, AU: Schedule 3, etc.)
    if c in ("IN", "AU", "JP", "KR"):
        return max(base_k, 60)
    # Tier 3: Medium corpora (500-2000 chunks)
    if c in ("CA", "UK", "MX", "SA", "UA"):
        return max(base_k, 30)
    return base_k


class VectorStore:
    """
    FAISS-based vector store with metadata filtering.
    Enforces country isolation (HC-5).
    """

    def __init__(self, index_path: Optional[str] = None):
        settings = get_settings()
        self.index_path = index_path or settings.faiss_index_path
        self.dimension = settings.openai_embedding_dimensions  # 3072
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[ChunkMetadata] = []
        self._metadata_db: Optional[MetadataDB] = None
        self._use_db = False  # True when SQLite metadata is available
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load index from disk if not already loaded.
        Prefers SQLite metadata (low RAM) over in-memory JSON."""
        if self._loaded:
            return
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.json")
        db_file = os.path.join(self.index_path, DB_FILENAME)

        if os.path.exists(index_file) and (os.path.exists(db_file) or os.path.exists(metadata_file)):
            # Use IO_FLAG_MMAP to avoid loading full index into RAM
            try:
                self.index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
                logger.info("Loaded FAISS index with mmap")
            except Exception:
                self.index = faiss.read_index(index_file)
                logger.info("Loaded FAISS index into memory (mmap unavailable)")

            # Prefer SQLite metadata (disk-backed, ~0 RAM)
            if os.path.exists(db_file):
                self._metadata_db = MetadataDB(db_file)
                self._use_db = True
                logger.info(f"Using SQLite metadata ({self._metadata_db.count()} chunks)")
            elif os.path.exists(metadata_file):
                # Auto-convert JSON → SQLite for next time
                try:
                    json_to_sqlite(metadata_file, db_file)
                    self._metadata_db = MetadataDB(db_file)
                    self._use_db = True
                    logger.info("Converted metadata.json → SQLite, using disk-backed store")
                except Exception:
                    # Fallback: load into memory
                    with open(metadata_file, "r") as f:
                        raw_metadata = json.load(f)
                    self.metadata = [ChunkMetadata.from_dict(m) for m in raw_metadata]
                    logger.info("Loaded metadata into memory (SQLite conversion failed)")

            logger.info(f"FAISS index has {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            logger.info("Created new empty FAISS index")
        self._loaded = True

    def _get_chunk(self, idx: int) -> Optional[ChunkMetadata]:
        """Get chunk metadata by FAISS index — from SQLite or in-memory list."""
        idx = int(idx)  # Ensure Python int (numpy.int64 breaks sqlite3 binding)
        if self._use_db and self._metadata_db:
            data = self._metadata_db.get(idx)
            if data is None:
                return None
            return ChunkMetadata.from_dict(data)
        if idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def add_chunks(
        self,
        embeddings: np.ndarray,
        metadata_list: list[ChunkMetadata],
    ) -> None:
        """Add embedded chunks to the index."""
        self._ensure_loaded()
        assert self.index is not None
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        logger.info(f"Added {len(metadata_list)} chunks. Total: {self.index.ntotal}")

    def search(
        self,
        query: str,
        country: str,
        device_class: Optional[str] = None,
        top_k: int = 10,
        active_only: bool = True,
    ) -> list[dict]:
        """
        Search for regulatory chunks matching query.
        HC-5 ENFORCED: Results are ALWAYS filtered by country.
        For EU/US, top_k is dynamically increased (100/150) to cover full regulatory baseline.
        """
        self._ensure_loaded()
        assert self.index is not None

        if self.index.ntotal == 0:
            return []

        effective_k = _dynamic_top_k(country, top_k, device_class)

        # Fix-3: Redis cache — return early on hit (bypasses FAISS + embed entirely)
        # Cache key includes country + device_class for HC-5 isolation.
        # active_only=True is the default and most common path; we only cache that.
        if active_only:
            try:
                from app.services.faiss_cache import get_cached, set_cached
                _cache_hit = get_cached(query, country, device_class)
                if _cache_hit:  # Truthy check: skip empty cached results
                    return _cache_hit[:effective_k]
            except Exception:
                pass  # Cache unavailable — fall through to FAISS

        # Embed query
        try:
            query_embedding = embed_text(query).reshape(1, -1)
        except Exception as e:
            logger.error("embed_text() failed in search(): %s", e)
            return []
        faiss.normalize_L2(query_embedding)

        # Search a large oversample to guarantee post-filter coverage.
        # Small countries (UA=477/34707=1.4%) need at least ~70x oversample
        # to reliably land enough country-specific hits.
        # Use max(effective_k * 80, 500) capped at index size.
        search_k = min(max(effective_k * 80, 500), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        raw_count = sum(1 for idx in indices[0] if idx != -1)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self._get_chunk(int(idx))
            if chunk is None:
                continue

            # HC-5: Country isolation — MANDATORY filter
            if chunk.country.upper() != country.upper():
                continue

            # Active-only filter
            if active_only and not chunk.is_active:
                continue

            # Fix-2: Lifecycle pre-filter — never return REVOKED/SUPERSEDED chunks.
            # These are dead law; downstream LLM must not see them as valid sources.
            if chunk.regulatory_status in ("REVOKED", "SUPERSEDED"):
                continue

            # Device class filter (optional)
            if device_class and chunk.device_classes:
                if device_class not in chunk.device_classes:
                    continue

            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "parent_text": chunk.parent_text,
                "regulation_name": chunk.regulation_name,
                "article": chunk.article,
                "clause": chunk.clause,
                "country": chunk.country,
                "device_classes": chunk.device_classes,
                "score": float(score),
                "source_url": chunk.source_url,
                "document_id": chunk.document_id,
                "section_path": chunk.section_path,
                # Fix-2: Expose lifecycle status so verify_claims/DVA can detect revocations
                "regulatory_status": chunk.regulatory_status,
                "superseded_by": chunk.superseded_by,
            })

            if len(results) >= effective_k:
                break

        yield_ratio = len(results) / max(raw_count, 1)
        if yield_ratio < 0.3:
            logger.warning(
                "FAISS post-filter yield %.2f%% for country=%s. Consider per-country indexes.",
                yield_ratio * 100,
                country,
            )

        # Fix-3: Store results in Redis cache for future identical queries
        if active_only and results:
            try:
                from app.services.faiss_cache import set_cached
                set_cached(query, country, results, device_class)
            except Exception:
                pass  # Cache write failure is non-fatal

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        self._ensure_loaded()
        assert self.index is not None

        Path(self.index_path).mkdir(parents=True, exist_ok=True)

        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.json")

        faiss.write_index(self.index, index_file)
        with open(metadata_file, "w") as f:
            json.dump([m.to_dict() for m in self.metadata], f)

        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")

    def get_countries(self) -> list[str]:
        """Get list of all countries in the index."""
        self._ensure_loaded()
        if self._use_db and self._metadata_db:
            return self._metadata_db.get_all_countries()
        return list(set(m.country for m in self.metadata))

    def get_chunk_count(self, country: Optional[str] = None) -> int:
        """Get count of chunks, optionally filtered by country."""
        self._ensure_loaded()
        if self._use_db and self._metadata_db:
            if country:
                return self._metadata_db.count_by_country(country)
            return self._metadata_db.count()
        if country:
            return sum(1 for m in self.metadata if m.country.upper() == country.upper())
        return len(self.metadata)

    def get_baseline_chunks(
        self,
        country: str,
        device_class: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """
        Return ALL baseline regulatory chunks for a country (and optional device class).
        Used for DVA Step 6: set difference baseline - matched_baseline = MISSING.
        Filters by country, is_active, and device_classes (if provided).
        """
        self._ensure_loaded()
        out = []

        if self._use_db and self._metadata_db:
            for idx, d in self._metadata_db.iter_by_country(country, active_only):
                if device_class and d.get("device_classes"):
                    if device_class not in d["device_classes"]:
                        continue
                out.append({
                    "index": idx,
                    "chunk_id": d.get("chunk_id", ""),
                    "text": d.get("text", ""),
                    "regulation_name": d.get("regulation_name", ""),
                    "article": d.get("article", ""),
                    "clause": d.get("clause"),
                    "country": d.get("country", ""),
                })
            return out

        for i, chunk in enumerate(self.metadata):
            if chunk.country.upper() != country.upper():
                continue
            if active_only and not chunk.is_active:
                continue
            if device_class and chunk.device_classes:
                if device_class not in chunk.device_classes:
                    continue
            out.append({
                "index": i,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "regulation_name": chunk.regulation_name,
                "article": chunk.article,
                "clause": chunk.clause,
                "country": chunk.country,
            })
        return out

    def get_chunks_by_document(
        self,
        country: str,
        document_id: Optional[str] = None,
        source_url: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """
        Return chunks for a given document (RAA change detection).
        Match by document_id and/or source_url; country is mandatory.
        Returns list with chunk_id, chunk_hash, text, regulation_name, etc.
        """
        self._ensure_loaded()
        out = []

        if self._use_db and self._metadata_db:
            for idx, d in self._metadata_db.iter_by_country(country, active_only):
                if document_id and (d.get("document_id") or "").strip() != (document_id or "").strip():
                    continue
                if source_url and (d.get("source_url") or "").strip() != (source_url or "").strip():
                    continue
                out.append({
                    "index": idx,
                    "chunk_id": d.get("chunk_id", ""),
                    "chunk_hash": d.get("chunk_hash"),
                    "text": d.get("text", ""),
                    "regulation_name": d.get("regulation_name", ""),
                    "article": d.get("article", ""),
                    "clause": d.get("clause"),
                    "country": d.get("country", ""),
                    "document_id": d.get("document_id"),
                    "source_url": d.get("source_url"),
                })
            return out

        for i, chunk in enumerate(self.metadata):
            if chunk.country.upper() != country.upper():
                continue
            if active_only and not chunk.is_active:
                continue
            if document_id and (chunk.document_id or "").strip() != (document_id or "").strip():
                continue
            if source_url and (chunk.source_url or "").strip() != (source_url or "").strip():
                continue
            out.append({
                "index": i,
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.chunk_hash,
                "text": chunk.text,
                "regulation_name": chunk.regulation_name,
                "article": chunk.article,
                "clause": chunk.clause,
                "country": chunk.country,
                "document_id": chunk.document_id,
                "source_url": chunk.source_url,
            })
        return out


# Module-level singleton
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
