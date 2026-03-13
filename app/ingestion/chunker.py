"""
OrthoLink Chunker
Hierarchy-preserving chunker that splits at Article/Section/Clause boundaries.
PRD: Split at Article/Section boundaries (NOT token count).
Parent Document Retriever pattern: child clauses for search, full parent article for generation.
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A regulatory text chunk with hierarchy metadata."""

    chunk_id: str
    text: str
    parent_text: str
    country: str
    regulation_name: str
    article: str
    clause: Optional[str] = None
    device_classes: list[str] = field(default_factory=list)
    source_url: Optional[str] = None
    language: str = "en"
    original_language: Optional[str] = None
    chunk_hash: str = ""
    document_id: Optional[str] = None
    section_path: Optional[str] = None

    def __post_init__(self):
        if not self.chunk_hash:
            self.chunk_hash = hashlib.sha256(self.text.encode()).hexdigest()
        if self.section_path is None and self.regulation_name:
            self.section_path = self.regulation_name + " > " + self.article
            if self.clause:
                self.section_path += " > Clause " + str(self.clause)


# Patterns for detecting article/section boundaries across different regulatory formats
ARTICLE_PATTERNS = [
    # European style: Article 1, Article 2, etc.
    r"(?:^|\n)(Article\s+\d+[\.\:]?\s*[^\n]*)",
    # US CFR style: § 820.1, § 801.1, etc.
    r"(?:^|\n)(§\s*\d+[\.\d]*\s*[^\n]*)",
    # Section style: Section 1, Section 2, etc.
    r"(?:^|\n)(Section\s+\d+[\.\:]?\s*[^\n]*)",
    # Chapter style
    r"(?:^|\n)(Chapter\s+\d+[\.\:]?\s*[^\n]*)",
    # Ukrainian Resolution style
    r"(?:^|\n)(Resolution\s+\d+[\.\:]?\s*[^\n]*)",
    r"(?:^|\n)(Annex\s+[IVXLCDM]+[\.\:]?\s*[^\n]*)",
    # Numbered style: 1. Title, 2. Title, etc.
    r"(?:^|\n)(\d+\.\s+[A-Z][^\n]*)",
    # Schedule style (common in Indian regulations)
    r"(?:^|\n)(Schedule\s+[IVXLCDM\d]+[\.\:]?\s*[^\n]*)",
    # Part style
    r"(?:^|\n)(Part\s+[IVXLCDM\d]+[\.\:]?\s*[^\n]*)",
]

# Clause-level patterns (sub-sections within articles)
CLAUSE_PATTERNS = [
    r"(?:^|\n)\s*\(([a-z])\)\s",          # (a), (b), (c)
    r"(?:^|\n)\s*\((\d+)\)\s",             # (1), (2), (3)
    r"(?:^|\n)\s*([ivxlcdm]+)\.\s",        # i., ii., iii.
    r"(?:^|\n)\s*(\d+\.\d+)\s",            # 1.1, 1.2, etc.
    r"(?:^|\n)\s*(\d+\.\d+\.\d+)\s",       # 1.1.1, 1.1.2, etc.
]


def chunk_regulatory_text(
    text: str,
    country: str,
    regulation_name: str,
    device_classes: Optional[list[str]] = None,
    source_url: Optional[str] = None,
    language: str = "en",
    original_language: Optional[str] = None,
    document_id: Optional[str] = None,
) -> list[Chunk]:
    """
    Split regulatory text into chunks at Article/Section boundaries.

    Uses Parent Document Retriever pattern:
    - Child clauses are stored as search-level chunks
    - Full parent article text is preserved for generation context

    PRD: Split at Article/Section boundaries (NOT token count).
    """
    device_classes = device_classes or []
    chunks: list[Chunk] = []

    # Find article-level boundaries
    articles = _split_into_articles(text)

    if not articles:
        # If no article boundaries found, treat entire text as one chunk
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            parent_text=text,
            country=country,
            regulation_name=regulation_name,
            article="Full Text",
            device_classes=device_classes,
            source_url=source_url,
            language=language,
            original_language=original_language,
            document_id=document_id,
        )
        chunks.append(chunk)
        return chunks

    for article_title, article_text in articles:
        # Create parent-level chunk (full article)
        parent_chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=article_text,
            parent_text=article_text,
            country=country,
            regulation_name=regulation_name,
            article=article_title,
            device_classes=device_classes,
            source_url=source_url,
            language=language,
            original_language=original_language,
            document_id=document_id,
        )
        chunks.append(parent_chunk)

        # Split into clause-level chunks for finer-grained search
        clauses = _split_into_clauses(article_text)
        for clause_id, clause_text in clauses:
            if len(clause_text.strip()) < 50:
                continue  # Skip very short clauses

            clause_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=clause_text,
                parent_text=article_text,  # Parent Document Retriever pattern
                country=country,
                regulation_name=regulation_name,
                article=article_title,
                clause=clause_id,
                device_classes=device_classes,
                source_url=source_url,
                language=language,
                original_language=original_language,
                document_id=document_id,
            )
            chunks.append(clause_chunk)

    logger.info(
        f"Chunked {regulation_name} ({country}): "
        f"{len(articles)} articles, {len(chunks)} total chunks"
    )

    return chunks


def _split_into_articles(text: str) -> list[tuple[str, str]]:
    """
    Split text into article-level sections.
    Returns list of (article_title, article_text) tuples.
    """
    # Try each pattern and use the one that produces the most splits
    best_splits: list[tuple[str, str]] = []
    best_count = 0

    for pattern in ARTICLE_PATTERNS:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if len(matches) > best_count:
            splits = []
            for i, match in enumerate(matches):
                title = match.group(1).strip()
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                article_text = text[start:end].strip()
                if article_text:
                    splits.append((title, article_text))
            if len(splits) > best_count:
                best_splits = splits
                best_count = len(splits)

    return best_splits


def _split_into_clauses(article_text: str) -> list[tuple[str, str]]:
    """
    Split an article into clause-level sections.
    Returns list of (clause_id, clause_text) tuples.
    """
    clauses: list[tuple[str, str]] = []

    for pattern in CLAUSE_PATTERNS:
        matches = list(re.finditer(pattern, article_text, re.MULTILINE))
        if len(matches) >= 2:  # Need at least 2 clauses to be meaningful
            for i, match in enumerate(matches):
                clause_id = match.group(1)
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(article_text)
                clause_text = article_text[start:end].strip()
                if clause_text:
                    clauses.append((clause_id, clause_text))
            break  # Use the first pattern that works

    return clauses
