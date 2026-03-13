"""
OrthoLink Chunk Audit
Post-ingestion validation: sample 10 random chunks per country.
PRD: This is how we caught the previous contractor's failures.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from app.tools.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ChunkAuditResult:
    """Result of a chunk audit for a single chunk."""

    chunk_id: str
    country: str
    regulation_name: str
    article: str
    passed: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Aggregate audit report for a country."""

    country: str
    total_chunks: int
    sampled: int
    passed: int
    failed: int
    results: list[ChunkAuditResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.sampled == 0:
            return 0.0
        return self.passed / self.sampled


def audit_chunks(
    country: str,
    sample_size: int = 10,
    vector_store: Optional[VectorStore] = None,
) -> AuditReport:
    """
    Audit embedded chunks for a country by sampling and validating.

    Checks:
    1. Metadata completeness (country, regulation_name, article)
    2. Text quality (not empty, no browser noise, minimum length)
    3. Hash integrity
    4. Country isolation (chunk country matches filter)

    PRD: "sample 10 random chunks, verify metadata correctness,
    verify text is clean legal content."
    """
    store = vector_store or get_vector_store()

    # Get all chunks for this country
    country_chunks = [m for m in store.metadata if m.country.upper() == country.upper()]
    total_chunks = len(country_chunks)

    if total_chunks == 0:
        return AuditReport(
            country=country,
            total_chunks=0,
            sampled=0,
            passed=0,
            failed=0,
        )

    # Sample random chunks
    sample = random.sample(country_chunks, min(sample_size, total_chunks))
    results: list[ChunkAuditResult] = []

    for chunk in sample:
        issues = _validate_chunk(chunk, country)
        result = ChunkAuditResult(
            chunk_id=chunk.chunk_id,
            country=chunk.country,
            regulation_name=chunk.regulation_name,
            article=chunk.article,
            passed=len(issues) == 0,
            issues=issues,
        )
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    report = AuditReport(
        country=country,
        total_chunks=total_chunks,
        sampled=len(results),
        passed=passed,
        failed=failed,
        results=results,
    )

    # Log results
    if failed > 0:
        logger.warning(
            f"CHUNK AUDIT {country}: {failed}/{len(results)} samples FAILED. "
            f"Total chunks: {total_chunks}"
        )
        for r in results:
            if not r.passed:
                logger.warning(f"  FAILED {r.chunk_id}: {r.issues}")
    else:
        logger.info(
            f"CHUNK AUDIT {country}: {passed}/{len(results)} samples PASSED. "
            f"Total chunks: {total_chunks}"
        )

    return report


def _validate_chunk(chunk, expected_country: str) -> list[str]:
    """Validate a single chunk. Returns list of issues (empty = pass)."""
    issues: list[str] = []

    # 1. Metadata completeness
    if not chunk.country:
        issues.append("Missing country metadata")
    if not chunk.regulation_name:
        issues.append("Missing regulation_name metadata")
    if not chunk.article:
        issues.append("Missing article metadata")

    # 2. Country isolation check
    if chunk.country and chunk.country.upper() != expected_country.upper():
        issues.append(
            f"Country mismatch: chunk has '{chunk.country}', expected '{expected_country}'"
        )

    # 3. Text quality
    if not chunk.text or len(chunk.text.strip()) == 0:
        issues.append("Empty text content")
    elif len(chunk.text.strip()) < 20:
        issues.append(f"Text too short ({len(chunk.text.strip())} chars)")

    # 4. Browser noise detection (PRD requirement)
    noise_indicators = ["firefox", "google chrome", "enable javascript", "your browser"]
    text_lower = chunk.text.lower()
    for noise in noise_indicators:
        if noise in text_lower:
            issues.append(f"Browser noise detected: '{noise}'")

    # 5. Hash integrity
    if not chunk.chunk_hash:
        issues.append("Missing chunk_hash")

    return issues


def main():
    """CLI: chunk_audit.py --country UA --sample 10"""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Audit embedded chunks for a country")
    parser.add_argument("--country", required=True, help="Country code (e.g. UA, US, IN)")
    parser.add_argument("--sample", type=int, default=10, help="Number of chunks to sample")
    args = parser.parse_args()
    report = audit_chunks(args.country, sample_size=args.sample)
    print(f"Country: {report.country}, Total: {report.total_chunks}, Sampled: {report.sampled}, Passed: {report.passed}, Failed: {report.failed}")
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.chunk_id} | {r.regulation_name} | {r.article} | {r.issues or []}")
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    import sys
    main()
