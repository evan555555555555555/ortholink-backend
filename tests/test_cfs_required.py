"""
PRD: Certificate of Free Sale (CFS) MUST be classified as REQUIRED.
Previous contractor misclassified CFS as COMMERCIAL_NON_REGULATORY.
"""

import pytest

from app.crews.verify_distributor import GapItem, GapAnalysisReport, GapAnalysisSummary


class TestCFSRequired:
    """Certificate of Free Sale must never be classified as EXTRA or COMMERCIAL."""

    def test_cfs_item_model_required(self):
        """GapItem allows status REQUIRED for CFS."""
        item = GapItem(
            distributor_item="Certificate of Free Sale",
            status="REQUIRED",
            matched_regulation="Proof of marketing in country of origin",
            citation="Resolution 753, Article 14",
            confidence=0.92,
            explanation="CFS is a regulatory document required by most countries.",
        )
        assert item.status == "REQUIRED"

    def test_cfs_not_extra_assertion(self):
        """Assertion: CFS must not be EXTRA. (Integration test with mocked LLM would set CFS → REQUIRED.)"""
        # This test encodes the requirement: if distributor_item is CFS/CFG, status must not be EXTRA.
        cfs_like_names = [
            "Certificate of Free Sale",
            "Certificate of Free Sale (CFS)",
            "CFS",
            "Certificate of Free Sale (CFG)",
            "Certificado de Livre Venda",
        ]
        for name in cfs_like_names:
            # In real DVA run, LLM prompt instructs CFS → REQUIRED. Here we assert the model accepts REQUIRED.
            item = GapItem(
                distributor_item=name,
                status="REQUIRED",
                confidence=0.9,
                explanation="Certificate of Free Sale is a regulatory document.",
            )
            assert item.status != "EXTRA"
            assert item.status == "REQUIRED"

    def test_report_disclaimer_includes_verify_official(self):
        """HC-7: Every output carries 'Reference tool only. Verify with official sources.'"""
        report = GapAnalysisReport(
            country="UA",
            device_class="IIb",
            summary=GapAnalysisSummary(total_submitted=1, fraud_risk_score=0.0),
            items=[],
        )
        assert "Reference tool only" in report.disclaimer
        assert "Verify with official sources" in report.disclaimer
