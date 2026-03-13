"""
Tests for DVA pipeline CSV parsing and report models.
"""

import pytest

from app.crews.verify_distributor import (
    GapAnalysisReport,
    GapAnalysisSummary,
    GapItem,
    parse_csv,
)


class TestCSVParsing:
    """Test CSV parsing for distributor document lists."""

    def test_single_column_csv(self):
        csv_content = "document\nTechnical File\nCE Marking Certificate\nISO 13485"
        items = parse_csv(csv_content)
        assert len(items) == 3
        assert "Technical File" in items
        assert "CE Marking Certificate" in items
        assert "ISO 13485" in items

    def test_multicolumn_csv(self):
        csv_content = (
            "id,document,status\n"
            "1,Technical File,pending\n"
            "2,CE Certificate,submitted\n"
        )
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert "Technical File" in items

    def test_empty_csv(self):
        items = parse_csv("")
        assert items == []

    def test_csv_with_extra_whitespace(self):
        csv_content = "document\n  Technical File  \n  CE Marking  \n"
        items = parse_csv(csv_content)
        assert len(items) == 2
        assert "Technical File" in items
        assert "CE Marking" in items

    def test_csv_without_header(self):
        csv_content = "Technical File\nCE Certificate\nISO 13485"
        items = parse_csv(csv_content)
        # Should still parse items
        assert len(items) >= 2


class TestGapItemModel:
    """Test GapItem Pydantic model."""

    def test_required_item(self):
        item = GapItem(
            distributor_item="Technical File",
            status="REQUIRED",
            matched_regulation="Article 10: Technical Documentation",
            citation="EU MDR 2017/745, Article 10",
            confidence=0.92,
            explanation="Technical File is explicitly required by Article 10.",
        )
        assert item.status == "REQUIRED"
        assert item.confidence == 0.92

    def test_extra_item(self):
        item = GapItem(
            distributor_item="Notarized Company Charter",
            status="EXTRA",
            confidence=0.85,
            explanation="No regulation requires a notarized company charter.",
        )
        assert item.status == "EXTRA"
        assert item.needs_human_review is False

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            GapItem(
                distributor_item="Test",
                status="REQUIRED",
                confidence=1.5,  # Out of bounds
                explanation="Test",
            )

    def test_human_review_flag(self):
        item = GapItem(
            distributor_item="Unknown Document",
            status="UNVERIFIABLE",
            confidence=0.3,
            explanation="Cannot determine requirement.",
            needs_human_review=True,
        )
        assert item.needs_human_review is True


class TestGapAnalysisReport:
    """Test GapAnalysisReport model."""

    def test_report_creation(self):
        summary = GapAnalysisSummary(
            total_submitted=10,
            required=6,
            extra=3,
            missing=1,
            fraud_risk_score=0.3,
        )
        report = GapAnalysisReport(
            country="UA",
            device_class="IIb",
            summary=summary,
            items=[],
        )
        assert report.country == "UA"
        assert report.summary.fraud_risk_score == 0.3
        assert report.analysis_id  # Auto-generated

    def test_report_has_disclaimer(self):
        summary = GapAnalysisSummary(
            total_submitted=0,
            fraud_risk_score=0.0,
        )
        report = GapAnalysisReport(
            country="US",
            device_class="II",
            summary=summary,
            items=[],
        )
        assert "legal advice" in report.disclaimer.lower()
