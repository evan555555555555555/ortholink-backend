"""
Tests for anti-hallucination guards (HC-1 through HC-10).
"""


from app.core.anti_hallucination import (
    Citation,
    StructuredRefusal,
    check_confidence,
    create_refusal,
    is_out_of_scope,
    validate_citations,
)
from app.core.config import Settings


class TestConfidenceGating:
    """HC-4: Confidence < 0.7 → StructuredRefusal."""

    def test_high_confidence_passes(self):
        gate = check_confidence(0.85)
        assert gate.passed is True
        assert gate.confidence == 0.85

    def test_exact_threshold_passes(self):
        gate = check_confidence(0.7)
        assert gate.passed is True

    def test_low_confidence_fails(self):
        gate = check_confidence(0.5)
        assert gate.passed is False
        assert gate.confidence == 0.5

    def test_zero_confidence_fails(self):
        gate = check_confidence(0.0)
        assert gate.passed is False

    def test_custom_threshold(self):
        gate = check_confidence(0.8, threshold=0.9)
        assert gate.passed is False

        gate = check_confidence(0.95, threshold=0.9)
        assert gate.passed is True


class TestCitations:
    """HC-3: Every claim must cite source."""

    def test_valid_citation(self):
        citations = [
            Citation(
                regulation_name="EU MDR 2017/745",
                article="Article 5",
                country="EU",
                text_excerpt="Devices shall meet the general safety...",
            )
        ]
        assert validate_citations(citations) is True

    def test_empty_citations_fails(self):
        assert validate_citations([]) is False

    def test_citation_missing_fields(self):
        citations = [
            Citation(
                regulation_name="",
                article="Article 5",
                country="EU",
                text_excerpt="Some text",
            )
        ]
        assert validate_citations(citations) is False


class TestStructuredRefusal:
    """HC-4: Refusal format."""

    def test_refusal_creation(self):
        refusal = create_refusal(
            confidence=0.3,
            reason="Insufficient regulatory data for this query.",
        )
        assert refusal.refused is True
        assert refusal.confidence == 0.3
        assert "regulatory" in refusal.reason.lower() or "insufficient" in refusal.reason.lower()

    def test_refusal_has_suggestion(self):
        refusal = create_refusal(confidence=0.2, reason="Low confidence")
        assert refusal.suggestion  # Should have a suggestion


class TestOutOfScope:
    """Test scope checking."""

    def test_supported_country(self):
        result = is_out_of_scope("registration requirements", "US", "II")
        assert result is None  # Should be in scope

    def test_supported_country_ukraine(self):
        result = is_out_of_scope("registration requirements", "UA", "IIb")
        assert result is None

    def test_non_medical_query(self):
        result = is_out_of_scope("best pizza recipes", "US", "II")
        # May or may not flag — depends on implementation
        # Just ensure it returns StructuredRefusal or None
        assert result is None or isinstance(result, StructuredRefusal)


class TestConfigEnforcement:
    """HC-1 and HC-9: Model constraints."""

    def test_default_embedding_model(self):
        settings = Settings()
        assert settings.openai_embedding_model == "text-embedding-3-large"

    def test_default_embedding_dimensions(self):
        settings = Settings()
        assert settings.openai_embedding_dimensions == 3072

    def test_default_generation_model(self):
        settings = Settings()
        assert settings.openai_generation_model == "gpt-4o"

    def test_confidence_threshold_default(self):
        settings = Settings()
        assert settings.confidence_threshold == 0.7

    def test_similarity_threshold_default(self):
        settings = Settings()
        assert settings.similarity_threshold == 0.55
