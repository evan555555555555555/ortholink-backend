"""
Tests for scraper content validator.
"""


from app.ingestion.scraper_validator import validate_scraped_content
from app.ingestion.scraper import validate_scraped_text, _clean_text


class TestScraperValidator:
    """Test content quality validation."""

    def test_valid_regulatory_content(self):
        text = (
            "Article 1: Scope and Application\n"
            "This regulation establishes the requirements for medical devices. "
            "The manufacturer must ensure compliance with all applicable standards. "
            "Classification of devices shall follow the risk-based approach. "
            "Technical documentation must be maintained throughout the device lifecycle. "
            "Post-market surveillance is mandatory for all device classes. "
        ) * 20  # Make it long enough

        result = validate_scraped_content(text)
        assert result.is_valid is True
        assert result.word_count >= 500

    def test_too_short_content(self):
        text = "This is too short."
        result = validate_scraped_content(text)
        assert result.is_valid is False
        assert any("word count" in e.lower() for e in result.errors)

    def test_browser_noise_detected(self):
        text = (
            "Enable JavaScript in your browser to continue. "
            "Please update Firefox or Google Chrome. "
        ) * 100

        result = validate_scraped_content(text)
        # Should flag browser noise
        assert any("noise" in e.lower() for e in result.errors)

    def test_no_legal_keywords(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Banana smoothie recipe with strawberries. "
            "Weather forecast for tomorrow is sunny. "
        ) * 50

        result = validate_scraped_content(text)
        assert result.is_valid is False

    def test_empty_content(self):
        result = validate_scraped_content("")
        assert result.is_valid is False


def test_scraper_rejects_noise():
    result = validate_scraped_text("Please enable Firefox to view this page", "UA", "http://test")
    assert result.passed is False
    assert any("Firefox" in w for w in result.warnings)


def test_scraper_rejects_short():
    result = validate_scraped_text("Article 1. Requirements.", "UA", "http://test")
    assert result.passed is False


def test_clean_text_strips_noise_lines():
    """After _clean_text, text with a Firefox line no longer fails validation (eCFR case)."""
    long_legal = (
        "Article 1. Scope. The manufacturer shall ensure compliance with all requirements. "
        * 150
    )
    text_with_noise = long_legal + "\nUse Firefox or Google Chrome for best experience."
    assert validate_scraped_text(text_with_noise, "US", "http://ecfr.gov").passed is False
    cleaned = _clean_text(text_with_noise)
    result = validate_scraped_text(cleaned, "US", "http://ecfr.gov")
    assert result.passed is True
