"""
OrthoLink Scraper Validator
Sanity checks: word count, keyword presence, noise detection.
PRD: Do not embed garbage.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum word count for a valid regulatory document
MIN_WORD_COUNT = 500

# Required legal keywords (at least one must be present)
# Multilingual to support 15-country regulatory platform
LEGAL_KEYWORDS = [
    # English
    "article", "requirement", "manufacturer", "regulation",
    "directive", "section", "clause", "compliance",
    "registration", "authorization", "approval", "device",
    "medical", "class", "annex", "schedule",
    # Spanish (MX)
    "artículo", "dispositivo", "médico", "registro",
    "fabricante", "regulación", "reglamento", "salud",
    "norma oficial", "cofepris", "requisito",
    # Portuguese (BR)
    "artigo", "dispositivo médico", "registro", "anvisa",
    "fabricante", "regulamento", "resolução",
    # Russian (RU)
    "статья", "медицинское изделие", "регистрация",
    "производитель", "требование", "постановление",
    "росздравнадзор",
    # Chinese (CN)
    "医疗器械", "注册", "生产", "监督管理",
    "分类", "条例", "规定",
    # Japanese (JP)
    "医療機器", "承認", "届出", "製造販売",
    "条", "規則",
    # Korean (KR)
    "의료기기", "허가", "등록", "제조",
    "조", "규정",
    # German (CH)
    "medizinprodukt", "verordnung", "artikel",
    "hersteller", "zulassung", "konformität",
    # Arabic (SA)
    "الأجهزة الطبية", "تسجيل", "تصنيف",
]

# Noise indicators (if found, scraper likely captured browser UI)
NOISE_INDICATORS = [
    "firefox", "google chrome", "internet explorer",
    "enable javascript", "your browser", "update your browser",
    "cookie settings", "accept all cookies",
]


@dataclass
class ValidationResult:
    """Result of scraper validation."""

    is_valid: bool
    word_count: int
    has_legal_keywords: bool
    has_noise: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    found_keywords: list[str] = field(default_factory=list)
    found_noise: list[str] = field(default_factory=list)


def validate_scraped_content(text: str) -> ValidationResult:
    """
    Validate scraped content for quality and relevance.

    Checks:
    1. Minimum word count (>500 words)
    2. Presence of legal keywords
    3. Absence of browser/noise indicators

    PRD: "Do not embed garbage."
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Word count check
    words = text.split()
    word_count = len(words)
    if word_count < MIN_WORD_COUNT:
        errors.append(
            f"Word count {word_count} is below minimum {MIN_WORD_COUNT}. "
            "Content may be incomplete or invalid."
        )

    # Legal keyword check
    text_lower = text.lower()
    found_keywords = [kw for kw in LEGAL_KEYWORDS if kw in text_lower]
    has_legal_keywords = len(found_keywords) > 0
    if not has_legal_keywords:
        errors.append(
            "No legal keywords found. Content may not be regulatory text. "
            f"Expected at least one of: {', '.join(LEGAL_KEYWORDS[:8])}"
        )

    # Noise detection
    found_noise = [indicator for indicator in NOISE_INDICATORS if indicator in text_lower]
    has_noise = len(found_noise) > 0
    if has_noise:
        errors.append(
            f"Browser noise detected: {', '.join(found_noise)}. "
            "Scraper likely captured browser UI instead of page content."
        )

    # Additional quality checks
    if word_count > 0:
        # Check for excessive repetition (sign of scraper failure)
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / word_count
        if uniqueness_ratio < 0.1:
            warnings.append(
                f"Low word uniqueness ratio ({uniqueness_ratio:.2f}). "
                "Content may contain excessive repetition."
            )

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        word_count=word_count,
        has_legal_keywords=has_legal_keywords,
        has_noise=has_noise,
        warnings=warnings,
        errors=errors,
        found_keywords=found_keywords,
        found_noise=found_noise,
    )

    if not is_valid:
        logger.warning(f"Scraper validation FAILED: {errors}")
    else:
        logger.info(
            f"Scraper validation passed: {word_count} words, "
            f"{len(found_keywords)} legal keywords found"
        )

    return result
