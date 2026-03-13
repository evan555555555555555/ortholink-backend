"""
OrthoLink Scraper
BeautifulSoup + Selenium fallback for government regulatory sites.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

NOISE_TOKENS = [
    "Firefox", "Google Chrome", "We use cookies",
    "Accept cookies", "JavaScript", "Enable JavaScript",
    "404", "Page not found", "Access denied",
]
MIN_WORD_COUNT = 500
REQUIRED_LEGAL_KEYWORDS = [
    # English
    "Article", "requirement", "shall", "manufacturer",
    "regulation", "device", "medical", "registration",
    "compliance", "approval", "section", "directive",
    # Spanish (MX)
    "artículo", "dispositivo", "fabricante", "salud",
    "requisito", "norma", "reglamento",
    # Portuguese (BR)
    "artigo", "fabricante", "regulamento", "resolução",
    # Russian (RU)
    "статья", "изделие", "регистрация", "производитель", "требование",
    # Chinese (CN)
    "医疗器械", "注册", "生产", "监督管理",
    # Japanese (JP)
    "医療機器", "承認", "届出",
    # Korean (KR)
    "의료기기", "허가", "등록",
    # German (CH)
    "Medizinprodukt", "Verordnung", "Artikel", "Hersteller",
    # Arabic (SA)
    "الأجهزة", "تسجيل",
]


@dataclass
class ValidationResult:
    """Result of validate_scraped_text. Caller decides whether to abort or continue."""

    passed: bool
    warnings: list[str]


def validate_scraped_text(text: str, country: str, url: str) -> ValidationResult:
    """
    Returns ValidationResult(passed=bool, warnings=list[str]).
    Logs WARNING for each failure. Does NOT raise — caller decides whether
    to abort or continue with degraded quality flag.
    """
    warnings: list[str] = []
    for token in NOISE_TOKENS:
        if token in text:
            warnings.append(f"Noise token '{token}' found — likely scraped nav/cookie banner")
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        warnings.append(f"Word count {word_count} below minimum {MIN_WORD_COUNT}")
    if not any(kw in text for kw in REQUIRED_LEGAL_KEYWORDS):
        warnings.append("No legal keywords found — may not be regulatory text")
    passed = len(warnings) == 0
    if not passed:
        logger.warning(f"[{country}] Scrape validation failed for {url}: {warnings}")
    return ValidationResult(passed=passed, warnings=warnings)

# Request headers to mimic a real browser
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class ScraperResult:
    """Result of a scraping operation."""

    def __init__(
        self,
        url: str,
        text: str,
        title: str = "",
        success: bool = True,
        error: Optional[str] = None,
        method: str = "requests",
    ):
        self.url = url
        self.text = text
        self.title = title
        self.success = success
        self.error = error
        self.method = method


def _extract_rada_main_content(soup: BeautifulSoup) -> str:
    """
    Extract the main regulatory text from Rada (zakon.rada.gov.ua) HTML.
    Used when the full body is dominated by nav/cookie/outdated-browser messages.
    """
    # Ukrainian legal keywords that indicate the regulation body
    legal_indicators = ["Стаття", "статті", "регламент", "виробник", "вимоги", "медичн", "постанов"]
    candidates = []
    for tag in soup.find_all(["div", "article", "main", "section"]):
        t = tag.get_text(separator="\n", strip=True)
        if len(t) < 500:
            continue
        count = sum(1 for kw in legal_indicators if kw in t)
        if count >= 2:
            candidates.append((len(t), t))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return _clean_text(candidates[0][1])
    # Fallback: full body
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    return _clean_text(soup.get_text(separator="\n", strip=True))


def scrape_url(url: str, use_selenium_fallback: bool = True) -> ScraperResult:
    """
    Scrape a URL for regulatory text content.
    First tries requests + BeautifulSoup, falls back to Selenium for JS-rendered pages.
    For zakon.rada.gov.ua, extracts main content block to avoid nav/outdated-browser noise.
    """
    is_rada = "zakon.rada.gov.ua" in url
    result = _scrape_with_requests(url)
    if result.success:
        if is_rada and len(result.text.split()) < MIN_WORD_COUNT:
            # Re-parse and extract main content from the HTML we have
            try:
                response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
                soup = BeautifulSoup(response.text, "html.parser")
                extracted = _extract_rada_main_content(soup)
                if len(extracted.split()) >= MIN_WORD_COUNT:
                    return ScraperResult(url=url, text=extracted, title=result.title, success=True, method="requests")
            except Exception:
                pass
        return result

    if use_selenium_fallback:
        logger.info(f"Falling back to Selenium for {url}")
        result = _scrape_with_selenium(url)
        if result.success and is_rada and len(result.text.split()) < MIN_WORD_COUNT:
            # page_source from Selenium may contain full HTML even when body.text was minimal
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.support.ui import WebDriverWait
                options = Options()
                options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument(f"user-agent={DEFAULT_HEADERS['User-Agent']}")
                driver = webdriver.Chrome(options=options)
                driver.set_page_load_timeout(30)
                try:
                    driver.get(url)
                    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    html = driver.page_source
                    driver.quit()
                    driver = None
                    soup = BeautifulSoup(html, "html.parser")
                    body = _extract_zakon_rada_body(soup)
                    if body and len(body.split()) >= MIN_WORD_COUNT:
                        return ScraperResult(url=url, text=_clean_text(body), title=result.title, success=True, method="selenium")
                    extracted = _extract_rada_main_content(soup)
                    if len(extracted.split()) >= MIN_WORD_COUNT:
                        return ScraperResult(url=url, text=extracted, title=result.title, success=True, method="selenium")
                finally:
                    if driver:
                        driver.quit()
            except Exception as e:
                logger.warning(f"Rada extraction after Selenium failed: {e}")
        return result

    return result


def _extract_zakon_rada_body(soup: BeautifulSoup) -> Optional[str]:
    """Extract main law content from zakon.rada.gov.ua HTML. Returns None if not found."""
    # Common selectors for Ukrainian Rada law pages
    for selector in ["#content", ".document-body", ".law-content", "main", "article", ".text"]:
        try:
            el = soup.select_one(selector)
            if el and len(el.get_text(strip=True)) > 200:
                for tag in el(["script", "style"]):
                    tag.decompose()
                return el.get_text(separator="\n", strip=True)
        except Exception:
            continue
    return None


def _scrape_with_requests(url: str) -> ScraperResult:
    """Scrape using requests + BeautifulSoup."""
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        title = soup.title.string if soup.title else ""
        if "zakon.rada.gov.ua" in url:
            body = _extract_zakon_rada_body(soup)
            text = body if body else soup.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean the text
        text = _clean_text(text)

        return ScraperResult(
            url=url,
            text=text,
            title=title or "",
            success=True,
            method="requests",
        )

    except requests.RequestException as e:
        logger.warning(f"requests failed for {url}: {e}")
        return ScraperResult(
            url=url,
            text="",
            success=False,
            error=str(e),
            method="requests",
        )


def _scrape_with_selenium(url: str) -> ScraperResult:
    """Scrape using Selenium for JS-rendered pages."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={DEFAULT_HEADERS['User-Agent']}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)

        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            title = driver.title
            # Get text from body
            body = driver.find_element(By.TAG_NAME, "body")
            text = body.text

            text = _clean_text(text)

            return ScraperResult(
                url=url,
                text=text,
                title=title,
                success=True,
                method="selenium",
            )
        finally:
            driver.quit()

    except ImportError:
        logger.error("Selenium not installed. Cannot use fallback scraper.")
        return ScraperResult(
            url=url,
            text="",
            success=False,
            error="Selenium not installed",
            method="selenium",
        )
    except Exception as e:
        logger.error(f"Selenium scraping failed for {url}: {e}")
        return ScraperResult(
            url=url,
            text="",
            success=False,
            error=str(e),
            method="selenium",
        )


def _clean_text(text: str) -> str:
    """
    Clean scraped text by removing noise.
    PRD: If clean_text contains 'Firefox' or 'Google Chrome', the scraper has failed.
    Strip lines containing NOISE_TOKENS (e.g. eCFR footer) so validation can pass.
    """
    # Drop lines that contain browser/nav noise (eCFR and others put these in footer)
    lines = text.split("\n")
    lines = [ln for ln in lines if not any(tok in ln for tok in NOISE_TOKENS)]
    text = "\n".join(lines)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove common noise patterns
    noise_patterns = [
        r"Accept\s+cookies?",
        r"Cookie\s+policy",
        r"Privacy\s+policy",
        r"Subscribe\s+to\s+newsletter",
        r"Share\s+on\s+(?:Facebook|Twitter|LinkedIn)",
        r"Copyright\s+\d{4}",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def load_from_file(file_path: str, extract_rada_main: bool = False) -> ScraperResult:
    """
    Load regulatory text from a local file (PDF, DOCX, TXT, HTML).
    Used for geo-blocked sources (China NMPA, Russia Roszdravnadzor).
    If extract_rada_main=True and file is HTML, extract main content (Rada pages).
    """
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_path.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif file_path.endswith((".docx", ".doc")):
            import docx
            doc = docx.Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif file_path.endswith((".html", ".htm")):
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = _extract_rada_main_content(soup) if extract_rada_main else soup.get_text(separator="\n", strip=True)
            text = _clean_text(text)
        else:
            return ScraperResult(
                url=file_path,
                text="",
                success=False,
                error=f"Unsupported file format: {file_path}",
                method="file",
            )

        text = _clean_text(text)
        return ScraperResult(
            url=file_path,
            text=text,
            title=file_path.split("/")[-1],
            success=True,
            method="file",
        )

    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return ScraperResult(
            url=file_path,
            text="",
            success=False,
            error=str(e),
            method="file",
        )
