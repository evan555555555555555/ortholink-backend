"""
OrthoLink Scout Scraper
Playwright-based scraper for JS-heavy regulatory portals (ANVISA, NMPA, etc.).
Falls back to requests+BS4 for static pages.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Noise tokens inherited from base scraper
NOISE_TOKENS = [
    "Firefox", "Google Chrome", "We use cookies",
    "Accept cookies", "JavaScript", "Enable JavaScript",
    "404", "Page not found", "Access denied",
]


@dataclass
class ScoutResult:
    """Result of a scout scraping operation."""
    url: str
    text: str
    title: str = ""
    success: bool = True
    error: Optional[str] = None
    method: str = "playwright"
    page_count: int = 0


def scrape_with_playwright(
    url: str,
    wait_selector: Optional[str] = None,
    wait_seconds: float = 5.0,
    scroll_to_bottom: bool = True,
    click_selectors: Optional[list[str]] = None,
) -> ScoutResult:
    """
    Scrape a JS-rendered page using Playwright.

    Args:
        url: Target URL
        wait_selector: CSS selector to wait for before extracting
        wait_seconds: Extra seconds to wait after page load for JS rendering
        scroll_to_bottom: Scroll page to trigger lazy-loaded content
        click_selectors: List of CSS selectors to click (e.g., cookie banners, expand buttons)
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            page = context.new_page()

            # Navigate
            page.goto(url, wait_until="networkidle", timeout=60000)

            # Wait for specific selector if provided
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=15000)
                except Exception:
                    logger.warning(f"Selector '{wait_selector}' not found, continuing...")

            # Click through cookie banners, expand buttons, etc.
            for selector in (click_selectors or []):
                try:
                    el = page.query_selector(selector)
                    if el:
                        el.click()
                        time.sleep(0.5)
                except Exception:
                    pass

            # Extra wait for JS rendering
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            # Scroll to bottom to trigger lazy-loaded content
            if scroll_to_bottom:
                page.evaluate("""
                    async () => {
                        const delay = ms => new Promise(r => setTimeout(r, ms));
                        for (let i = 0; i < 10; i++) {
                            window.scrollBy(0, window.innerHeight);
                            await delay(300);
                        }
                        window.scrollTo(0, 0);
                    }
                """)
                time.sleep(1)

            title = page.title()

            # Extract main content, stripping nav/footer/scripts
            text = page.evaluate("""
                () => {
                    // Remove noise elements
                    const remove = ['script', 'style', 'nav', 'footer', 'header',
                                    'aside', 'noscript', 'iframe'];
                    remove.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });

                    // Try to find main content container
                    const selectors = ['main', 'article', '#content', '.content',
                                       '.document-body', '.law-content', '.text-body',
                                       '#main-content', '.main-content'];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText.length > 500) {
                            return el.innerText;
                        }
                    }

                    // Fallback to body
                    return document.body.innerText;
                }
            """)

            browser.close()

            # Clean text
            text = _clean_scout_text(text)

            return ScoutResult(
                url=url,
                text=text,
                title=title,
                success=bool(text and len(text.split()) > 100),
                method="playwright",
            )

    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")
        return ScoutResult(url=url, text="", success=False, error="Playwright not installed")
    except Exception as e:
        logger.error(f"Playwright scraping failed for {url}: {e}")
        return ScoutResult(url=url, text="", success=False, error=str(e))


def scrape_pdf_url(url: str) -> ScoutResult:
    """
    Download and extract text from a PDF URL.
    Used for regulations published as PDFs (ANVISA, TGA, etc.).
    """
    import io
    import requests

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Try pdfplumber first, fall back to pymupdf
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                text = "\n\n".join(pages)
                page_count = len(pages)
        except ImportError:
            try:
                import fitz  # pymupdf
                doc = fitz.open(stream=response.content, filetype="pdf")
                pages = [page.get_text() for page in doc]
                text = "\n\n".join(pages)
                page_count = len(pages)
                doc.close()
            except ImportError:
                return ScoutResult(
                    url=url, text="", success=False,
                    error="Neither pdfplumber nor pymupdf installed"
                )

        text = _clean_scout_text(text)

        return ScoutResult(
            url=url,
            text=text,
            title=url.split("/")[-1],
            success=bool(text and len(text.split()) > 100),
            method="pdf",
            page_count=page_count,
        )

    except Exception as e:
        logger.error(f"PDF scraping failed for {url}: {e}")
        return ScoutResult(url=url, text="", success=False, error=str(e))


def scout_regulatory_portal(
    urls: list[dict],
) -> list[ScoutResult]:
    """
    Scout multiple URLs from a regulatory portal.

    Each URL dict should have:
        - url: str
        - type: "html" | "pdf"
        - wait_selector: Optional[str] (for HTML pages)
        - click_selectors: Optional[list[str]] (for HTML pages)

    Returns list of ScoutResults.
    """
    results = []
    for entry in urls:
        url = entry["url"]
        url_type = entry.get("type", "html")

        logger.info(f"[Scout] Scraping {url_type}: {url}")

        if url_type == "pdf":
            result = scrape_pdf_url(url)
        else:
            result = scrape_with_playwright(
                url=url,
                wait_selector=entry.get("wait_selector"),
                click_selectors=entry.get("click_selectors"),
            )

        results.append(result)

        if result.success:
            word_count = len(result.text.split())
            logger.info(f"  -> SUCCESS: {word_count} words extracted ({result.method})")
        else:
            logger.warning(f"  -> FAILED: {result.error}")

    return results


def _clean_scout_text(text: str) -> str:
    """Clean extracted text, removing noise."""
    lines = text.split("\n")
    lines = [ln for ln in lines if not any(tok in ln for tok in NOISE_TOKENS)]
    text = "\n".join(lines)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove noise patterns
    noise_patterns = [
        r"Accept\s+cookies?",
        r"Cookie\s+policy",
        r"Privacy\s+policy",
        r"Subscribe\s+to\s+newsletter",
        r"Share\s+on\s+(?:Facebook|Twitter|LinkedIn)",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()
