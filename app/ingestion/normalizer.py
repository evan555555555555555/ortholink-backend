"""
OrthoLink Document Normalizer
Normalises regulatory documents from any input format into a standard structure.

Preserves legal hierarchy (HC-3: Article/Section/Clause boundaries).

Input formats handled:
  - FHIR/HL7 JSON  (Swissdamed M2M, ANVISA SIUD)
  - HTML           (TGA, FDA, EUDAMED web scraping)
  - Plain text     (most regulatory documents)
  - JSON API responses (GUDID, MDALL)

LegalSection hierarchy:
  PART > CHAPTER > ARTICLE > SECTION > CLAUSE > SUBCLAUSE

Usage:
    from app.ingestion.normalizer import DocumentNormalizer

    normalizer = DocumentNormalizer()
    doc = normalizer.normalize(raw=html_str, source_type="html", metadata={})
    print(doc.title, doc.body)
"""

import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Legal hierarchy enum
# ─────────────────────────────────────────────────────────────────────────────

class LegalLevel(str, Enum):
    PART = "PART"
    CHAPTER = "CHAPTER"
    ARTICLE = "ARTICLE"
    SECTION = "SECTION"
    CLAUSE = "CLAUSE"
    SUBCLAUSE = "SUBCLAUSE"


# Ordinal rank for sorting / nesting — lower number = higher level
_LEVEL_RANK: dict[LegalLevel, int] = {
    LegalLevel.PART: 1,
    LegalLevel.CHAPTER: 2,
    LegalLevel.ARTICLE: 3,
    LegalLevel.SECTION: 4,
    LegalLevel.CLAUSE: 5,
    LegalLevel.SUBCLAUSE: 6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class LegalSection(BaseModel):
    """A single node in the legal document hierarchy."""
    level: LegalLevel
    number: str = ""
    title: str = ""
    content: str = ""
    children: list["LegalSection"] = Field(default_factory=list)

    class Config:
        # Needed for self-referential model in Pydantic v2
        arbitrary_types_allowed = True


LegalSection.model_rebuild()


class NormalizedDocument(BaseModel):
    """Standardised representation of any regulatory document."""
    title: str = ""
    body: str = ""
    sections: list[LegalSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: Optional[str] = None
    country: str = ""
    doc_type: str = ""
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for legal hierarchy detection
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (LegalLevel, compiled pattern, group index for number, group index for title)
_SECTION_PATTERNS: list[tuple[LegalLevel, re.Pattern, int, int]] = [
    (
        LegalLevel.PART,
        re.compile(
            r"^(?:PART|Part)\s+([IVXLCDM\d]+)[\.\:]?\s*(.*)?$",
            re.MULTILINE,
        ),
        1, 2,
    ),
    (
        LegalLevel.CHAPTER,
        re.compile(
            r"^(?:CHAPTER|Chapter|Kapitel|Chapitre|Capítulo|Capitolo)\s+([IVXLCDM\d]+)[\.\:]?\s*(.*)?$",
            re.MULTILINE,
        ),
        1, 2,
    ),
    (
        LegalLevel.ARTICLE,
        re.compile(
            r"^(?:Article|ARTICLE|Artikel|Artículo|Article|Статья|Стаття)\s+(\d+[\w\.\-]*)[\.\:]?\s*(.*)?$",
            re.MULTILINE,
        ),
        1, 2,
    ),
    (
        LegalLevel.SECTION,
        re.compile(
            r"^(?:Section|SECTION|§\s*)\s*(\d+[\w\.\-]*)[\.\:]?\s*(.*)?$",
            re.MULTILINE,
        ),
        1, 2,
    ),
    (
        LegalLevel.CLAUSE,
        re.compile(
            r"^(\d+\.\d+)[\.\s]+([A-Z][^\n]{0,80})$",
            re.MULTILINE,
        ),
        1, 2,
    ),
    (
        LegalLevel.SUBCLAUSE,
        re.compile(
            r"^(\d+\.\d+\.\d+)[\.\s]+([^\n]{0,80})$",
            re.MULTILINE,
        ),
        1, 2,
    ),
]

# Additional patterns for bracket clauses: (a), (b), (1), (2)
_BRACKET_CLAUSE_RE = re.compile(
    r"^\s*\(([a-z\d]+)\)\s+(.+)$",
    re.MULTILINE,
)

# § symbol followed by a section number (CFR, Swiss, German style)
_PARAGRAPH_RE = re.compile(
    r"§\s*(\d+[\w\.\-]*)\s*([^\n]{0,100})?",
    re.MULTILINE,
)

# FHIR resourceType detection
_FHIR_RESOURCE_TYPES = {
    "Bundle", "Composition", "DocumentReference", "Binary",
    "Device", "DeviceDefinition", "Organization", "Practitioner",
    "Medication", "MedicationKnowledge", "Regulatory",
}


# ─────────────────────────────────────────────────────────────────────────────
# DocumentNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class DocumentNormalizer:
    """
    Normalises regulatory documents from any format into a standard structure.
    Preserves legal hierarchy (HC-3: Article/Section/Clause boundaries).

    Input formats handled:
    - FHIR/HL7 JSON (Swissdamed M2M, ANVISA SIUD)
    - HTML (TGA, FDA, EUDAMED web scraping)
    - Plain text regulatory documents
    - JSON API responses (GUDID, MDALL)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(
        self,
        raw: Any,
        source_type: str,
        metadata: dict[str, Any],
    ) -> NormalizedDocument:
        """
        Normalise any raw input into a NormalizedDocument.

        Args:
            raw:         The raw document data (str, dict, bytes, or list).
            source_type: Hint for format: "html", "fhir", "json", "plain", or "auto".
            metadata:    Caller-supplied metadata dict (country, source_url, etc.).

        Returns:
            NormalizedDocument with title, body, sections, and metadata.
        """
        detected = self.detect_content_type(raw) if source_type == "auto" else source_type

        try:
            if detected == "fhir":
                body, title = self._process_fhir(raw)
            elif detected == "html":
                body, title = self._process_html(raw)
            elif detected == "json":
                body, title = self._process_json(raw)
            else:
                body, title = self._process_plain(raw)
        except Exception as exc:
            logger.warning("Normalizer failed for source_type=%s: %s", detected, exc)
            body = str(raw) if not isinstance(raw, (bytes, bytearray)) else raw.decode("utf-8", errors="replace")
            title = metadata.get("title", "")

        # Override title from metadata if richer
        if not title:
            title = (
                metadata.get("title")
                or metadata.get("regulation_name")
                or metadata.get("name")
                or ""
            )

        sections = self.extract_legal_sections(body)

        return NormalizedDocument(
            title=title,
            body=body,
            sections=sections,
            metadata=metadata,
            source_url=metadata.get("source_url") or metadata.get("url"),
            country=metadata.get("country", ""),
            doc_type=detected,
        )

    def detect_content_type(self, raw: Any) -> str:
        """
        Heuristically detect the content type of the raw input.

        Returns one of: "fhir", "html", "json", "plain".
        """
        if isinstance(raw, (dict, list)):
            # Already parsed — check for FHIR resourceType
            if isinstance(raw, dict) and raw.get("resourceType") in _FHIR_RESOURCE_TYPES:
                return "fhir"
            return "json"

        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8", errors="replace")
            except Exception:
                return "plain"

        if not isinstance(raw, str):
            return "plain"

        stripped = raw.lstrip()

        # JSON?
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("resourceType") in _FHIR_RESOURCE_TYPES:
                    return "fhir"
                return "json"
            except json.JSONDecodeError:
                pass

        # HTML?
        if re.search(r"<(?:html|head|body|div|p|span|table|ul|ol|li|h[1-6])\b", stripped, re.IGNORECASE):
            return "html"

        # FHIR JSON in string form
        if '"resourceType"' in raw and any(rt in raw for rt in _FHIR_RESOURCE_TYPES):
            return "fhir"

        return "plain"

    def extract_legal_sections(self, text: str) -> list[LegalSection]:
        """
        Extract legal sections from normalised text using hierarchy-aware patterns.

        Returns a flat list of LegalSection objects ordered by appearance.
        The hierarchy (parent/child nesting) is resolved via rank comparison.
        """
        if not text or not text.strip():
            return []

        # Collect all matches across all patterns
        raw_sections: list[tuple[int, LegalLevel, str, str]] = []  # (offset, level, number, title)

        for level, pattern, num_group, title_group in _SECTION_PATTERNS:
            for match in pattern.finditer(text):
                number = match.group(num_group).strip() if match.lastindex and match.lastindex >= num_group else ""
                title = match.group(title_group).strip() if match.lastindex and match.lastindex >= title_group and match.group(title_group) else ""
                raw_sections.append((match.start(), level, number, title))

        # Also extract § clauses
        for match in _PARAGRAPH_RE.finditer(text):
            number = match.group(1).strip()
            title = (match.group(2) or "").strip()
            raw_sections.append((match.start(), LegalLevel.SECTION, number, title))

        if not raw_sections:
            return []

        # Sort by document offset
        raw_sections.sort(key=lambda x: x[0])

        # Deduplicate (same offset may match multiple patterns — keep highest-level)
        deduped: list[tuple[int, LegalLevel, str, str]] = []
        seen_offsets: set[int] = set()
        for offset, level, number, title in raw_sections:
            # Allow a tolerance window of 5 chars to deduplicate near-identical matches
            bucket = offset // 5
            if bucket not in seen_offsets:
                seen_offsets.add(bucket)
                deduped.append((offset, level, number, title))

        # Slice text into section content blocks
        sections: list[LegalSection] = []
        for i, (offset, level, number, title) in enumerate(deduped):
            next_offset = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            content = text[offset:next_offset].strip()
            # Remove the header line from content body
            content_lines = content.split("\n")
            content = "\n".join(content_lines[1:]).strip() if len(content_lines) > 1 else ""
            sections.append(
                LegalSection(
                    level=level,
                    number=number,
                    title=title,
                    content=content,
                )
            )

        # Build parent-child hierarchy in-place (children list only, no recursive tree needed
        # for chunking purposes — this remains a flat list with level metadata)
        return sections

    # ── Internal format processors ────────────────────────────────────────────

    def _process_html(self, raw: Any) -> tuple[str, str]:
        """
        Strip HTML tags while preserving structural text.
        Headings → capitalised lines; <li> → bullet points; <p> → paragraphs.
        """
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        html_str: str = raw if isinstance(raw, str) else str(raw)
        return self.strip_html_preserve_structure(html_str), self._extract_html_title(html_str)

    def _process_plain(self, raw: Any) -> tuple[str, str]:
        """Handle plain text (already usable; apply light cleaning)."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        text = str(raw)
        text = self._clean_plain(text)
        # Try to find a title on the first non-empty line
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        title = lines[0] if lines else ""
        return text, title

    def _process_json(self, raw: Any) -> tuple[str, str]:
        """
        Convert a JSON API response (GUDID-style, MDALL-style) to readable text.
        """
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return raw, ""
        elif isinstance(raw, (dict, list)):
            data = raw
        else:
            return str(raw), ""

        parts: list[str] = []
        title = ""

        if isinstance(data, dict):
            title = (
                data.get("deviceDescription")
                or data.get("catalogNumber")
                or data.get("name")
                or data.get("title")
                or ""
            )
            parts = self._dict_to_text_parts(data, depth=0)
        elif isinstance(data, list):
            for item in data[:50]:  # Cap at 50 items
                if isinstance(item, dict):
                    parts.extend(self._dict_to_text_parts(item, depth=0))
                    parts.append("")

        body = "\n".join(parts)
        return body, str(title)

    def _process_fhir(self, raw: Any) -> tuple[str, str]:
        """
        Convert a FHIR/HL7 resource or Bundle to plain regulatory text.
        """
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return raw, ""
        elif isinstance(raw, (dict, list)):
            data = raw
        else:
            return str(raw), ""

        text = self.fhir_to_text(data)
        title = ""
        if isinstance(data, dict):
            # Extract title from FHIR Composition or DocumentReference
            title = (
                data.get("title")
                or data.get("description")
                or (data.get("name", [{}])[0].get("text", "") if isinstance(data.get("name"), list) else "")
                or ""
            )
        return text, str(title)

    # ── Format-specific helpers ───────────────────────────────────────────────

    def strip_html_preserve_structure(self, html: str) -> str:
        """
        Convert HTML to clean text preserving document structure.

        - Headings (h1–h6) → UPPERCASE lines
        - List items (<li>) → "- item" bullets
        - Paragraphs (<p>), <br> → newlines
        - Tables → tab-delimited rows
        - Scripts, styles, nav, footer → removed
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "noscript", "iframe", "form"]):
                tag.decompose()

            lines: list[str] = []

            def _walk(node: Any) -> None:
                if isinstance(node, str):
                    text = node.strip()
                    if text:
                        lines.append(text)
                    return
                if not hasattr(node, "name") or node.name is None:
                    return

                name = node.name.lower()

                if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    text = node.get_text(separator=" ", strip=True).upper()
                    if text:
                        lines.append("")
                        lines.append(text)
                        lines.append("")
                elif name in ("p", "div", "section", "article", "main"):
                    # Process children
                    for child in node.children:
                        _walk(child)
                    lines.append("")
                elif name == "li":
                    text = node.get_text(separator=" ", strip=True)
                    if text:
                        lines.append(f"- {text}")
                elif name in ("ul", "ol"):
                    lines.append("")
                    for child in node.children:
                        _walk(child)
                    lines.append("")
                elif name == "br":
                    lines.append("")
                elif name in ("td", "th"):
                    text = node.get_text(separator=" ", strip=True)
                    lines.append(text + "\t")
                elif name == "tr":
                    row_parts: list[str] = []
                    for child in node.children:
                        if hasattr(child, "name") and child.name in ("td", "th"):
                            row_parts.append(child.get_text(separator=" ", strip=True))
                    if row_parts:
                        lines.append(" | ".join(row_parts))
                elif name in ("table", "tbody", "thead"):
                    for child in node.children:
                        _walk(child)
                    lines.append("")
                elif name == "a":
                    text = node.get_text(separator=" ", strip=True)
                    href = node.get("href", "")
                    if text:
                        lines.append(f"{text} [{href}]" if href else text)
                else:
                    for child in node.children:
                        _walk(child)

            _walk(soup)

            # Collapse blank lines
            result_lines: list[str] = []
            blank_count = 0
            for line in lines:
                if not line.strip():
                    blank_count += 1
                    if blank_count <= 2:
                        result_lines.append("")
                else:
                    blank_count = 0
                    result_lines.append(line.rstrip())

            return "\n".join(result_lines).strip()

        except ImportError:
            # BeautifulSoup not available — regex fallback
            return self._strip_html_regex(html)

    def _strip_html_regex(self, html: str) -> str:
        """Minimal regex-based HTML tag stripper (no BeautifulSoup)."""
        # Remove script and style blocks
        html = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", html,
                      flags=re.DOTALL | re.IGNORECASE)
        # Headings → uppercase + newlines
        html = re.sub(
            r"<h[1-6][^>]*>(.*?)</h[1-6]>",
            lambda m: f"\n\n{m.group(1).upper()}\n\n",
            html, flags=re.DOTALL | re.IGNORECASE,
        )
        # List items → bullets
        html = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: f"\n- {m.group(1)}", html,
                      flags=re.DOTALL | re.IGNORECASE)
        # Paragraph / div breaks
        html = re.sub(r"<(?:p|div|br)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Strip remaining tags
        html = re.sub(r"<[^>]+>", " ", html)
        # Decode common entities
        html = html.replace("&amp;", "&").replace("&lt;", "<").replace(
            "&gt;", ">").replace("&nbsp;", " ").replace("&quot;", '"')
        # Collapse whitespace
        html = re.sub(r"\n{3,}", "\n\n", html)
        html = re.sub(r" {2,}", " ", html)
        return html.strip()

    def fhir_to_text(self, fhir_resource: dict) -> str:
        """
        Convert a FHIR resource or Bundle to plain regulatory text.

        Handles:
        - Bundle: recursively processes each entry.resource
        - Composition: extracts sections
        - DocumentReference: extracts content[].attachment.data or url
        - Device / DeviceDefinition: extracts device fields
        - Generic resources: key-value flattening
        """
        if not isinstance(fhir_resource, dict):
            return str(fhir_resource)

        resource_type = fhir_resource.get("resourceType", "Unknown")
        parts: list[str] = []

        if resource_type == "Bundle":
            parts.append(f"FHIR Bundle — type: {fhir_resource.get('type', '')}")
            for entry in fhir_resource.get("entry", []):
                resource = entry.get("resource", {})
                if isinstance(resource, dict):
                    parts.append("")
                    parts.append(self.fhir_to_text(resource))

        elif resource_type == "Composition":
            parts.append(f"Composition: {fhir_resource.get('title', '')}")
            parts.append(f"Status: {fhir_resource.get('status', '')}")
            parts.append(f"Date: {fhir_resource.get('date', '')}")
            for section in fhir_resource.get("section", []):
                title = section.get("title", "")
                if title:
                    parts.append(f"\n{title.upper()}")
                for entry_item in section.get("entry", []):
                    parts.append(f"  Reference: {entry_item.get('reference', '')}")
                text_node = section.get("text", {})
                if text_node:
                    div = text_node.get("div", "")
                    if div:
                        parts.append(self.strip_html_preserve_structure(div))
                for child in section.get("section", []):
                    child_title = child.get("title", "")
                    if child_title:
                        parts.append(f"  {child_title}")

        elif resource_type == "DocumentReference":
            parts.append(f"Document Reference: {fhir_resource.get('description', '')}")
            parts.append(f"Status: {fhir_resource.get('status', '')}")
            for content in fhir_resource.get("content", []):
                attachment = content.get("attachment", {})
                content_type = attachment.get("contentType", "")
                data = attachment.get("data", "")
                url = attachment.get("url", "")
                title = attachment.get("title", "")
                if title:
                    parts.append(f"Attachment: {title}")
                if url:
                    parts.append(f"URL: {url}")
                if data and content_type in ("text/plain", "text/html"):
                    import base64
                    try:
                        decoded = base64.b64decode(data).decode("utf-8", errors="replace")
                        if "html" in content_type.lower():
                            decoded = self.strip_html_preserve_structure(decoded)
                        parts.append(decoded)
                    except Exception:
                        parts.append(f"[binary data, {len(data)} chars encoded]")

        elif resource_type in ("Device", "DeviceDefinition"):
            parts.append(f"Device: {fhir_resource.get('deviceName', [{}])[0].get('name', '') if isinstance(fhir_resource.get('deviceName'), list) else ''}")
            parts.append(f"Status: {fhir_resource.get('status', '')}")
            manufacturer = fhir_resource.get("manufacturer", "")
            if isinstance(manufacturer, dict):
                manufacturer = manufacturer.get("display", "")
            parts.append(f"Manufacturer: {manufacturer}")
            for prop in fhir_resource.get("property", []):
                code = prop.get("type", {}).get("text", "")
                value = prop.get("valueCode", {}).get("text", "") if isinstance(prop.get("valueCode"), dict) else prop.get("valueString", "")
                if code:
                    parts.append(f"{code}: {value}")

        else:
            # Generic: flatten key-value pairs
            parts.extend(self._dict_to_text_parts(fhir_resource, depth=0))

        return "\n".join(parts).strip()

    # ── Private utilities ─────────────────────────────────────────────────────

    def _extract_html_title(self, html: str) -> str:
        """Extract the <title> or first <h1> from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r"<[^>]+>", "", match.group(1)).strip()
        match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return ""

    def _clean_plain(self, text: str) -> str:
        """Light cleaning for plain text: collapse excessive whitespace."""
        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3+ blank lines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace on each line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _dict_to_text_parts(self, d: dict, depth: int) -> list[str]:
        """
        Recursively flatten a dict to readable key: value lines.
        Depth-limited to prevent infinite recursion on deep structures.
        """
        if depth > 5:
            return [str(d)]

        parts: list[str] = []
        indent = "  " * depth

        for key, val in d.items():
            if key.startswith("_") or key in ("resourceType",):
                continue
            if isinstance(val, dict):
                parts.append(f"{indent}{key}:")
                parts.extend(self._dict_to_text_parts(val, depth + 1))
            elif isinstance(val, list):
                if not val:
                    continue
                if all(isinstance(v, (str, int, float, bool)) for v in val):
                    parts.append(f"{indent}{key}: {', '.join(str(v) for v in val)}")
                else:
                    parts.append(f"{indent}{key}:")
                    for item in val[:20]:  # Cap list expansion
                        if isinstance(item, dict):
                            parts.extend(self._dict_to_text_parts(item, depth + 1))
                        else:
                            parts.append(f"{indent}  - {item}")
            elif isinstance(val, bool):
                parts.append(f"{indent}{key}: {'yes' if val else 'no'}")
            elif val is not None and str(val).strip():
                parts.append(f"{indent}{key}: {val}")

        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

_normalizer: Optional[DocumentNormalizer] = None


def get_normalizer() -> DocumentNormalizer:
    """Return the module-level DocumentNormalizer singleton."""
    global _normalizer
    if _normalizer is None:
        _normalizer = DocumentNormalizer()
    return _normalizer
