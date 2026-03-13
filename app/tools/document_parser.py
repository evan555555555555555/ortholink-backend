"""
Multi-format document parser for DVA pipeline.

Supports:
  - CSV: Standard CSV parsing (existing logic)
  - PDF: Text extraction via pymupdf (fitz)
  - Excel (.xlsx, .xls): Row reading via openpyxl
  - TXT/Plain text: Line splitting

All formats return a list of extracted document item strings.
"""

import csv
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_document(content: bytes, filename: str) -> list[str]:
    """
    Parse uploaded document content into a list of document item strings.

    Args:
        content: Raw file bytes
        filename: Original filename (used to detect format)

    Returns:
        List of extracted document item strings (one per row/line/page)

    Raises:
        ValueError: If format is unsupported or content is empty
    """
    if not content:
        raise ValueError("Empty file content")

    suffix = Path(filename).suffix.lower() if filename else ""

    if suffix == ".csv":
        return _parse_csv(content)
    elif suffix == ".pdf":
        return _parse_pdf(content)
    elif suffix in (".xlsx", ".xls"):
        return _parse_excel(content)
    elif suffix in (".txt", ".text", ""):
        return _parse_text(content)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            "Supported formats: .csv, .pdf, .xlsx, .xls, .txt"
        )


def _parse_csv(content: bytes) -> list[str]:
    """Parse CSV content into document items."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    # Return as CSV text — the existing parse_csv() in verify_distributor.py handles column detection
    return [text]


def _parse_pdf(content: bytes) -> list[str]:
    """Extract text from PDF and return as line items."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ValueError(
            "PDF support requires pymupdf. Install with: pip install pymupdf"
        )

    items: list[str] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text = page.get_text()
            # Split into lines and filter empty ones
            for line in text.split("\n"):
                line = line.strip()
                if line and len(line) > 2:  # Skip very short fragments
                    items.append(line)
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")

    if not items:
        raise ValueError("No text content found in PDF")

    return items


def _parse_excel(content: bytes) -> list[str]:
    """Extract rows from Excel file and return as line items."""
    try:
        import openpyxl
    except ImportError:
        raise ValueError(
            "Excel support requires openpyxl. Install with: pip install openpyxl"
        )

    items: list[str] = []
    try:
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            raise ValueError("Excel file has no active worksheet")

        for row in ws.iter_rows(values_only=True):
            # Take first non-empty cell in each row as document item
            for cell in row:
                if cell is not None:
                    val = str(cell).strip()
                    if val and len(val) > 1:
                        items.append(val)
                        break  # Only first non-empty cell per row
        wb.close()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {e}")

    if not items:
        raise ValueError("No content found in Excel file")

    return items


def _parse_text(content: bytes) -> list[str]:
    """Parse plain text file into line items."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    items = [line.strip() for line in text.split("\n") if line.strip()]

    if not items:
        raise ValueError("No content found in text file")

    return items


def content_to_csv_text(items: list[str]) -> str:
    """Convert extracted items to CSV text format for the DVA pipeline."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["document"])
    for item in items:
        writer.writerow([item])
    return output.getvalue()
