"""Layout-Aware Visual Document Extractor with Structured Markdown Tables and OCR.

Extracts text, preserves document hierarchy, formats tables into Markdown tables,
and applies OCR on scanned pages or image regions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from src.core.config import get_settings
from src.core.exceptions import DocumentParsingError
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedSection:
    """Represents an extracted structural page or section from a document."""
    content: str
    page_number: int
    source_name: str
    has_tables: bool = False
    has_images: bool = False
    section_title: str | None = None
    metadata: dict[str, Any] | None = None


def table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a 2D list of table cells into a clean Markdown table."""
    if not table or not any(table):
        return ""

    # Clean None and newlines in cells
    cleaned: list[list[str]] = [
        [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        for row in table
    ]

    # Filter out empty rows
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    # Normalize column count
    max_cols = max(len(row) for row in cleaned)
    padded = [row + [""] * (max_cols - len(row)) for row in cleaned]

    header = padded[0]
    separator = ["---"] * max_cols
    body = padded[1:] if len(padded) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n" + "\n".join(lines) + "\n"


class VisualDocumentExtractor:
    """Extractor for rich documents (PDFs, Markdown, Text, CSV)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def extract_document(self, file_path: Path | str) -> list[ExtractedSection]:
        """Extract sections, text, and tables from a file path."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentParsingError(f"File not found: {path}")

        suffix = path.suffix.lower()
        logger.info(f"Extracting document '{path.name}' (type: {suffix})")

        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix in (".txt", ".md", ".markdown"):
            return self._extract_text(path)
        elif suffix == ".csv":
            return self._extract_csv(path)
        else:
            # Fallback to plain text read
            return self._extract_text(path)

    def _extract_pdf(self, path: Path) -> list[ExtractedSection]:
        """Extract PDF with layout and table preservation."""
        sections: list[ExtractedSection] = []

        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    has_tables = False
                    page_text = page.extract_text() or ""

                    # Extract tables if configured
                    if self.settings.ingestion.extract_markdown_tables:
                        tables = page.extract_tables()
                        if tables:
                            has_tables = True
                            for tbl in tables:
                                md_tbl = table_to_markdown(tbl)
                                if md_tbl:
                                    page_text += f"\n\n{md_tbl}\n"

                    # OCR Fallback for scanned pages
                    if not page_text.strip() and self.settings.ingestion.enable_ocr:
                        page_text = self._ocr_page(page)

                    if page_text.strip():
                        sections.append(
                            ExtractedSection(
                                content=page_text.strip(),
                                page_number=idx,
                                source_name=path.name,
                                has_tables=has_tables,
                                has_images=len(page.images) > 0,
                            )
                        )

            logger.info(f"✔ Extracted {len(sections)} pages from PDF '{path.name}'")
            return sections

        except ImportError:
            # Fallback to PyMuPDF (fitz)
            return self._extract_pdf_pymupdf(path)
        except Exception as err:
            raise DocumentParsingError(f"Failed to parse PDF '{path.name}': {err}") from err

    def _extract_pdf_pymupdf(self, path: Path) -> list[ExtractedSection]:
        """PyMuPDF fallback extractor."""
        try:
            import fitz
            doc = fitz.open(path)
            sections: list[ExtractedSection] = []
            for idx, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    sections.append(
                        ExtractedSection(
                            content=text.strip(),
                            page_number=idx,
                            source_name=path.name,
                        )
                    )
            return sections
        except Exception as err:
            raise DocumentParsingError(f"PyMuPDF failed on '{path.name}': {err}") from err

    def _ocr_page(self, page: Any) -> str:
        """Run OCR on page image."""
        try:
            import pytesseract
            img = page.to_image(resolution=200).original
            ocr_text = pytesseract.image_to_string(img, lang=self.settings.ingestion.ocr_language)
            return ocr_text.strip()
        except Exception as err:
            logger.warning(f"OCR failed for page: {err}")
            return ""

    def _extract_text(self, path: Path) -> list[ExtractedSection]:
        """Read plain text or markdown file."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [
                ExtractedSection(
                    content=content.strip(),
                    page_number=1,
                    source_name=path.name,
                )
            ]
        except Exception as err:
            raise DocumentParsingError(f"Failed to read text file '{path.name}': {err}") from err

    def _extract_csv(self, path: Path) -> list[ExtractedSection]:
        """Read CSV and represent as Markdown table."""
        try:
            import csv
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = list(csv.reader(f))
            md_table = table_to_markdown(reader)
            return [
                ExtractedSection(
                    content=md_table.strip(),
                    page_number=1,
                    source_name=path.name,
                    has_tables=True,
                )
            ]
        except Exception as err:
            raise DocumentParsingError(f"Failed to parse CSV '{path.name}': {err}") from err
