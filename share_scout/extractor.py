"""Extract text content from files and compute content hashes."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(filepath: str, max_chars: int = 4000) -> str | None:
    """Extract text from a file based on its extension.

    Returns up to max_chars of text, or None if extraction fails.
    """
    ext = Path(filepath).suffix.lower()

    try:
        if ext in (".txt", ".md", ".mdx", ".csv", ".html", ".rtf", ".log",
                   ".rst", ".adoc", ".yaml", ".yml", ".toml", ".json",
                   ".py", ".ts", ".tsx", ".js", ".jsx", ".sql", ".css", ".sh"):
            return _extract_plain_text(filepath, max_chars)
        elif ext == ".pdf":
            return _extract_pdf(filepath, max_chars)
        elif ext in (".docx", ".doc"):
            return _extract_docx(filepath, max_chars)
        elif ext in (".xlsx", ".xls"):
            return _extract_xlsx(filepath, max_chars)
        else:
            logger.debug("No extractor for extension: %s", ext)
            return None
    except Exception as e:
        logger.warning("Extraction failed for %s: %s", filepath, e)
        return None


def compute_partial_hash(filepath: str, hash_bytes: int = 65536) -> str | None:
    """Compute SHA-256 hash of the first hash_bytes of a file."""
    try:
        with open(filepath, "rb") as f:
            data = f.read(hash_bytes)
        return hashlib.sha256(data).hexdigest()
    except (OSError, PermissionError) as e:
        logger.warning("Cannot hash %s: %s", filepath, e)
        return None


def _extract_plain_text(filepath: str, max_chars: int) -> str | None:
    """Read plain text with encoding detection."""
    # Try UTF-8 first, fall back to chardet
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except UnicodeDecodeError:
        pass

    try:
        import chardet
        with open(filepath, "rb") as f:
            raw = f.read(max_chars * 2)  # read more bytes to account for encoding
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "latin-1") or "latin-1"
        return raw.decode(encoding, errors="replace")[:max_chars]
    except Exception as e:
        logger.warning("Plain text extraction failed for %s: %s", filepath, e)
        return None


def _extract_pdf(filepath: str, max_chars: int) -> str | None:
    """Extract text from PDF using pymupdf."""
    import pymupdf

    text_parts = []
    total = 0
    with pymupdf.open(filepath) as doc:
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break

    text = "\n".join(text_parts)[:max_chars]
    return text if text.strip() else None


def _extract_docx(filepath: str, max_chars: int) -> str | None:
    """Extract text from .docx using python-docx."""
    from docx import Document

    doc = Document(filepath)
    text_parts = []
    total = 0
    for para in doc.paragraphs:
        text_parts.append(para.text)
        total += len(para.text)
        if total >= max_chars:
            break

    text = "\n".join(text_parts)[:max_chars]
    return text if text.strip() else None


def _extract_xlsx(filepath: str, max_chars: int) -> str | None:
    """Extract sheet names and header rows from .xlsx using openpyxl."""
    from openpyxl import load_workbook

    wb = load_workbook(filepath, read_only=True, data_only=True)
    text_parts = []
    total = 0

    for sheet_name in wb.sheetnames:
        text_parts.append(f"Sheet: {sheet_name}")
        total += len(sheet_name) + 8
        ws = wb[sheet_name]
        row_count = 0
        for row in ws.iter_rows(max_row=20, values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = " | ".join(cells)
            text_parts.append(line)
            total += len(line)
            row_count += 1
            if total >= max_chars:
                break
        if total >= max_chars:
            break

    wb.close()
    text = "\n".join(text_parts)[:max_chars]
    return text if text.strip() else None
