"""In-memory PDF extraction and deterministic, exact source passages."""

from io import BytesIO

from pypdf import PdfReader


class PDFError(ValueError):
    """An upload cannot be used as a text paper."""


def extract_pdf(data: bytes, max_upload_mb: int = 10,
                max_text_chars: int = 20_000, mode: str = "layout") -> dict[int, str]:
    """Return physical page numbers and text. Reject limits, never truncate."""
    if mode not in {"plain", "layout"}:
        raise PDFError("PDF extraction mode must be plain or layout.")
    if max_upload_mb <= 0 or max_text_chars <= 0:
        raise PDFError("PDF limits must be positive.")
    if not data:
        raise PDFError("The uploaded file is empty.")
    if len(data) > max_upload_mb * 1024 * 1024:
        raise PDFError(f"Upload a PDF smaller than {max_upload_mb} MB.")
    if not data.startswith(b"%PDF-"):
        raise PDFError("This file does not appear to be a PDF.")
    pages = {}
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise PDFError("Encrypted PDFs are unsupported. Upload an unencrypted copy.")
        total = 0
        for number, page in enumerate(reader.pages, 1):
            # Fixed-width layout preserves horizontal positions useful in tables.
            # It remains extracted text, not table/figure or equation interpretation.
            options = {'extraction_mode': mode}
            if mode == 'layout':
                options['layout_mode_space_vertically'] = False
            text = (page.extract_text(**options) or "").strip()
            total += len(text)
            if total > max_text_chars:
                raise PDFError(f"Extracted text exceeds {max_text_chars:,} characters. "
                               "Provide a shorter PDF; text has not been truncated.")
            pages[number] = text
    except PDFError:
        raise
    except Exception:
        raise PDFError("The PDF is unreadable or damaged. Export a fresh PDF and retry.") from None
    if sum(char.isalnum() for text in pages.values() for char in text) < 50:
        raise PDFError("Insufficient extractable text. The PDF may be scanned or empty. "
                       "Upload an English PDF with embedded text; OCR is not supported.")
    return pages


def prepare_passages(pages: dict[int, str], passage_chars: int = 800) -> dict[str, dict]:
    """Slice exact substrings, retaining offsets. IDs are assigned only by Python."""
    if not 1 <= passage_chars <= 800:
        raise ValueError("Passage size must be between 1 and 800 characters.")
    evidence = {}
    for page, text in pages.items():
        for start in range(0, len(text), passage_chars):
            passage = text[start:start + passage_chars]
            if passage.strip():
                evidence_id = f"E{len(evidence) + 1:04d}"
                evidence[evidence_id] = {"evidence_id": evidence_id, "page": page,
                                         "text": passage, "start": start,
                                         "end": start + len(passage)}
    return evidence


def labeled_text(pages: dict[int, str]) -> str:
    return "\n\n".join(f"[PDF page {number}]\n{text}" for number, text in pages.items())
