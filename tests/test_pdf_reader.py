"""Generate a tiny original PDF with pypdf; no external fixture or OCR."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pdf_reader import PDFError, extract_pdf, labeled_text, prepare_passages


@pytest.fixture
def original_pdf():
    writer = PdfWriter()
    for text in ["Our original study compares two sorting methods on small inputs.",
                 "The study is limited to one machine. Larger experiments are future work."]:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_page_labels_and_original_text(original_pdf):
    pages = extract_pdf(original_pdf)
    text = labeled_text(pages)
    assert list(pages) == [1, 2]
    assert text.startswith("[PDF page 1]\nOur original study")
    assert "[PDF page 2]\nThe study is limited" in text


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.7\nbroken"])
def test_empty_invalid_or_damaged(data):
    with pytest.raises(PDFError):
        extract_pdf(data)


def test_scanned_or_blank():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    out = BytesIO()
    writer.write(out)
    with pytest.raises(PDFError, match="Insufficient extractable text"):
        extract_pdf(out.getvalue())


def test_limits_never_truncate(original_pdf):
    pages = extract_pdf(original_pdf)
    size = sum(map(len, pages.values()))
    assert extract_pdf(original_pdf, max_text_chars=size) == pages
    with pytest.raises(PDFError, match="shorter PDF"):
        extract_pdf(original_pdf, max_text_chars=size - 1)
    with pytest.raises(PDFError, match="smaller than"):
        extract_pdf(original_pdf + b" " * (1024 * 1024), max_upload_mb=1)


def test_encrypted_pdf(original_pdf):
    writer = PdfWriter(clone_from=BytesIO(original_pdf))
    writer.encrypt("test-password")
    out = BytesIO()
    writer.write(out)
    with pytest.raises(PDFError, match="Encrypted"):
        extract_pdf(out.getvalue())


def test_passages_are_exact_and_deterministic(original_pdf):
    pages = extract_pdf(original_pdf)
    registry = prepare_passages(pages, passage_chars=30)
    assert registry == prepare_passages(pages, passage_chars=30)
    assert len(registry) > 2
    for evidence_id, entry in registry.items():
        assert entry['evidence_id'] == evidence_id
        assert pages[entry['page']][entry['start']:entry['end']] == entry['text']
