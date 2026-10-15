import pytest
from utils.pdf_loader import load_pdf, get_pdf_metadata


def test_pdf_ingestion_and_page_numbering(sample_multi_page_pdf):
    """
    Test 1: Verify PDF ingestion and exact 1-indexed page numbering across all pages.
    """
    pages = load_pdf(sample_multi_page_pdf)

    # Must contain 3 pages
    assert len(pages) == 3

    # Verify 1-indexed page numbering
    for expected_page, page_dict in enumerate(pages, 1):
        assert page_dict["page"] == expected_page
        assert page_dict["is_empty"] is False
        assert page_dict["char_count"] > 0
        assert page_dict["width"] == 612.0
        assert page_dict["height"] == 792.0

    # Verify content on each page
    assert "Section 1" in pages[0]["text"]
    assert "Section 2" in pages[1]["text"]
    assert "Section 3" in pages[2]["text"]


def test_pdf_metadata_extraction(sample_multi_page_pdf):
    """
    Verify document-level metadata extraction.
    """
    meta = get_pdf_metadata(sample_multi_page_pdf)
    assert meta["page_count"] == 3
    assert meta["is_scanned"] is False
    assert meta["total_chars"] > 50


def test_pdf_parsing_errors_and_scanned_pdfs(scanned_empty_pdf, corrupted_pdf):
    """
    Test 8: Verify robust error handling for missing files, corrupted PDFs, and scanned/empty PDFs.
    """
    # 1. Non-existent file
    with pytest.raises(ValueError, match="does not exist"):
        load_pdf("non_existent_file_path.pdf")

    # 2. Corrupted PDF file
    with pytest.raises(ValueError, match="Failed to open PDF"):
        load_pdf(corrupted_pdf)

    # 3. Scanned / empty PDF
    scanned_pages = load_pdf(scanned_empty_pdf)
    assert len(scanned_pages) == 1
    assert scanned_pages[0]["page"] == 1
    assert scanned_pages[0]["is_empty"] is True
    assert scanned_pages[0]["char_count"] == 0

    meta = get_pdf_metadata(scanned_empty_pdf)
    assert meta["page_count"] == 1
    assert meta["is_scanned"] is True
