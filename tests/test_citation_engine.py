from utils.citation_engine import CitationResolutionService


def test_highlight_coordinates_for_known_text(sample_multi_page_pdf):
    """
    Test 3: Verify highlight coordinates for known text passages on deterministic fixture PDF.
    """
    # On Page 1, search for 'autonomous agent architectures'
    rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=1,
        snippet="autonomous agent architectures"
    )

    assert len(rects) >= 1
    r = rects[0]
    # Check bounding box format [x0, y0, x1, y1]
    assert len(r) == 4
    x0, y0, x1, y1 = r
    # Coordinates were placed near Point(72, 130) on Page 1
    assert 60 <= x0 <= 200
    assert 100 <= y0 <= 150
    assert x1 > x0
    assert y1 > y0


def test_correct_source_to_page_mapping(sample_multi_page_pdf):
    """
    Test 2: Ensure source text mapped to page N resolves on page N, but does NOT resolve on page M.
    """
    # 'Section 2 focuses on hybrid retrieval' is ONLY on Page 2
    rects_on_p2 = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=2,
        snippet="Section 2 focuses on hybrid retrieval"
    )
    assert len(rects_on_p2) >= 1

    # Should NOT be found on Page 1 or Page 3
    rects_on_p1 = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=1,
        snippet="Section 2 focuses on hybrid retrieval"
    )
    assert len(rects_on_p1) == 0

    rects_on_p3 = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=3,
        snippet="Section 2 focuses on hybrid retrieval"
    )
    assert len(rects_on_p3) == 0


def test_multiple_citations_from_different_pages(sample_multi_page_pdf):
    """
    Test 4: Verify multiple citations from different pages resolve correctly to their respective pages.
    """
    citations = [
        {"id": 1, "page": 1, "snippet": "Section 1 covers autonomous agent architectures"},
        {"id": 2, "page": 2, "snippet": "Reciprocal rank fusion reconciles sparse and dense"},
        {"id": 3, "page": 3, "snippet": "Multi-document environments require dedicated isolation"}
    ]

    resolved_citations = []
    for cit in citations:
        rects = CitationResolutionService.resolve_citation_rects(
            pdf_path=sample_multi_page_pdf,
            page_number=cit["page"],
            snippet=cit["snippet"]
        )
        assert len(rects) >= 1, f"Failed to resolve citation on page {cit['page']}"
        resolved_citations.append({
            "citation_id": cit["id"],
            "page_number": cit["page"],
            "rects": rects
        })

    assert len(resolved_citations) == 3
    assert [c["page_number"] for c in resolved_citations] == [1, 2, 3]


def test_unverified_position_fallback(sample_multi_page_pdf):
    """
    Verify that when exact text positioning cannot be established,
    an empty list of rects is returned instead of misleading highlight boxes.
    """
    rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=1,
        snippet="completely hallucinated sentence that does not exist anywhere in this document"
    )
    assert rects == []


def test_viewer_annotations_formatting():
    """
    Verify conversion of bounding boxes into viewer annotations format.
    """
    sample_rects = [[72.0, 100.0, 250.0, 120.0], [72.0, 125.0, 300.0, 145.0]]
    annots = CitationResolutionService.to_viewer_annotations(
        rects=sample_rects,
        page_number=2,
        color="#f59e0b"
    )

    assert len(annots) == 2
    assert annots[0]["page"] == 2
    assert annots[0]["x"] == 72.0
    assert annots[0]["y"] == 100.0
    assert annots[0]["width"] == 178.0
    assert annots[0]["height"] == 20.0
    assert annots[0]["color"] == "#f59e0b"
    assert annots[0]["id"] == "hl_2_0"


def test_render_page_with_highlights(sample_multi_page_pdf):
    """
    Verify high-resolution PNG rendering with and without highlights.
    """
    # 1. Clean page
    clean_bytes = CitationResolutionService.render_page_with_highlights(
        pdf_path=sample_multi_page_pdf,
        page_number=1,
        rects=None,
        zoom=1.0
    )
    assert clean_bytes is not None
    assert len(clean_bytes) > 1000

    # 2. Highlighted page
    highlight_rects = [[72.0, 100.0, 200.0, 120.0]]
    hl_bytes = CitationResolutionService.render_page_with_highlights(
        pdf_path=sample_multi_page_pdf,
        page_number=1,
        rects=highlight_rects,
        zoom=1.0
    )
    assert hl_bytes is not None
    assert len(hl_bytes) > 1000
    # The highlighted image has extra graphical shapes, byte sizes should differ
    assert hl_bytes != clean_bytes
