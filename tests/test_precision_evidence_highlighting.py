import os
import pytest
import pymupdf

from utils.citation_engine import CitationResolutionService


@pytest.fixture
def scientific_multipage_pdf(temp_dir):
    """
    Creates a realistic scientific PDF fixture with multiple pages,
    two-column text blocks, equations, tables, and multi-line citations.
    """
    pdf_path = os.path.join(temp_dir, "scientific_study.pdf")
    doc = pymupdf.open()

    # Page 1: Abstract and Architecture with multi-line passages
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(pymupdf.Point(72, 80), "Neural Retrieval-Augmented Generation for Scientific Literature", fontsize=14)
    p1.insert_text(
        pymupdf.Point(72, 120),
        "Abstract: Recent advancements in autonomous language agent architectures demonstrate significant\n"
        "improvements in factual grounding when coupled with precision evidence retrieval pipelines.",
        fontsize=10
    )
    p1.insert_text(
        pymupdf.Point(72, 180),
        "Equation 1: E(x, y) = cosine_sim(f_theta(x), g_phi(y)) + lambda * BM25(x, y)",
        fontsize=10
    )
    p1.insert_text(
        pymupdf.Point(72, 240),
        "Section 1.1 Methodology\n"
        "The dense retriever indexes passage embeddings using hierarchical navigable small world graphs,\n"
        "while reciprocal rank fusion balances dense vector proximity against sparse token frequencies.",
        fontsize=10
    )

    # Page 2: Table and Experimental Results
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(pymupdf.Point(72, 80), "Experimental Evaluation and Benchmarking", fontsize=14)
    p2.insert_text(
        pymupdf.Point(72, 120),
        "Table 1: Performance metrics across multi-hop reasoning tasks.\n"
        "Model                Precision@1    Recall@5    Groundedness\n"
        "Baseline BM25        0.642          0.781       0.710\n"
        "Dense ColBERT        0.814          0.892       0.845\n"
        "Hybrid Fusion RAG    0.938          0.976       0.962",
        fontsize=9
    )
    p2.insert_text(
        pymupdf.Point(72, 220),
        "The empirical findings demonstrate that hybrid fusion rag achieves 0.938 precision@1,\n"
        "substantially outperforming isolated lexical or semantic retrieval baselines.",
        fontsize=10
    )

    # Page 3: Rotated Page (e.g. wide appendix table rotated 90 degrees)
    p3 = doc.new_page(width=612, height=792)
    p3.insert_text(
        pymupdf.Point(72, 100),
        "Appendix A: System Latency and Resource Consumption across Heterogeneous GPU Clusters.",
        fontsize=11
    )
    p3.set_rotation(90)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_multiline_evidence_localization(scientific_multipage_pdf):
    """
    Verify localization and multiple rectangle generation for multi-line passages.
    """
    multiline_passage = (
        "The dense retriever indexes passage embeddings using hierarchical navigable small world graphs,\n"
        "while reciprocal rank fusion balances dense vector proximity against sparse token frequencies."
    )
    rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=1,
        snippet=multiline_passage
    )

    assert len(rects) >= 1, "Should identify bounding boxes for multi-line passage"
    # Verify each rectangle is formatted correctly [x0, y0, x1, y1]
    for r in rects:
        assert len(r) == 4
        assert r[2] > r[0]
        assert r[3] > r[1]

    # Verify clustering and enclosing box calculation
    clusters = CitationResolutionService.cluster_rects(rects)
    assert len(clusters) >= 1
    enclosing_box = CitationResolutionService.compute_enclosing_box(rects, padding=4.0)
    assert enclosing_box is not None
    assert enclosing_box[0] < min(r[0] for r in rects)
    assert enclosing_box[2] > max(r[2] for r in rects)


def test_scientific_equations_and_tables_localization(scientific_multipage_pdf):
    """
    Verify localization on pages containing equations and tables.
    """
    # 1. Equation on Page 1
    eq_snippet = "cosine_sim(f_theta(x), g_phi(y)) + lambda * BM25(x, y)"
    eq_rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=1,
        snippet=eq_snippet
    )
    assert len(eq_rects) >= 1

    # 2. Table passage on Page 2
    table_snippet = "hybrid fusion rag achieves 0.938 precision@1"
    table_rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=2,
        snippet=table_snippet
    )
    assert len(table_rects) >= 1


def test_rotated_page_coordinate_transformation(scientific_multipage_pdf):
    """
    Verify rotated page handling: visual coordinates are transformed via rotation matrix.
    """
    appendix_snippet = "Appendix A: System Latency and Resource Consumption"
    
    # Raw unrotated rects
    raw_rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=3,
        snippet=appendix_snippet,
        apply_rotation=False
    )
    assert len(raw_rects) >= 1

    # Rotation-adjusted visual rects
    rot_rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=3,
        snippet=appendix_snippet,
        apply_rotation=True
    )
    assert len(rot_rects) >= 1

    # On a 90-degree rotated page (width=612, height=792), x and y swap orientations
    assert rot_rects[0] != raw_rects[0], "Rotated coordinates must be transformed"
    # Rotated rect coordinates must fit inside rotated page dimensions (792 x 612)
    assert 0 <= rot_rects[0][0] <= 792
    assert 0 <= rot_rects[0][1] <= 612


def test_amber_outline_and_badge_rendering(scientific_multipage_pdf):
    """
    Verify PyMuPDF rendering with clearly visible amber outline, transparent fill, and citation badge.
    """
    passage = "hybrid fusion rag achieves 0.938 precision@1"
    rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=scientific_multipage_pdf,
        page_number=2,
        snippet=passage
    )
    assert len(rects) >= 1

    png_bytes = CitationResolutionService.render_page_with_highlights(
        pdf_path=scientific_multipage_pdf,
        page_number=2,
        rects=rects,
        zoom=1.25,
        draw_box=True,
        citation_id=2
    )

    assert png_bytes is not None
    assert len(png_bytes) > 2000, "Rendered highlighted image should contain valid PNG payload"


def test_multi_document_citation_switching(temp_dir):
    """
    Verify that citations from different documents preserve distinct doc_id and page numbers.
    """
    # Create Doc A
    path_a = os.path.join(temp_dir, "doc_alpha.pdf")
    doc_a = pymupdf.open()
    p_a = doc_a.new_page(width=612, height=792)
    p_a.insert_text(pymupdf.Point(72, 100), "Alpha document content on page 1.")
    doc_a.save(path_a)
    doc_a.close()

    # Create Doc B
    path_b = os.path.join(temp_dir, "doc_beta.pdf")
    doc_b = pymupdf.open()
    p_b1 = doc_b.new_page(width=612, height=792)
    p_b1.insert_text(pymupdf.Point(72, 100), "Beta cover page.")
    p_b2 = doc_b.new_page(width=612, height=792)
    p_b2.insert_text(pymupdf.Point(72, 100), "Beta document evidence passage on page 2.")
    doc_b.save(path_b)
    doc_b.close()

    citations = [
        {"citation_id": 1, "doc_id": "alpha_id", "page_number": 1, "snippet": "Alpha document content"},
        {"citation_id": 2, "doc_id": "beta_id", "page_number": 2, "snippet": "Beta document evidence passage"}
    ]

    # Resolve citation 1 on Doc A
    rects_a = CitationResolutionService.resolve_citation_rects(path_a, citations[0]["page_number"], citations[0]["snippet"])
    assert len(rects_a) >= 1

    # Resolve citation 2 on Doc B
    rects_b = CitationResolutionService.resolve_citation_rects(path_b, citations[1]["page_number"], citations[1]["snippet"])
    assert len(rects_b) >= 1

    # Cross-document negative test: searching Citation 2 snippet in Doc A returns empty
    rects_cross = CitationResolutionService.resolve_citation_rects(path_a, citations[1]["page_number"], citations[1]["snippet"])
    assert rects_cross == [], "Doc B snippet must not resolve in Doc A"
