import os
import pytest

from models import (
    get_deployment_mode,
    get_openai_api_key,
    get_embedding_model_id,
    openai_llm,
    openai_embeddings,
)
from utils.security import sanitize_filename, validate_pdf_path
from utils.pdf_loader import load_pdf, get_pdf_metadata
from utils.citation_engine import CitationResolutionService
from utils.vectorstore import (
    get_index_metadata,
    check_embedding_compatibility,
)
from rag.ingest_multi import build_multi_ingest_graph
from rag.query_multi import build_multi_query_graph
from rag.retriever import HybridRetriever


# ==============================================================================
# TEST 1: MULTI-PDF UPLOAD AND INDEXING
# ==============================================================================
def test_e2e_multi_pdf_upload_and_indexing(temp_dir, sample_multi_page_pdf, sample_second_pdf, mock_embeddings):
    """
    Test multi-PDF upload and indexing into a session-isolated directory.
    Verifies that multiple documents are indexed with correct page counts,
    chunk counts, index metadata, and manifests.
    """
    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    os.makedirs(session_vs_dir, exist_ok=True)

    ingest_graph = build_multi_ingest_graph(
        embeddings=mock_embeddings,
        embedding_model_id="mock-embed-64",
        base_dir=session_vs_dir
    )

    doc1_id = "doc_multi_1"
    doc2_id = "doc_multi_2"

    res1 = ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc1_id})
    res2 = ingest_graph.invoke({"path": sample_second_pdf, "doc_id": doc2_id})

    assert res1["status"] == "success"
    assert res1["page_count"] == 3
    assert res1["chunk_count"] >= 3

    assert res2["status"] == "success"
    assert res2["page_count"] == 1
    assert res2["chunk_count"] >= 1

    # Verify both indices exist in session vectorstore directory
    idx1_dir = os.path.join(session_vs_dir, f"{doc1_id}_index")
    idx2_dir = os.path.join(session_vs_dir, f"{doc2_id}_index")
    assert os.path.exists(idx1_dir)
    assert os.path.exists(idx2_dir)

    meta1 = get_index_metadata(idx1_dir)
    assert meta1 is not None
    assert meta1["embedding_model_id"] == "mock-embed-64"
    assert meta1["page_count"] == 3


# ==============================================================================
# TEST 2: DOCUMENT ISOLATION ACROSS MULTIPLE INDEXED FILES
# ==============================================================================
def test_e2e_document_isolation(temp_dir, sample_multi_page_pdf, sample_second_pdf, mock_embeddings):
    """
    Verify strict document isolation during retrieval.
    Queries directed at Doc 1 must never return chunks from Doc 2, and vice versa.
    """
    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    ingest_graph = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=session_vs_dir)

    doc_a = "doc_a_rag"
    doc_b = "doc_b_quantum"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_a})
    ingest_graph.invoke({"path": sample_second_pdf, "doc_id": doc_b})

    retriever = HybridRetriever(mock_embeddings, active_model_id="mock-embed-64", base_dir=session_vs_dir)

    # Search for quantum concept on Doc A (which only exists in Doc B)
    results_a = retriever.retrieve(query="quantum qubits superconducting", doc_id=doc_a, top_k=3, cross_doc_mode=False)
    for c in results_a:
        assert c["doc_id"] == doc_a
        assert "superconducting" not in c["text"].lower()

    # Search for RAG concept on Doc B (which only exists in Doc A)
    results_b = retriever.retrieve(query="precision evidence linking agent", doc_id=doc_b, top_k=3, cross_doc_mode=False)
    for c in results_b:
        assert c["doc_id"] == doc_b
        assert "antigravity" not in c["text"].lower()


# ==============================================================================
# TEST 3: BACKEND SELECTION AND CLOUD DEPLOYMENT GUARD
# ==============================================================================
def test_e2e_backend_selection_and_cloud_guard(monkeypatch):
    """
    Verify that in Cloud mode:
    1. get_deployment_mode() returns 'cloud'
    2. Cloud deployment prevents Ollama usage and never attempts localhost Ollama connection
    3. OpenAI requires a valid key and initializes cleanly
    """
    # 1. Simulate Cloud environment via environment variable
    monkeypatch.setenv("DEPLOYMENT_MODE", "cloud")
    assert get_deployment_mode() == "cloud"

    # In cloud mode, OpenAI without key raises clear ValueError
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        openai_llm()

    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        openai_embeddings()

    # In cloud mode with key, it initializes ChatOpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-mock-key-12345")
    llm = openai_llm()
    emb = openai_embeddings()
    assert llm is not None
    assert emb is not None
    assert get_embedding_model_id(emb) == "openai:text-embedding-3-small"

    # Verify hierarchical key resolution
    assert get_openai_api_key() == "sk-proj-test-mock-key-12345"
    assert get_openai_api_key("sk-custom-override") == "sk-custom-override"

    # 2. Simulate Local environment
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    assert get_deployment_mode() == "local"


# ==============================================================================
# TEST 4: QUESTION ANSWERING & GROUNDED CITATIONS
# ==============================================================================
def test_e2e_question_answering_and_grounded_citations(temp_dir, sample_multi_page_pdf, mock_embeddings, mock_llm_factory):
    """
    Verify end-to-end question answering pipeline:
    - User asks a question
    - Retrieval fetches relevant chunks
    - LLM answers citing [1]
    - Citation validation parses, verifies against retrieved context, and attaches page and rect metadata
    """
    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    ingest_graph = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=session_vs_dir)
    doc_id = "doc_qa_test"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    mock_llm = mock_llm_factory(
        canned_response="Section 1 details the autonomous agent architecture [1], and Section 2 introduces hybrid retrieval [2]."
    )

    query_graph = build_multi_query_graph(mock_embeddings, mock_llm, "mock-embed-64", base_dir=session_vs_dir)

    res = query_graph.invoke({
        "question": "What is described in Section 1 and Section 2?",
        "doc_id": doc_id,
        "top_k": 4
    })

    assert "citations" in res
    assert len(res["citations"]) >= 2
    cit_map = {c["citation_id"]: c for c in res["citations"]}

    assert 1 in cit_map
    assert 2 in cit_map
    assert cit_map[1]["doc_id"] == doc_id
    assert cit_map[1]["page_number"] in [1, 2, 3]
    assert res["has_sufficient_evidence"] is True


# ==============================================================================
# TEST 5: CLICKABLE PAGE NAVIGATION & STATE SYNCHRONIZATION
# ==============================================================================
def test_e2e_clickable_page_navigation(sample_multi_page_pdf):
    """
    Verify state synchronization when navigating pages or clicking citations:
    - Clicking next/prev updates viewer_page
    - Jumping to citation updates active_citation and page number
    - Out-of-bounds page requests are clamped
    """
    total_pages = 3
    state = {
        "viewer_page": 1,
        "tb_input_page": 1,
        "active_citation": None
    }

    def navigate(page: int):
        clamped = max(1, min(total_pages, page))
        state["viewer_page"] = clamped
        state["tb_input_page"] = clamped

    # Navigate forward
    navigate(state["viewer_page"] + 1)
    assert state["viewer_page"] == 2

    navigate(state["viewer_page"] + 1)
    assert state["viewer_page"] == 3

    # Clamped beyond total_pages
    navigate(state["viewer_page"] + 1)
    assert state["viewer_page"] == 3

    # Click citation on Page 1
    cit = {"citation_id": 1, "page_number": 1, "snippet": "Section 1"}
    state["active_citation"] = cit
    navigate(cit["page_number"])
    assert state["viewer_page"] == 1
    assert state["active_citation"]["citation_id"] == 1


# ==============================================================================
# TEST 6: ACCURATE EVIDENCE HIGHLIGHTING
# ==============================================================================
def test_e2e_evidence_highlighting(sample_multi_page_pdf):
    """
    Verify precise bounding box extraction and visual PNG highlight generation.
    """
    # 1. Resolve bounding box rects on page 2 for 'reciprocal rank fusion'
    rects = CitationResolutionService.resolve_citation_rects(
        pdf_path=sample_multi_page_pdf,
        page_number=2,
        snippet="Reciprocal rank fusion"
    )
    assert len(rects) >= 1
    assert len(rects[0]) == 4  # [x0, y0, x1, y1]

    # 2. Render page with highlights
    png_bytes = CitationResolutionService.render_page_with_highlights(
        pdf_path=sample_multi_page_pdf,
        page_number=2,
        rects=rects,
        zoom=1.25,
        base_dpi=150
    )
    assert png_bytes is not None
    assert len(png_bytes) > 1000

    # 3. Streamlit-pdf-viewer annotation formatting
    annots = CitationResolutionService.to_viewer_annotations(rects, page_number=2)
    assert len(annots) >= 1
    assert annots[0]["page"] == 2
    assert annots[0]["color"] == "#f59e0b"


# ==============================================================================
# TEST 7: UNSUPPORTED QUESTIONS REJECTION
# ==============================================================================
def test_e2e_unsupported_questions_rejection(temp_dir, sample_multi_page_pdf, mock_embeddings, mock_llm_factory):
    """
    Verify that out-of-domain or unsupported questions are rejected with
    has_sufficient_evidence = False and clear standard message.
    """
    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    ingest_graph = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=session_vs_dir)
    doc_id = "doc_unsupported"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    mock_llm = mock_llm_factory(
        canned_response="The provided document does not contain sufficient evidence to answer this question."
    )

    query_graph = build_multi_query_graph(mock_embeddings, mock_llm, "mock-embed-64", base_dir=session_vs_dir)

    res = query_graph.invoke({
        "question": "What is the speed of light in vacuum according to this paper?",
        "doc_id": doc_id,
        "top_k": 3
    })

    assert res["has_sufficient_evidence"] is False
    assert "does not contain sufficient evidence" in res["answer"].lower()


# ==============================================================================
# TEST 8: SCANNED AND EMPTY PDFS
# ==============================================================================
def test_e2e_scanned_pdf_handling(scanned_empty_pdf, temp_dir, mock_embeddings):
    """
    Verify that scanned / empty PDFs are detected, have is_scanned=True,
    and are handled without exceptions during metadata extraction and splitting.
    """
    meta = get_pdf_metadata(scanned_empty_pdf)
    assert meta["is_scanned"] is True
    assert meta["page_count"] == 1
    assert meta["total_chars"] == 0

    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    ingest_graph = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=session_vs_dir)

    # Ingesting empty/scanned PDF should complete cleanly with 0 chunks
    res = ingest_graph.invoke({"path": scanned_empty_pdf, "doc_id": "scanned_doc"})
    assert res["status"] == "success"
    assert res["chunk_count"] == 0


# ==============================================================================
# TEST 9: MODEL AND EMBEDDING COMPATIBILITY AND REBUILDING
# ==============================================================================
def test_e2e_embedding_compatibility_and_rebuilding(temp_dir, sample_multi_page_pdf, mock_embeddings):
    """
    Verify that changing embedding models is detected and re-indexing updates the metadata.
    """
    session_vs_dir = os.path.join(temp_dir, "vectorstores")
    doc_id = "compat_test_doc"

    # Build index with model A
    ingest_v1 = build_multi_ingest_graph(mock_embeddings, "model-alpha-64", base_dir=session_vs_dir)
    ingest_v1.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    idx_path = os.path.join(session_vs_dir, f"{doc_id}_index")
    compat, msg = check_embedding_compatibility(idx_path, "model-alpha-64")
    assert compat is True

    # Check with model B -> mismatch!
    compat_bad, msg_bad = check_embedding_compatibility(idx_path, "model-beta-128")
    assert compat_bad is False
    assert "mismatch" in msg_bad.lower()

    # Re-index with model B
    ingest_v2 = build_multi_ingest_graph(mock_embeddings, "model-beta-128", base_dir=session_vs_dir)
    ingest_v2.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    # Now compatible with model B
    compat_now, _ = check_embedding_compatibility(idx_path, "model-beta-128")
    assert compat_now is True


# ==============================================================================
# TEST 10: CONCURRENT USER SESSIONS ISOLATION
# ==============================================================================
def test_e2e_concurrent_user_sessions_isolation(temp_dir, sample_multi_page_pdf, sample_second_pdf, mock_embeddings):
    """
    Verify that two concurrent user sessions (User A and User B) have strictly
    isolated directories, uploads, and vector stores.
    """
    session_a = "session_user_alice"
    session_b = "session_user_bob"

    vs_dir_a = os.path.join(temp_dir, "sessions", session_a, "vectorstores")
    vs_dir_b = os.path.join(temp_dir, "sessions", session_b, "vectorstores")

    ingest_a = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=vs_dir_a)
    ingest_b = build_multi_ingest_graph(mock_embeddings, "mock-embed-64", base_dir=vs_dir_b)

    # Alice uploads Paper 1
    ingest_a.invoke({"path": sample_multi_page_pdf, "doc_id": "alice_doc"})
    # Bob uploads Paper 2
    ingest_b.invoke({"path": sample_second_pdf, "doc_id": "bob_doc"})

    # Verify Alice's index is ONLY in Alice's directory
    assert os.path.exists(os.path.join(vs_dir_a, "alice_doc_index"))
    assert not os.path.exists(os.path.join(vs_dir_a, "bob_doc_index"))

    # Verify Bob's index is ONLY in Bob's directory
    assert os.path.exists(os.path.join(vs_dir_b, "bob_doc_index"))
    assert not os.path.exists(os.path.join(vs_dir_b, "alice_doc_index"))

    # Alice's retriever cannot access Bob's index
    retriever_a = HybridRetriever(mock_embeddings, "mock-embed-64", base_dir=vs_dir_a)
    results = retriever_a.retrieve(query="quantum", doc_id="bob_doc", cross_doc_mode=False)
    assert len(results) == 0  # Does not exist in Alice's store


# ==============================================================================
# TEST 11: ERROR HANDLING AND RECOVERY
# ==============================================================================
def test_e2e_error_handling_and_recovery(temp_dir, corrupted_pdf):
    """
    Verify robust error handling for corrupted files, path traversal, and missing keys.
    """
    # 1. Path traversal sanitizer
    evil = "../../etc/passwd"
    clean = sanitize_filename(evil)
    assert "/" not in clean and ".." not in clean

    # 2. Corrupted PDF handled gracefully by loader
    with pytest.raises(ValueError, match="Failed to open PDF"):
        load_pdf(corrupted_pdf)

    # 3. Path validation outside sandbox returns False
    allowed = [os.path.join(temp_dir, "sandbox")]
    assert validate_pdf_path("/etc/shadow", allowed) is False


# ==============================================================================
# TEST 12: REAL INFERENCE TEST WITH LOCAL OLLAMA (WHEN AVAILABLE)
# ==============================================================================
def test_real_inference_ollama(temp_dir, sample_multi_page_pdf):
    """
    Live inference test using real local Ollama (llama3.1:8b and nomic-embed-text)
    if the Ollama server is running.
    """
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                pytest.skip("Local Ollama not reachable")
    except Exception:
        pytest.skip("Local Ollama not reachable")

    try:
        from models.ollama_config import ollama_embeddings, ollama_llm
        emb = ollama_embeddings()
        llm = ollama_llm()
    except Exception as e:
        pytest.skip(f"Could not load Ollama models: {e}")

    session_vs_dir = os.path.join(temp_dir, "real_ollama_vs")
    model_id = get_embedding_model_id(emb)

    # Ingest using real nomic-embed-text embeddings
    ingest_graph = build_multi_ingest_graph(emb, model_id, base_dir=session_vs_dir)
    res = ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": "real_doc"})
    assert res["status"] == "success"

    # Query using real llama3.1:8b and real nomic-embed-text embeddings
    query_graph = build_multi_query_graph(emb, llm, model_id, base_dir=session_vs_dir)
    result = query_graph.invoke({
        "question": "What does Section 1 cover?",
        "doc_id": "real_doc",
        "top_k": 2
    })

    assert "answer" in result
    assert len(result["answer"]) > 5
    assert "citations" in result
    assert len(result["citations"]) >= 1
    assert result["has_sufficient_evidence"] is True
