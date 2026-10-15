from rag.ingest_multi import build_multi_ingest_graph
from rag.retriever import HybridRetriever


def test_document_isolation(temp_dir, sample_multi_page_pdf, sample_second_pdf, mock_embeddings, monkeypatch):
    """
    Test 5: Verify strict document isolation.
    Querying doc A must NEVER leak or return context from doc B.
    """
    # Point MULTI_BASE_VS_DIR to temp_dir for isolated testing
    from rag import ingest_multi, retriever
    monkeypatch.setattr(ingest_multi, "MULTI_BASE_VS_DIR", temp_dir)
    monkeypatch.setattr(retriever, "MULTI_BASE_VS_DIR", temp_dir)

    ingest_graph = build_multi_ingest_graph(mock_embeddings, embedding_model_id="mock-embed-64")

    # Ingest Doc A (AI / RAG paper)
    doc_a_id = "doc_a_rag"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_a_id})

    # Ingest Doc B (Quantum computing)
    doc_b_id = "doc_b_quantum"
    ingest_graph.invoke({"path": sample_second_pdf, "doc_id": doc_b_id})

    active_model_id = "mock-embed-64"
    h_retriever = HybridRetriever(mock_embeddings, active_model_id=active_model_id, base_dir=temp_dir)

    # 1. Query Doc A for 'quantum qubits' (a concept only in Doc B)
    results_a = h_retriever.retrieve(
        query="quantum qubits",
        doc_id=doc_a_id,
        top_k=3,
        cross_doc_mode=False
    )
    # Every returned chunk MUST belong to doc_a_id, never doc_b_id!
    for chunk in results_a:
        assert chunk["doc_id"] == doc_a_id
        assert "superconducting" not in chunk["text"].lower()

    # 2. Query Doc B for 'autonomous agent architectures' (a concept only in Doc A)
    results_b = h_retriever.retrieve(
        query="autonomous agent architectures",
        doc_id=doc_b_id,
        top_k=3,
        cross_doc_mode=False
    )
    for chunk in results_b:
        assert chunk["doc_id"] == doc_b_id
        assert "antigravity" not in chunk["text"].lower()


def test_cross_document_retrieval_mode(temp_dir, sample_multi_page_pdf, sample_second_pdf, mock_embeddings, monkeypatch):
    """
    Verify that when cross-document mode is explicitly enabled, the retriever
    aggregates results across both documents.
    """
    from rag import ingest_multi, retriever
    monkeypatch.setattr(ingest_multi, "MULTI_BASE_VS_DIR", temp_dir)
    monkeypatch.setattr(retriever, "MULTI_BASE_VS_DIR", temp_dir)

    ingest_graph = build_multi_ingest_graph(mock_embeddings, embedding_model_id="mock-embed-64")
    doc_a_id = "doc_a"
    doc_b_id = "doc_b"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_a_id})
    ingest_graph.invoke({"path": sample_second_pdf, "doc_id": doc_b_id})

    h_retriever = HybridRetriever(mock_embeddings, active_model_id="mock-embed-64", base_dir=temp_dir)

    cross_results = h_retriever.retrieve(
        query="methodologies and quantum circuits",
        doc_id=doc_a_id,
        top_k=6,
        cross_doc_mode=True,
        available_doc_ids=[doc_a_id, doc_b_id]
    )

    doc_ids_present = {c["doc_id"] for c in cross_results}
    assert doc_a_id in doc_ids_present
    assert doc_b_id in doc_ids_present


def test_hybrid_bm25_lexical_matching(temp_dir, sample_multi_page_pdf, mock_embeddings, monkeypatch):
    """
    Verify that BM25 lexical keyword search ranks exact vocabulary matches highly.
    """
    from rag import ingest_multi, retriever
    monkeypatch.setattr(ingest_multi, "MULTI_BASE_VS_DIR", temp_dir)
    monkeypatch.setattr(retriever, "MULTI_BASE_VS_DIR", temp_dir)

    ingest_graph = build_multi_ingest_graph(mock_embeddings, embedding_model_id="mock-embed-64")
    doc_id = "doc_bm25_test"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    h_retriever = HybridRetriever(mock_embeddings, active_model_id="mock-embed-64", base_dir=temp_dir)

    # 'reciprocal rank fusion' is a very specific keyword term on Page 2
    results = h_retriever.retrieve(
        query="reciprocal rank fusion",
        doc_id=doc_id,
        top_k=3,
        use_hybrid=True
    )

    assert len(results) > 0
    top_chunk = results[0]
    assert "reciprocal rank fusion" in top_chunk["text"].lower()
    assert top_chunk["page"] == 2
