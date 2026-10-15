from rag.query_multi import build_multi_query_graph


def test_grounded_citations_validation(temp_dir, sample_multi_page_pdf, mock_embeddings, mock_llm_factory, monkeypatch):
    """
    Verify citation validation: citations [1] in answer map to retrieved chunks,
    and hallucinated citations (e.g. [99]) are safely stripped/ignored.
    """
    from rag import ingest_multi, retriever
    monkeypatch.setattr(ingest_multi, "MULTI_BASE_VS_DIR", temp_dir)
    monkeypatch.setattr(retriever, "MULTI_BASE_VS_DIR", temp_dir)

    from rag.ingest_multi import build_multi_ingest_graph
    ingest_graph = build_multi_ingest_graph(mock_embeddings, embedding_model_id="mock-embed-64")
    doc_id = "grounded_qa_doc"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    # Mock LLM citing both valid [1] and hallucinated [99]
    mock_llm = mock_llm_factory(
        canned_response="The platform introduces agent architectures [1]. An unverified claim citing [99]."
    )

    query_graph = build_multi_query_graph(
        mock_embeddings,
        mock_llm,
        embedding_model_id="mock-embed-64",
        base_dir=temp_dir
    )

    result = query_graph.invoke({
        "question": "What does the platform introduce?",
        "doc_id": doc_id,
        "top_k": 3
    })

    assert "citations" in result
    citations = result["citations"]
    assert len(citations) >= 1

    # Citation [1] should be present and validated
    cit_ids = [c["citation_id"] for c in citations]
    assert 1 in cit_ids
    # Citation [99] must NOT be present
    assert 99 not in cit_ids

    # Check citation provenance structure
    c1 = next(c for c in citations if c["citation_id"] == 1)
    assert c1["doc_id"] == doc_id
    assert c1["page_number"] in [1, 2, 3]
    assert len(c1["snippet"]) > 0
    assert result["has_sufficient_evidence"] is True


def test_unsupported_answer_behavior(temp_dir, sample_multi_page_pdf, mock_embeddings, mock_llm_factory, monkeypatch):
    """
    Test 7: Verify clear indication when evidence is insufficient.
    """
    from rag import ingest_multi, retriever
    monkeypatch.setattr(ingest_multi, "MULTI_BASE_VS_DIR", temp_dir)
    monkeypatch.setattr(retriever, "MULTI_BASE_VS_DIR", temp_dir)

    from rag.ingest_multi import build_multi_ingest_graph
    ingest_graph = build_multi_ingest_graph(mock_embeddings, embedding_model_id="mock-embed-64")
    doc_id = "unsupported_doc"
    ingest_graph.invoke({"path": sample_multi_page_pdf, "doc_id": doc_id})

    # Mock LLM declaring insufficient evidence
    mock_llm = mock_llm_factory(
        canned_response="The provided document does not contain sufficient evidence to answer this question."
    )

    query_graph = build_multi_query_graph(
        mock_embeddings,
        mock_llm,
        embedding_model_id="mock-embed-64",
        base_dir=temp_dir
    )

    result = query_graph.invoke({
        "question": "What is the capital of Mars?",
        "doc_id": doc_id,
        "top_k": 3
    })

    assert result["has_sufficient_evidence"] is False
    assert "does not contain sufficient evidence" in result["answer"].lower()
