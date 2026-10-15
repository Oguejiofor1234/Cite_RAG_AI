import os
import pytest
from utils.security import sanitize_filename, validate_pdf_path
from utils.vectorstore import (
    save_index_metadata,
    check_embedding_compatibility,
    load_faiss,
)
from models import (
    get_embedding_model_id,
    ollama_llm,
    ollama_embeddings,
    openai_llm,
    openai_embeddings,
)


def test_file_upload_validation_and_security(temp_dir):
    """
    Test 11: Verify secure file upload validation, directory traversal prevention,
    and path containment checks.
    """
    # 1. Directory traversal in filename
    malicious_names = [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\cmd.exe",
        "nested/../../secret.pdf",
        "evil;rm -rf;.pdf"
    ]
    for m in malicious_names:
        sanitized = sanitize_filename(m)
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized
        assert sanitized.endswith(".pdf")

    # 2. Path validation within allowed directories
    allowed_dirs = [os.path.join(temp_dir, "uploads")]
    os.makedirs(allowed_dirs[0], exist_ok=True)

    safe_path = os.path.join(allowed_dirs[0], "doc.pdf")
    with open(safe_path, "w") as f:
        f.write("content")

    unsafe_path = os.path.join(temp_dir, "unauthorized.pdf")
    with open(unsafe_path, "w") as f:
        f.write("content")

    assert validate_pdf_path(safe_path, allowed_dirs) is True
    assert validate_pdf_path(unsafe_path, allowed_dirs) is False
    assert validate_pdf_path("/etc/passwd", allowed_dirs) is False


def test_index_compatibility_and_migration(temp_dir):
    """
    Test 10: Verify index compatibility tracking between embedding models
    and migration handling for legacy indexes.
    """
    index_dir = os.path.join(temp_dir, "test_index")
    os.makedirs(index_dir, exist_ok=True)

    # 1. Legacy index with no metadata file
    compat, msg = check_embedding_compatibility(index_dir, "openai:text-embedding-3-small")
    assert compat is True
    assert "Legacy index" in msg

    # 2. Index built with Ollama nomic-embed-text
    save_index_metadata(index_dir, {
        "doc_id": "test_doc",
        "embedding_model_id": "ollama:nomic-embed-text",
        "chunk_count": 5
    })

    # Checking with matching model
    compat_ok, msg_ok = check_embedding_compatibility(index_dir, "ollama:nomic-embed-text")
    assert compat_ok is True

    # Checking with mismatched model (OpenAI)
    compat_bad, msg_bad = check_embedding_compatibility(index_dir, "openai:text-embedding-3-small")
    assert compat_bad is False
    assert "mismatch" in msg_bad.lower()

    # load_faiss with expected_model_id should raise ValueError
    with pytest.raises(ValueError, match="mismatch"):
        load_faiss(index_dir, embeddings=None, expected_model_id="openai:text-embedding-3-small")


def test_backend_switching(monkeypatch):
    """
    Test 9: Verify backend switching and model factory initialization.
    """
    # Ollama models should instantiate without crashing
    ol_model = ollama_llm()
    ol_emb = ollama_embeddings()
    assert ol_model is not None
    assert ol_emb is not None
    assert get_embedding_model_id(ol_emb) == "ollama:nomic-embed-text"

    # OpenAI models should instantiate properly when API key is configured
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-mock")
    oa_model = openai_llm()
    oa_emb = openai_embeddings()
    assert oa_model is not None
    assert oa_emb is not None
    assert get_embedding_model_id(oa_emb) == "openai:text-embedding-3-small"
