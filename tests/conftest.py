import sys
import os
import shutil
import tempfile
import pytest
import pymupdf
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class DeterministicMockEmbeddings:
    """
    Fast, deterministic embeddings fixture for testing.
    Produces repeatable 64-dimensional float vectors based on text hash
    without requiring external Ollama or OpenAI connections.
    """
    def __init__(self, model: str = "mock-embed-64"):
        self.model = model

    def _embed(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [float((b % 100) / 100.0) for b in h[:32]] + [0.5] * 32
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class DeterministicMockLLM:
    """
    Deterministic mock LLM for testing grounded QA workflows.
    """
    def __init__(self, canned_response: str = "According to the document [1], precision citations are verified."):
        self.canned_response = canned_response

    def invoke(self, prompt, **kwargs):
        class MockResponse:
            def __init__(self, content):
                self.content = content
            def __str__(self):
                return self.content
        return MockResponse(self.canned_response)

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


@pytest.fixture
def temp_dir():
    """Temporary test directory cleaned up after each test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_llm_factory():
    def _create(canned_response: str = "Grounded response [1]."):
        return DeterministicMockLLM(canned_response)
    return _create


@pytest.fixture
def mock_embeddings():
    return DeterministicMockEmbeddings(model="test-embed-v1")


@pytest.fixture
def sample_multi_page_pdf(temp_dir):
    """
    Create a deterministic 3-page PDF with known text and known coordinates.
    """
    pdf_path = os.path.join(temp_dir, "multi_page_test.pdf")
    doc = pymupdf.open()

    # Page 1 (1-indexed)
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(pymupdf.Point(72, 100), "Antigravity Research Platform Introduction.")
    p1.insert_text(pymupdf.Point(72, 130), "Section 1 covers autonomous agent architectures and precision evidence linking.")

    # Page 2 (1-indexed)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(pymupdf.Point(72, 100), "Section 2 focuses on hybrid retrieval combining BM25 and vector embeddings.")
    p2.insert_text(pymupdf.Point(72, 130), "Reciprocal rank fusion reconciles sparse and dense score distributions.")

    # Page 3 (1-indexed)
    p3 = doc.new_page(width=612, height=792)
    p3.insert_text(pymupdf.Point(72, 100), "Section 3 discusses limitations, evaluation methodologies, and strict context isolation.")
    p3.insert_text(pymupdf.Point(72, 130), "Multi-document environments require dedicated isolation per document id.")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_second_pdf(temp_dir):
    """
    Create a second distinct PDF for testing document isolation.
    """
    pdf_path = os.path.join(temp_dir, "isolated_doc_two.pdf")
    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(pymupdf.Point(72, 100), "Quantum computing algorithms utilize qubits and superconducting circuits.")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def scanned_empty_pdf(temp_dir):
    """
    Create a 1-page PDF with no extractable text.
    """
    pdf_path = os.path.join(temp_dir, "scanned_empty.pdf")
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def corrupted_pdf(temp_dir):
    """
    Create a corrupted file.
    """
    path = os.path.join(temp_dir, "corrupt.pdf")
    with open(path, "wb") as f:
        f.write(b"%PDF-INVALID_CORRUPTED_BYTES_NOT_A_REAL_PDF")
    return path
