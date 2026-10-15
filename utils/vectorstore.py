import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import FAISS


INDEX_META_FILE = "index_meta.json"
CHUNKS_MANIFEST_FILE = "chunks_manifest.json"


def build_faiss_from_texts(texts: List[str], embeddings, metadatas: Optional[List[Dict[str, Any]]] = None):
    """
    Build a FAISS vector store from texts and metadata.
    """
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)


def save_faiss(
    vs: FAISS,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    chunks: Optional[List[Dict[str, Any]]] = None
):
    """
    Save FAISS index locally with accompanying index metadata and chunks manifest.
    
    Persisting index_meta.json prevents silent crashes caused by embedding dimension
    mismatches when switching between Ollama and OpenAI backends.
    """
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)

    if meta is not None:
        save_index_metadata(path, meta)

    if chunks is not None:
        save_chunks_manifest(path, chunks)


def save_index_metadata(path: str, meta: Dict[str, Any]):
    """
    Save metadata JSON associated with a FAISS index directory.
    """
    os.makedirs(path, exist_ok=True)
    meta_copy = dict(meta)
    if "updated_at" not in meta_copy:
        meta_copy["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta_path = os.path.join(path, INDEX_META_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_copy, f, indent=2)


def get_index_metadata(path: str) -> Optional[Dict[str, Any]]:
    """
    Load index metadata JSON if present.
    """
    meta_path = os.path.join(path, INDEX_META_FILE)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_chunks_manifest(path: str, chunks: List[Dict[str, Any]]):
    """
    Save the list of text chunks with provenance to chunks_manifest.json.
    """
    os.makedirs(path, exist_ok=True)
    manifest_path = os.path.join(path, CHUNKS_MANIFEST_FILE)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


def load_chunks_manifest(path: str) -> List[Dict[str, Any]]:
    """
    Load text chunks from chunks_manifest.json if available.
    """
    manifest_path = os.path.join(path, CHUNKS_MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def check_embedding_compatibility(path: str, active_model_id: str) -> Tuple[bool, str]:
    """
    Verify whether the index saved at path is compatible with the active embedding model.
    Returns (is_compatible, message).
    """
    meta = get_index_metadata(path)
    if not meta:
        # Legacy index without metadata file: allow with note
        return True, "Legacy index detected (no metadata file)."

    indexed_model = meta.get("embedding_model_id")
    if not indexed_model:
        return True, "No embedding model ID in metadata."

    if indexed_model != active_model_id:
        return False, (
            f"Index model mismatch: Index was built with '{indexed_model}', "
            f"but active embedding model is '{active_model_id}'. "
            "Please re-index the document to avoid query errors."
        )

    return True, "Compatible."


def load_faiss(path: str, embeddings, expected_model_id: Optional[str] = None) -> FAISS:
    """
    Load a local FAISS index with model compatibility verification.
    Raises ValueError if expected_model_id is specified and does not match the index.
    """
    if expected_model_id:
        compatible, msg = check_embedding_compatibility(path, expected_model_id)
        if not compatible:
            raise ValueError(msg)

    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
