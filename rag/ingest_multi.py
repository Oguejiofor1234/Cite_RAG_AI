import os
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from utils.pdf_loader import load_pdf, get_pdf_metadata
from utils.text_splitter import split_pages
from utils.vectorstore import (
    build_faiss_from_texts,
    save_faiss,
    save_index_metadata,
    save_chunks_manifest,
)

from models import get_embedding_model_id

MULTI_BASE_VS_DIR = "data/multi/vectorstores"


def build_multi_ingest_graph(
    embeddings,
    embedding_model_id: Optional[str] = None,
    base_dir: Optional[str] = None
):
    """
    Build and compile a LangGraph ingestion workflow for multi-document indexing.
    
    Nodes:
    1. load: Loads PDF with PyMuPDF, extracts text and 1-indexed page enumeration.
    2. chunk: Splits pages while maintaining complete chunk IDs, character spans,
              and bounding boxes.
    3. embed_store: Embeds chunks into FAISS, saves index alongside index_meta.json
                   and chunks_manifest.json to enforce index compatibility.
    """
    model_id = embedding_model_id or get_embedding_model_id(embeddings)
    target_base_dir = base_dir or MULTI_BASE_VS_DIR


    def load_node(state: Dict[str, Any]):
        path = state["path"]
        doc_id = state["doc_id"]
        pages = load_pdf(path)
        meta = get_pdf_metadata(path)
        return {
            "pages": pages,
            "doc_id": doc_id,
            "path": path,
            "pdf_meta": meta
        }

    def chunk_node(state: Dict[str, Any]):
        pages = state["pages"]
        doc_id = state["doc_id"]
        path = state.get("path")
        chunks = split_pages(
            pages=pages,
            chunk_size=800,
            chunk_overlap=100,
            doc_id=doc_id,
            pdf_path=path,
            embedding_model_id=model_id
        )
        return {
            "chunks": chunks,
            "doc_id": doc_id,
            "path": path,
            "pdf_meta": state.get("pdf_meta", {})
        }

    def embed_store_node(state: Dict[str, Any]):
        doc_id = state["doc_id"]
        chunks = state["chunks"]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "page": c["page"],
                "page_number": c["page"],
                "doc_id": doc_id,
                "chunk_id": c["chunk_id"],
                "text_span": c.get("text_span", [0, len(c["text"])]),
                "bounding_boxes": c.get("bounding_boxes", []),
                "embedding_model_id": model_id
            }
            for c in chunks
        ]

        vs_dir = os.path.join(target_base_dir, f"{doc_id}_index")
        os.makedirs(vs_dir, exist_ok=True)

        pdf_meta = state.get("pdf_meta", {})
        index_meta = {
            "doc_id": doc_id,
            "embedding_model_id": model_id,
            "chunk_count": len(chunks),
            "page_count": pdf_meta.get("page_count", len(state.get("pages", []))),
            "path": state.get("path", ""),
            "title": pdf_meta.get("title", ""),
            "is_scanned": (len(chunks) == 0)
        }

        if chunks:
            vs = build_faiss_from_texts(texts, embeddings, metadatas)
            save_faiss(vs, vs_dir, meta=index_meta, chunks=chunks)
        else:
            save_index_metadata(vs_dir, index_meta)
            save_chunks_manifest(vs_dir, [])

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "page_count": index_meta["page_count"]
        }


    graph = StateGraph(dict)
    graph.add_node("load", load_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed_store", embed_store_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "chunk")
    graph.add_edge("chunk", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()
