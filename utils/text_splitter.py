from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.citation_engine import CitationResolutionService


def split_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    doc_id: str = "doc",
    pdf_path: Optional[str] = None,
    embedding_model_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Split pages into chunks while preserving complete provenance:
    - doc_id
    - page_number (1-indexed)
    - chunk_id (deterministic, e.g. {doc_id}_p{page}_c{idx})
    - source_text
    - text_span ([start_char, end_char])
    - bounding_boxes (PDF coordinate rects [x0, y0, x1, y1])
    - embedding_model_id
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    chunk_global_idx = 0

    for p in pages:
        page_num = p["page"]
        page_text = p.get("text", "")
        if not page_text.strip():
            continue

        raw_chunks = splitter.split_text(page_text)
        search_cursor = 0

        for chunk_idx, text_chunk in enumerate(raw_chunks):
            # Locate character span within original page text
            start_pos = page_text.find(text_chunk, max(0, search_cursor - chunk_overlap))
            if start_pos == -1:
                start_pos = page_text.find(text_chunk)
            
            if start_pos != -1:
                end_pos = start_pos + len(text_chunk)
                search_cursor = end_pos
                text_span = [start_pos, end_pos]
            else:
                text_span = [0, len(text_chunk)]

            chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"

            # Compute bounding boxes if PDF path is available
            bboxes = []
            if pdf_path:
                try:
                    bboxes = CitationResolutionService.resolve_citation_rects(
                        pdf_path=pdf_path,
                        page_number=page_num,
                        snippet=text_chunk
                    )
                except Exception:
                    bboxes = []

            chunk_item = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page_num,
                "page_number": page_num,
                "text": text_chunk,
                "source_text": text_chunk,
                "text_span": text_span,
                "bounding_boxes": bboxes,
                "embedding_model_id": embedding_model_id or "default",
                "index": chunk_global_idx
            }
            chunks.append(chunk_item)
            chunk_global_idx += 1

    return chunks
