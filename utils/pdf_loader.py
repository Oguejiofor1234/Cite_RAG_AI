import os
from typing import List, Dict, Any
import pymupdf


def load_pdf(path: str, enable_ocr: bool = True) -> List[Dict[str, Any]]:
    """
    Load a PDF document using PyMuPDF and extract page-level text and dimensions.
    
    Pages are 1-indexed. Handles normal text PDFs, detects scanned/empty pages,
    applies OCR fallback for image-based/scanned pages when appropriate,
    and raises ValueError for missing or invalid PDF files.
    
    Returns:
        List of dicts:
        [
            {
                "page": int,         # 1-indexed page number
                "text": str,         # Extracted raw text
                "width": float,      # Page width in PDF points
                "height": float,     # Page height in PDF points
                "char_count": int,   # Character length
                "is_empty": bool     # True if page contains no extractable text
            },
            ...
        ]
    """
    if not os.path.exists(path):
        raise ValueError(f"PDF file does not exist at path: {path}")

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file {path}: {str(e)}") from e

    try:
        pages = []
        for i, page in enumerate(doc):
            page_num = i + 1
            text = page.get_text("text") or ""
            is_empty = len(text.strip()) == 0

            # Automatic OCR fallback for scanned pages or image-only pages
            if enable_ocr and (len(text.strip()) < 50) and len(page.get_images()) > 0:
                try:
                    tp = page.get_textpage_ocr(language='eng', dpi=150)
                    ocr_text = tp.extractTEXT() or ""
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        is_empty = len(text.strip()) == 0
                except Exception:
                    pass
            
            pages.append({
                "page": page_num,
                "text": text,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "char_count": len(text.strip()),
                "is_empty": is_empty
            })
        return pages
    finally:
        doc.close()


def get_pdf_metadata(path: str) -> Dict[str, Any]:
    """
    Retrieve document-level metadata using PyMuPDF.
    """
    if not os.path.exists(path):
        raise ValueError(f"PDF file does not exist: {path}")
        
    doc = pymupdf.open(path)
    try:
        meta = doc.metadata or {}
        page_count = len(doc)
        total_chars = sum(len(page.get_text("text").strip()) for page in doc)
        return {
            "page_count": page_count,
            "title": meta.get("title") or os.path.basename(path),
            "author": meta.get("author") or "Unknown",
            "is_scanned": (total_chars == 0 and page_count > 0),
            "total_chars": total_chars
        }
    finally:
        doc.close()
