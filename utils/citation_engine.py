import os
import re
from typing import List, Dict, Any, Optional, Tuple
import pymupdf


class CitationResolutionService:
    """
    Precision citation resolution service using PyMuPDF.
    
    Resolves citation metadata to exact PDF coordinates, generates multi-line /
    multi-rectangle highlight bounding boxes, exports annotations for PDF viewers,
    and renders high-precision page images with crisp visual evidence highlights and
    prominent enclosing rectangle bounding boxes.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip markdown markers, citation tags, and normalize whitespace."""
        if not text:
            return ""
        # Strip citation brackets like [1], [2], [99]
        cleaned = re.sub(r'\[\d+\]', '', text)
        # Strip markdown syntax
        cleaned = re.sub(r'[*_#`]', '', cleaned)
        # Normalize special quotes, dashes, non-breaking spaces
        cleaned = cleaned.replace('\xa0', ' ').replace('—', '-').replace('–', '-')
        cleaned = cleaned.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
        # Normalize punctuation spacing
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    @classmethod
    def _search_page_for_text(
        cls,
        page: pymupdf.Page,
        needle: str,
        textpage: Optional[Any] = None
    ) -> List[List[float]]:
        """Search page or OCR textpage for a string, using de-hyphenation flags."""
        if not needle or len(needle.strip()) < 3:
            return []
        
        cleaned = needle.strip()
        kwargs: Dict[str, Any] = {"flags": pymupdf.TEXT_DEHYPHENATE}
        if textpage is not None:
            kwargs["textpage"] = textpage
            
        try:
            matches = page.search_for(cleaned, **kwargs)
            if matches:
                return [[round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)] for r in matches]
        except Exception:
            pass
        return []

    @classmethod
    def find_text_rects(
        cls,
        page: pymupdf.Page,
        snippet: str,
        enable_ocr: bool = True,
        apply_rotation: bool = False
    ) -> List[List[float]]:
        """
        Locate bounding box rectangles for a text passage on a PyMuPDF page.
        
        Applies multi-stage fallback matching:
        1. Exact passage search (with dehyphenation)
        2. Normalized passage search
        3. Line-by-line search for multi-line passages
        4. Sentence / clause matching for formatted text
        5. Significant phrase n-grams
        6. OCR textpage fallback for scanned or image-based pages
        
        Returns an empty list if exact positioning cannot be established,
        preventing misleading or hallucinated highlight boxes.
        """
        if not snippet or not snippet.strip():
            return []

        cleaned_snippet = snippet.strip()
        clean_norm = cls.normalize_text(cleaned_snippet)

        def _perform_search(tp: Optional[Any] = None) -> List[List[float]]:
            # Strategy 1 & 2: Exact search on raw snippet & normalized snippet
            for s_str in [cleaned_snippet, clean_norm]:
                if s_str and len(s_str) >= 3:
                    rects = cls._search_page_for_text(page, s_str, textpage=tp)
                    if rects:
                        return rects

            # Strategy 3: Line-by-line search (handles line-wrapping across PDF text blocks)
            lines = [l.strip() for l in cleaned_snippet.splitlines() if len(l.strip()) >= 4]
            line_rects = []
            for line in lines:
                matches = cls._search_page_for_text(page, line, textpage=tp)
                for m in matches:
                    if m not in line_rects:
                        line_rects.append(m)
            if line_rects:
                return line_rects

            # Strategy 4: Sentence / clause matching
            raw_sentences = re.split(r'[.!?\n]+', cleaned_snippet)
            sentences = [cls.normalize_text(s) for s in raw_sentences if len(cls.normalize_text(s)) >= 12]
            sent_rects = []
            for sent in sentences:
                matches = cls._search_page_for_text(page, sent, textpage=tp)
                for m in matches:
                    if m not in sent_rects:
                        sent_rects.append(m)
            if sent_rects:
                return sent_rects

            # Strategy 5: Significant phrase matching (windows of 6-8 words)
            words = clean_norm.split()
            if len(words) >= 6:
                for window_size in [8, 6, 5]:
                    if len(words) >= window_size:
                        phrase = " ".join(words[:window_size])
                        phrase_rects = cls._search_page_for_text(page, phrase, textpage=tp)
                        if phrase_rects:
                            return phrase_rects

            return []

        # First attempt on standard native text page
        rects = _perform_search(tp=None)

        # Second attempt using OCR textpage fallback if page is scanned or native text was sparse
        if not rects and enable_ocr:
            native_text_len = len((page.get_text("text") or "").strip())
            has_images = len(page.get_images()) > 0
            if native_text_len < 120 or has_images:
                try:
                    tp = page.get_textpage_ocr(language='eng', dpi=150)
                    rects = _perform_search(tp=tp)
                except Exception:
                    pass

        if not rects:
            return []

        # Apply page rotation transformation if requested
        if apply_rotation and page.rotation != 0:
            rot_matrix = page.rotation_matrix
            rotated_rects = []
            for r in rects:
                rot_r = pymupdf.Rect(r[0], r[1], r[2], r[3]) * rot_matrix
                rotated_rects.append([
                    round(rot_r.x0, 2),
                    round(rot_r.y0, 2),
                    round(rot_r.x1, 2),
                    round(rot_r.y1, 2)
                ])
            return rotated_rects

        return rects

    @classmethod
    def resolve_citation_rects(
        cls,
        pdf_path: str,
        page_number: int,
        snippet: str,
        apply_rotation: bool = False,
        enable_ocr: bool = True
    ) -> List[List[float]]:
        """
        Open a PDF and find bounding box rectangles for the given snippet on the cited page.
        Page numbers are 1-indexed.
        """
        if not os.path.exists(pdf_path) or page_number < 1:
            return []

        try:
            doc = pymupdf.open(pdf_path)
        except Exception:
            return []

        try:
            if page_number > len(doc):
                return []
            page = doc[page_number - 1]
            return cls.find_text_rects(
                page=page,
                snippet=snippet,
                enable_ocr=enable_ocr,
                apply_rotation=apply_rotation
            )
        finally:
            doc.close()

    @staticmethod
    def compute_enclosing_box(
        rects: List[List[float]],
        padding: float = 4.0
    ) -> Optional[List[float]]:
        """
        Compute an enclosing bounding box [x0, y0, x1, y1] around a list of rects,
        adding padding for clean visual framing.
        """
        if not rects:
            return None
        x0 = min(r[0] for r in rects) - padding
        y0 = min(r[1] for r in rects) - padding
        x1 = max(r[2] for r in rects) + padding
        y1 = max(r[3] for r in rects) + padding
        return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]

    @staticmethod
    def cluster_rects(
        rects: List[List[float]],
        max_vertical_gap: float = 28.0
    ) -> List[List[List[float]]]:
        """
        Cluster rectangles that belong to the same text block / paragraph,
        preventing giant awkwardly-spanning boxes across distant page sections.
        """
        if not rects:
            return []
        sorted_rects = sorted(rects, key=lambda r: (r[1], r[0]))
        clusters = []
        current_cluster = [sorted_rects[0]]

        for r in sorted_rects[1:]:
            prev_y1 = max(c[3] for c in current_cluster)
            if r[1] <= prev_y1 + max_vertical_gap:
                current_cluster.append(r)
            else:
                clusters.append(current_cluster)
                current_cluster = [r]
        if current_cluster:
            clusters.append(current_cluster)
        return clusters

    @classmethod
    def to_viewer_annotations(
        cls,
        rects: List[List[float]],
        page_number: int,
        color: str = "#f59e0b",
        id_prefix: str = "hl",
        include_box: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Convert bounding box rects [x0, y0, x1, y1] to streamlit-pdf-viewer annotation format.
        Coordinates are preserved in native PDF points.
        """
        annotations = []
        for i, r in enumerate(rects):
            x0, y0, x1, y1 = r
            annotations.append({
                "page": int(page_number),
                "x": float(x0),
                "y": float(y0),
                "width": float(max(1.0, x1 - x0)),
                "height": float(max(1.0, y1 - y0)),
                "color": color,
                "border": "solid",
                "id": f"{id_prefix}_{page_number}_{i}"
            })

        if include_box and rects:
            clusters = cls.cluster_rects(rects)
            for c_idx, cluster in enumerate(clusters):
                box = cls.compute_enclosing_box(cluster, padding=4.0)
                if box:
                    bx0, by0, bx1, by1 = box
                    annotations.append({
                        "page": int(page_number),
                        "x": float(bx0),
                        "y": float(by0),
                        "width": float(max(1.0, bx1 - bx0)),
                        "height": float(max(1.0, by1 - by0)),
                        "color": "#d97706",
                        "border": "solid",
                        "id": f"{id_prefix}_box_{page_number}_{c_idx}"
                    })

        return annotations

    @classmethod
    def render_page_with_highlights(
        cls,
        pdf_path: str,
        page_number: int,
        rects: Optional[List[List[float]]] = None,
        zoom: float = 1.0,
        base_dpi: int = 150,
        draw_box: bool = True,
        citation_id: Optional[int] = None,
        box_color: Tuple[float, float, float] = (0.91, 0.45, 0.0),
        box_width: float = 2.5
    ) -> Optional[bytes]:
        """
        Render a PDF page to high-resolution PNG bytes with highlighted evidence boxes.
        
        Highlights matched text lines with golden translucent fill and draws a prominent,
        crisp amber rectangular outline around the evidence block with an optional citation tag.
        """
        if not os.path.exists(pdf_path) or page_number < 1:
            return None

        try:
            doc = pymupdf.open(pdf_path)
        except Exception:
            return None

        try:
            if page_number > len(doc):
                return None
            page = doc[page_number - 1]

            if rects:
                # 1. Translucent highlight fill for each line/span
                for r in rects:
                    shape = page.new_shape()
                    shape.draw_rect(pymupdf.Rect(r[0] - 1, r[1] - 1, r[2] + 1, r[3] + 1))
                    shape.finish(
                        fill=(1.0, 0.92, 0.23),
                        fill_opacity=0.45,
                        stroke_opacity=0.0
                    )
                    shape.commit()

                # 2. Prominent enclosing box/rectangle around evidence cluster(s)
                if draw_box:
                    clusters = cls.cluster_rects(rects)
                    for c_idx, cluster in enumerate(clusters):
                        box = cls.compute_enclosing_box(cluster, padding=5.0)
                        if not box:
                            continue
                        bx0 = max(0.0, box[0])
                        by0 = max(0.0, box[1])
                        bx1 = min(float(page.rect.width), box[2])
                        by1 = min(float(page.rect.height), box[3])

                        box_shape = page.new_shape()
                        box_shape.draw_rect(pymupdf.Rect(bx0, by0, bx1, by1))
                        box_shape.finish(
                            color=box_color,
                            fill=(1.0, 0.92, 0.23),
                            fill_opacity=0.18,
                            width=box_width,
                            stroke_opacity=0.95
                        )
                        box_shape.commit()

                        # Draw small badge tag for citation ID
                        if citation_id is not None and c_idx == 0:
                            badge_text = f"[{citation_id}]"
                            badge_w = 20.0 + (len(str(citation_id)) - 1) * 6.0
                            badge_h = 12.0
                            badge_x0 = bx0
                            badge_y0 = max(0.0, by0 - badge_h)
                            badge_shape = page.new_shape()
                            badge_shape.draw_rect(pymupdf.Rect(badge_x0, badge_y0, badge_x0 + badge_w, badge_y0 + badge_h))
                            badge_shape.finish(fill=box_color, stroke_opacity=0.0)
                            badge_shape.insert_text(
                                (badge_x0 + 3.0, badge_y0 + 9.0),
                                badge_text,
                                fontsize=8,
                                color=(1.0, 1.0, 1.0)
                            )
                            badge_shape.commit()

            target_dpi = max(72, int(base_dpi * zoom))
            pix = page.get_pixmap(dpi=target_dpi)
            return pix.tobytes("png")
        finally:
            doc.close()


citation_engine = CitationResolutionService()
