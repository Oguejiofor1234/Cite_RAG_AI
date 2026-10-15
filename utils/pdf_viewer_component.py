import os
import json
import base64
from typing import List, Optional
import streamlit as st
import streamlit.components.v1 as components
import pymupdf


def render_pdfjs_viewer(
    pdf_path: str,
    page_number: int = 1,
    rects: Optional[List[List[float]]] = None,
    zoom: float = 1.0,
    citation_id: Optional[int] = None,
    height: int = 660,
    key: Optional[str] = None
):
    """
    Renders an interactive, vector-based PDF.js viewer with selectable text,
    precision bounding-box highlighting (amber/yellow rectangular outline with
    transparent fill), and automatic smooth scrolling to the highlighted region.
    """
    if not os.path.exists(pdf_path):
        st.warning(f"PDF file not found at: {pdf_path}")
        return

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as e:
        st.error(f"Failed to read PDF file: {str(e)}")
        return

    # Account for page rotation in visual coordinate mapping
    processed_rects = []
    page_rotation = 0
    try:
        doc = pymupdf.open(pdf_path)
        if 1 <= page_number <= len(doc):
            page = doc[page_number - 1]
            page_rotation = page.rotation
            if rects:
                if page_rotation != 0:
                    rot_matrix = page.rotation_matrix
                    for r in rects:
                        rot_r = pymupdf.Rect(r[0], r[1], r[2], r[3]) * rot_matrix
                        processed_rects.append([
                            round(rot_r.x0, 2),
                            round(rot_r.y0, 2),
                            round(rot_r.x1, 2),
                            round(rot_r.y1, 2)
                        ])
                else:
                    processed_rects = [[round(x, 2) for x in r] for r in rects]
        doc.close()
    except Exception:
        processed_rects = rects or []

    rects_json = json.dumps(processed_rects)
    cit_id_json = json.dumps(citation_id)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf_viewer.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  #viewer-container {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: auto;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 16px;
    background: #1e293b;
    scroll-behavior: smooth;
  }}
  .page-viewport-wrapper {{
    position: relative;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    background-color: #ffffff;
    border-radius: 4px;
    margin: 0 auto;
  }}
  #pdf-canvas {{
    display: block;
    border-radius: 4px;
  }}
  .textLayer {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    overflow: hidden;
    opacity: 0.25;
    line-height: 1.0;
    z-index: 5;
    border-radius: 4px;
  }}
  .evidence-layer {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
    z-index: 15;
    border-radius: 4px;
  }}
  .evidence-line-highlight {{
    position: absolute;
    background-color: rgba(254, 240, 138, 0.45);
    border-bottom: 2px solid #f59e0b;
    pointer-events: none;
    z-index: 10;
  }}
  .evidence-bounding-box {{
    position: absolute;
    border: 2.5px solid #d97706;
    background-color: rgba(245, 158, 11, 0.20);
    border-radius: 4px;
    box-shadow: 0 0 12px rgba(217, 119, 6, 0.4);
    pointer-events: none;
    z-index: 20;
    animation: pulse-border 2.5s infinite ease-in-out;
  }}
  @keyframes pulse-border {{
    0% {{ box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.5); }}
    70% {{ box-shadow: 0 0 0 8px rgba(217, 119, 6, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(217, 119, 6, 0); }}
  }}
  .evidence-badge {{
    position: absolute;
    top: -22px;
    left: -2px;
    background: #d97706;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 3px 3px 0 0;
    white-space: nowrap;
    box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    letter-spacing: 0.02em;
  }}
  #loading-indicator {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #94a3b8;
    font-size: 14px;
  }}
</style>
</head>
<body>
<div id="viewer-container">
  <div id="loading-indicator">Rendering page {page_number}...</div>
  <div id="page-wrapper" class="page-viewport-wrapper" style="display: none;">
    <canvas id="pdf-canvas"></canvas>
    <div id="text-layer" class="textLayer"></div>
    <div id="evidence-layer" class="evidence-layer"></div>
  </div>
</div>

<script>
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  async function initViewer() {{
    const b64Data = '{pdf_b64}';
    const targetPageNum = {page_number};
    const zoomScale = {zoom};
    const rects = {rects_json};
    const citationId = {cit_id_json};

    const raw = atob(b64Data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {{
      bytes[i] = raw.charCodeAt(i);
    }}

    const loadingTask = pdfjsLib.getDocument({{ data: bytes }});
    const pdfDoc = await loadingTask.promise;
    const page = await pdfDoc.getPage(targetPageNum);

    const viewport = page.getViewport({{ scale: zoomScale }});

    const canvas = document.getElementById('pdf-canvas');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext('2d');

    const renderTask = page.render({{
      canvasContext: ctx,
      viewport: viewport
    }});
    await renderTask.promise;

    const wrapper = document.getElementById('page-wrapper');
    wrapper.style.width = viewport.width + 'px';
    wrapper.style.height = viewport.height + 'px';
    wrapper.style.display = 'block';
    document.getElementById('loading-indicator').style.display = 'none';

    // Text Layer for text selection and search
    try {{
      const textContent = await page.getTextContent();
      const textLayerDiv = document.getElementById('text-layer');
      textLayerDiv.style.width = viewport.width + 'px';
      textLayerDiv.style.height = viewport.height + 'px';
      textLayerDiv.style.setProperty('--scale-factor', zoomScale);
      pdfjsLib.renderTextLayer({{
        textContentSource: textContent,
        container: textLayerDiv,
        viewport: viewport,
        textDivs: []
      }});
    }} catch (e) {{
      console.warn("Text layer render:", e);
    }}

    // Evidence Highlighting Layer
    const evLayer = document.getElementById('evidence-layer');
    if (rects && rects.length > 0) {{
      // 1. Line-by-line translucent highlights
      for (const r of rects) {{
        const left = r[0] * zoomScale;
        const top = r[1] * zoomScale;
        const width = Math.max(2, (r[2] - r[0]) * zoomScale);
        const height = Math.max(2, (r[3] - r[1]) * zoomScale);

        const lineDiv = document.createElement('div');
        lineDiv.className = 'evidence-line-highlight';
        lineDiv.style.left = left + 'px';
        lineDiv.style.top = top + 'px';
        lineDiv.style.width = width + 'px';
        lineDiv.style.height = height + 'px';
        evLayer.appendChild(lineDiv);
      }}

      // 2. Enclosing bounding box with amber rectangular outline & subtle transparent background
      const minX = Math.min(...rects.map(r => r[0])) * zoomScale - 4;
      const minY = Math.min(...rects.map(r => r[1])) * zoomScale - 4;
      const maxX = Math.max(...rects.map(r => r[2])) * zoomScale + 4;
      const maxY = Math.max(...rects.map(r => r[3])) * zoomScale + 4;

      const boxDiv = document.createElement('div');
      boxDiv.className = 'evidence-bounding-box';
      boxDiv.id = 'active-evidence-box';
      boxDiv.style.left = Math.max(0, minX) + 'px';
      boxDiv.style.top = Math.max(0, minY) + 'px';
      boxDiv.style.width = Math.max(10, maxX - minX) + 'px';
      boxDiv.style.height = Math.max(10, maxY - minY) + 'px';

      if (citationId !== null && citationId !== undefined) {{
        const badge = document.createElement('div');
        badge.className = 'evidence-badge';
        badge.innerText = `📍 [${{citationId}}] Evidence`;
        boxDiv.appendChild(badge);
      }}

      evLayer.appendChild(boxDiv);

      // 3. Automatic smooth scrolling to centered highlighted region
      setTimeout(() => {{
        const target = document.getElementById('active-evidence-box');
        if (target) {{
          target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}
      }}, 120);
    }}
  }}

  initViewer().catch(err => {{
    document.getElementById('loading-indicator').innerText = 'Viewer error: ' + err.message;
  }});
</script>
</body>
</html>"""

    components.html(html_code, height=height, scrolling=False)
