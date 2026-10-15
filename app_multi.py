import os
import glob
import uuid
import shutil
from typing import List, Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components

from models import (
    openai_llm,
    openai_embeddings,
    ollama_llm,
    ollama_embeddings,
    get_embedding_model_id,
    get_deployment_mode,
    get_openai_api_key,
)
from utils.security import sanitize_filename
from utils.pdf_loader import get_pdf_metadata
from utils.citation_engine import CitationResolutionService
from utils.pdf_viewer_component import render_pdfjs_viewer
from utils.vectorstore import check_embedding_compatibility
from rag.ingest_multi import (
    build_multi_ingest_graph,
)
from rag.query_multi import (
    build_multi_query_graph,
)


# =====================================================
# PAGE CONFIGURATION & RESPONSIVE THEME STYLING
# =====================================================
st.set_page_config(
    page_title="Cite_RAG_AI (Ogueji)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto"
)

# Professional responsive dashboard styling & Theme Consistency Tokens
st.markdown("""
<style>
    /* =====================================================
       THEME SYSTEM: LIGHT & DARK CONSISTENCY TOKENS
       ===================================================== */
    :root {
        --app-heading-color: #0f172a;
        --app-text-color: #1e293b;
        --app-text-muted: #475569;
        --app-card-bg: #f8fafc;
        --app-card-border: #e2e8f0;
        --app-card-active-bg: #eff6ff;
        --app-card-active-border: #2563eb;
        --app-banner-bg: #f0f7ff;
        --app-banner-border: #bfdbfe;
        --app-banner-accent: #2563eb;
        --app-alert-bg: #fffbeb;
        --app-alert-border: #fde68a;
        --app-alert-text: #92400e;
        --app-answer-bg: #fefce8;
        --app-answer-border: #fef08a;
        --app-answer-accent: #ea580c;
        --app-answer-text: #713f12;
        --app-snippet-bg: #f8fafc;
        --app-snippet-border: #e2e8f0;
        --app-snippet-text: #334155;
        --app-badge-bg: #f1f5f9;
        --app-badge-border: #cbd5e1;
        --app-badge-text: #0f172a;
        --app-header-border: #e2e8f0;
    }

    /* Streamlit / OS Dark Theme Detection via @media */
    @media (prefers-color-scheme: dark) {
        :root {
            --app-heading-color: #f8fafc;
            --app-text-color: #f1f5f9;
            --app-text-muted: #94a3b8;
            --app-card-bg: #1e293b;
            --app-card-border: #334155;
            --app-card-active-bg: rgba(59, 130, 246, 0.22);
            --app-card-active-border: #60a5fa;
            --app-banner-bg: rgba(37, 99, 235, 0.16);
            --app-banner-border: rgba(96, 165, 250, 0.35);
            --app-banner-accent: #60a5fa;
            --app-alert-bg: rgba(245, 158, 11, 0.18);
            --app-alert-border: rgba(245, 158, 11, 0.50);
            --app-alert-text: #fde68a;
            --app-answer-bg: rgba(245, 158, 11, 0.16);
            --app-answer-border: rgba(245, 158, 11, 0.45);
            --app-answer-accent: #f59e0b;
            --app-answer-text: #fef08a;
            --app-snippet-bg: #1e293b;
            --app-snippet-border: #334155;
            --app-snippet-text: #cbd5e1;
            --app-badge-bg: #334155;
            --app-badge-border: #475569;
            --app-badge-text: #f8fafc;
            --app-header-border: #334155;
        }
    }

    /* Explicit Dark Theme container selectors (matches data-theme, Streamlit dark classes) */
    html[data-theme="dark"],
    body[data-theme="dark"],
    .stApp[data-theme="dark"],
    [data-theme="dark"] .stApp {
        --app-heading-color: #f8fafc !important;
        --app-text-color: #f1f5f9 !important;
        --app-text-muted: #94a3b8 !important;
        --app-card-bg: #1e293b !important;
        --app-card-border: #334155 !important;
        --app-card-active-bg: rgba(59, 130, 246, 0.22) !important;
        --app-card-active-border: #60a5fa !important;
        --app-banner-bg: rgba(37, 99, 235, 0.16) !important;
        --app-banner-border: rgba(96, 165, 250, 0.35) !important;
        --app-banner-accent: #60a5fa !important;
        --app-alert-bg: rgba(245, 158, 11, 0.18) !important;
        --app-alert-border: rgba(245, 158, 11, 0.50) !important;
        --app-alert-text: #fde68a !important;
        --app-answer-bg: rgba(245, 158, 11, 0.16) !important;
        --app-answer-border: rgba(245, 158, 11, 0.45) !important;
        --app-answer-accent: #f59e0b !important;
        --app-answer-text: #fef08a !important;
        --app-snippet-bg: #1e293b !important;
        --app-snippet-border: #334155 !important;
        --app-snippet-text: #cbd5e1 !important;
        --app-badge-bg: #334155 !important;
        --app-badge-border: #475569 !important;
        --app-badge-text: #f8fafc !important;
        --app-header-border: #334155 !important;
    }

    /* Prevent page horizontal scrollbar across all viewports */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
        box-sizing: border-box !important;
    }

    /* Streamlit header transparency and layer alignment */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 10 !important;
    }

    /* Block container responsive padding & hygiene */
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    @media (max-width: 1024px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 3.25rem !important;
        }
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 3rem !important;
            padding-bottom: 1.5rem !important;
        }
    }

    /* Horizontal blocks & column flex hygiene: prevent children from clipping or blowing out widths */
    [data-testid="stHorizontalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        gap: 1rem !important;
        align-items: stretch !important;
    }
    [data-testid="stColumn"] {
        min-width: 0 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stColumn"] > div {
        min-width: 0 !important;
        width: 100% !important;
    }

    /* Top application header styling with contrast preservation */
    .app-main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 1.32rem;
        font-weight: 700;
        color: var(--app-heading-color, inherit) !important;
        margin-top: 0 !important;
        margin-bottom: 0.15rem !important;
        line-height: 1.35 !important;
        word-break: break-word;
    }

    /* Panel header styling with clean border and vertical alignment */
    .panel-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--app-heading-color, inherit) !important;
        margin-top: 0 !important;
        margin-bottom: 0.65rem !important;
        padding-top: 0 !important;
        padding-bottom: 0.35rem !important;
        line-height: 1.35 !important;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        border-bottom: 1.5px solid var(--app-header-border, #e2e8f0) !important;
    }

    /* Typography hygiene */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
        letter-spacing: -0.015em;
        line-height: 1.35 !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        color: var(--app-heading-color, inherit) !important;
    }

    /* Button wrapping and height resilience across narrow viewports */
    div[data-testid="stButton"] > button {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        min-height: 2.25rem !important;
        height: auto !important;
        padding: 0.35rem 0.55rem !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
    }

    /* Document card styling */
    .doc-card {
        background-color: var(--app-card-bg, #f8fafc) !important;
        border: 1px solid var(--app-card-border, #e2e8f0) !important;
        border-radius: 6px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.4rem;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .doc-card-active {
        border-color: var(--app-card-active-border, #3b82f6) !important;
        background-color: var(--app-card-active-bg, #eff6ff) !important;
    }
    .doc-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--app-heading-color, inherit) !important;
        overflow: hidden;
        text-overflow: ellipsis;
        word-break: break-word;
        line-height: 1.35;
    }
    .doc-meta {
        font-size: 0.75rem;
        color: var(--app-text-muted, #64748b) !important;
        margin-top: 0.2rem;
        word-break: break-word;
    }

    /* Chat message hygiene */
    [data-testid="stChatMessage"] {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        padding: 0.75rem 0.85rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.65rem !important;
    }
    [data-testid="stChatMessageContent"] {
        min-width: 0 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    /* Citation badge style */
    .citation-badge {
        display: inline-block;
        background-color: var(--app-badge-bg, #f1f5f9) !important;
        color: var(--app-badge-text, inherit) !important;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        border: 1px solid var(--app-badge-border, #cbd5e1) !important;
        margin-right: 0.35rem;
        margin-top: 0.25rem;
    }

    /* Warning badge for insufficient evidence */
    .evidence-alert {
        background-color: var(--app-alert-bg, #fffbeb) !important;
        color: var(--app-alert-text, #92400e) !important;
        border: 1px solid var(--app-alert-border, #fde68a) !important;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        word-break: break-word;
    }

    /* Active citation banner */
    .citation-banner {
        background-color: var(--app-banner-bg, #f0f7ff) !important;
        border: 1px solid var(--app-banner-border, #bfdbfe) !important;
        border-left: 4px solid var(--app-banner-accent, #2563eb) !important;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.65rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
        color: var(--app-text-color, inherit) !important;
        word-break: break-word;
    }

    /* Active cited answer claim box */
    .answer-box {
        background-color: var(--app-answer-bg, #fefce8) !important;
        border: 1px solid var(--app-answer-border, #fef08a) !important;
        border-left: 4px solid var(--app-answer-accent, #ea580c) !important;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        font-size: 0.84rem;
        line-height: 1.45;
        color: var(--app-answer-text, #713f12) !important;
        margin-top: 0.45rem;
        margin-bottom: 0.45rem;
        word-break: break-word;
    }

    /* Supporting snippet box */
    .evidence-snippet {
        background-color: var(--app-snippet-bg, #f8fafc) !important;
        border: 1px solid var(--app-snippet-border, #e2e8f0) !important;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.82rem;
        line-height: 1.45;
        color: var(--app-snippet-text, #334155) !important;
        margin-top: 0.5rem;
        max-height: 160px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* PDF Viewer image wrapper to strictly preserve aspect ratio and fit container */
    .stImage {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }
    .stImage img {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* Interactive PDF viewer iframe hygiene */
    iframe {
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Sidebar hygiene on medium screens */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--app-header-border, #e2e8f0) !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Tabs styling for mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.45rem 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Theme synchronization component to guarantee instant data-theme attribute binding
components.html(
    """
    <script>
    (function() {
        function syncTheme() {
            try {
                const parentDoc = window.parent.document;
                const app = parentDoc.querySelector('.stApp');
                if (!app) return;
                const bg = window.getComputedStyle(app).backgroundColor;
                const rgb = bg.match(/\\d+/g);
                if (rgb && rgb.length >= 3) {
                    const lum = 0.299 * parseInt(rgb[0]) + 0.587 * parseInt(rgb[1]) + 0.114 * parseInt(rgb[2]);
                    const theme = lum < 128 ? 'dark' : 'light';
                    if (parentDoc.documentElement.getAttribute('data-theme') !== theme) {
                        parentDoc.documentElement.setAttribute('data-theme', theme);
                    }
                    if (app.getAttribute('data-theme') !== theme) {
                        app.setAttribute('data-theme', theme);
                    }
                }
            } catch (e) {}
        }
        syncTheme();
        try {
            const parentDoc = window.parent.document;
            const observer = new MutationObserver(syncTheme);
            observer.observe(parentDoc.body, { attributes: true, attributeFilter: ['class', 'style'], subtree: false });
            setInterval(syncTheme, 400);
        } catch (e) {}
    })();
    </script>
    """,
    height=0,
    width=0
)

# Resource limits & deployment configuration
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", 15))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 60))
MAX_DOCS_PER_SESSION = int(os.environ.get("MAX_DOCS_PER_SESSION", 5))

deployment_mode = get_deployment_mode()

# =====================================================
# SESSION STATE INITIALIZATION & MULTI-USER ISOLATION
# =====================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:12]

SESSION_BASE_DIR = os.path.join("data", "multi", "sessions", st.session_state.session_id)
SESSION_UPLOADS_DIR = os.path.join(SESSION_BASE_DIR, "uploads")
SESSION_VS_DIR = os.path.join(SESSION_BASE_DIR, "vectorstores")
os.makedirs(SESSION_UPLOADS_DIR, exist_ok=True)
os.makedirs(SESSION_VS_DIR, exist_ok=True)


def cleanup_expired_sessions(base_dir: str = "data/multi/sessions", max_age_hours: int = 4):
    """Safely delete session folders older than max_age_hours to prevent disk growth."""
    if not os.path.exists(base_dir):
        return
    import time
    now = time.time()
    for item in os.listdir(base_dir):
        p = os.path.join(base_dir, item)
        if os.path.isdir(p) and item != st.session_state.get("session_id"):
            try:
                mtime = os.path.getmtime(p)
                if (now - mtime) > max_age_hours * 3600:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

cleanup_expired_sessions()


def auto_discover_documents(uploads_dir: str, vs_dir: str) -> List[Dict[str, Any]]:
    """Discover existing uploaded documents strictly within this user's session."""
    discovered = []
    pdf_files = glob.glob(os.path.join(uploads_dir, "*.pdf"))
    for f in pdf_files:
        base = os.path.basename(f)
        parts = base.split("_", 1)
        if len(parts) == 2 and len(parts[0]) >= 8:
            doc_id, original_name = parts
        else:
            doc_id = str(uuid.uuid4())
            original_name = base

        vs_path = os.path.join(vs_dir, f"{doc_id}_index")
        is_indexed = os.path.exists(vs_path)

        # Get metadata if possible
        try:
            meta = get_pdf_metadata(f)
            page_count = meta.get("page_count", 1)
        except Exception:
            page_count = 1

        discovered.append({
            "id": doc_id,
            "name": original_name,
            "path": f,
            "page_count": page_count,
            "is_indexed": is_indexed,
            "status": "Ready" if is_indexed else "Not Indexed"
        })
    return discovered


if "docs" not in st.session_state:
    st.session_state.docs = auto_discover_documents(SESSION_UPLOADS_DIR, SESSION_VS_DIR)

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = st.session_state.docs[0]["id"] if st.session_state.docs else None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "active_citation" not in st.session_state:
    st.session_state.active_citation = None

if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = 1

if "tb_input_page" not in st.session_state:
    st.session_state.tb_input_page = 1


def set_viewer_page(p: int):
    st.session_state.viewer_page = int(p)
    st.session_state.tb_input_page = int(p)

if "viewer_zoom" not in st.session_state:
    st.session_state.viewer_zoom = 1.0

if "viewer_mode" not in st.session_state:
    st.session_state.viewer_mode = "Interactive PDF Reader"

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "use_hybrid" not in st.session_state:
    st.session_state.use_hybrid = True

if "cross_doc_mode" not in st.session_state:
    st.session_state.cross_doc_mode = False

if "backend_choice" not in st.session_state:
    if deployment_mode == "cloud":
        st.session_state.backend_choice = "OpenAI (Cloud)"
    else:
        st.session_state.backend_choice = "Ollama (Local)"

if "layout_mode" not in st.session_state:
    st.session_state.layout_mode = "desktop"


# Helper: Get current selected document
def get_selected_doc() -> Optional[Dict[str, Any]]:
    if not st.session_state.selected_doc_id:
        return None
    for d in st.session_state.docs:
        if d["id"] == st.session_state.selected_doc_id:
            return d
    return None


selected_doc = get_selected_doc()


# Helper: Initialize active models and embeddings
def get_active_models():
    mode = get_deployment_mode()
    choice = st.session_state.backend_choice

    # Enforce Cloud invariant: never attempt localhost Ollama connection in cloud mode!
    if mode == "cloud" and choice != "OpenAI (Cloud)":
        choice = "OpenAI (Cloud)"
        st.session_state.backend_choice = "OpenAI (Cloud)"

    if choice == "OpenAI (Cloud)":
        api_key = get_openai_api_key(st.session_state.get("openai_api_key_input"))
        if not api_key:
            return None, None, "OpenAI API Key Required"
        try:
            emb = openai_embeddings(api_key=api_key)
            model_llm = openai_llm(api_key=api_key)
            model_id = get_embedding_model_id(emb)
            return emb, model_llm, model_id
        except Exception as e:
            return None, None, f"OpenAI Config Error: {str(e)}"
    else:
        # Ollama (Local) - only permissible in local deployment mode
        if mode == "cloud":
            return None, None, "Ollama Disabled in Cloud Mode"
        try:
            emb = ollama_embeddings()
            model_llm = ollama_llm()
            model_id = get_embedding_model_id(emb)
            return emb, model_llm, model_id
        except Exception as e:
            return None, None, f"Ollama Connection Error: {str(e)}"


embeddings, llm, active_model_id = get_active_models()



# =====================================================
# RESPONSIVE LAYOUT RESOLUTION & TOP CONTROLS
# =====================================================
# Determine layout mode from query parameter or session state (defaults to desktop)
query_layout = st.query_params.get("layout", "").lower()
if query_layout in ["desktop", "medium", "mobile"]:
    active_layout = query_layout
    st.session_state.layout_mode = query_layout
elif "layout_mode" in st.session_state:
    active_layout = st.session_state.layout_mode
else:
    active_layout = "desktop"
    st.session_state.layout_mode = "desktop"

# Top Application Header with responsive layout toggle
layout_map = {
    "desktop": "🖥️ Desktop (3-Panel)",
    "medium": "💻 Medium (Sidebar)",
    "mobile": "📱 Mobile (Tabs)"
}
reverse_map = {v: k for k, v in layout_map.items()}
curr_idx = 0 if active_layout == "desktop" else (1 if active_layout == "medium" else 2)

if active_layout == "mobile":
    st.markdown('<div class="app-main-title">📄 Cite_RAG_AI (Ogueji) — Agentic Document Intelligence & Precision Evidence Retrieval</div>', unsafe_allow_html=True)
    st.caption("Enterprise multi-document research with verifiable precision visual PDF citations.")
    selected_layout_label = st.radio(
        "Layout Mode",
        options=["🖥️ Desktop (3-Panel)", "💻 Medium (Sidebar)", "📱 Mobile (Tabs)"],
        index=curr_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="layout_switcher_radio"
    )
else:
    top_col1, top_col2 = st.columns([3.2, 1.8], gap="medium")
    with top_col1:
        st.markdown('<div class="app-main-title">📄 Cite_RAG_AI (Ogueji) — Agentic Document Intelligence & Precision Evidence Retrieval</div>', unsafe_allow_html=True)
        st.caption("Enterprise multi-document research with verifiable precision visual PDF citations.")
    with top_col2:
        selected_layout_label = st.radio(
            "Layout Mode",
            options=["🖥️ Desktop (3-Panel)", "💻 Medium (Sidebar)", "📱 Mobile (Tabs)"],
            index=curr_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="layout_switcher_radio"
        )

chosen_layout = reverse_map[selected_layout_label]
if chosen_layout != active_layout:
    st.session_state.layout_mode = chosen_layout
    st.query_params["layout"] = chosen_layout
    st.rerun()


# =====================================================
# MODULAR PANEL RENDERERS
# =====================================================

def render_document_library(in_sidebar: bool = False):
    """Render Document Library & System Controls panel with strict session isolation."""
    st.markdown('<div class="panel-header">🗂️ Document Library</div>', unsafe_allow_html=True)

    # Backend Selector
    if deployment_mode == "cloud":
        st.radio(
            "AI Engine (Cloud Mode)",
            ["OpenAI (Cloud)"],
            index=0,
            horizontal=True,
            key=f"backend_sel_{'sb' if in_sidebar else 'main'}",
            help="Streamlit Cloud uses OpenAI for cloud generation and embeddings. Local Ollama is disabled."
        )
        st.session_state.backend_choice = "OpenAI (Cloud)"
    else:
        backend_opts = ["Ollama (Local)", "OpenAI (Cloud)"]
        backend_idx = 0 if st.session_state.backend_choice == "Ollama (Local)" else 1
        new_backend = st.radio(
            "AI Engine",
            backend_opts,
            index=backend_idx,
            horizontal=True,
            key=f"backend_sel_{'sb' if in_sidebar else 'main'}"
        )
        if new_backend != st.session_state.backend_choice:
            st.session_state.backend_choice = new_backend
            st.rerun()

    # OpenAI API Key Configuration
    if st.session_state.backend_choice == "OpenAI (Cloud)":
        curr_key = get_openai_api_key(st.session_state.get("openai_api_key_input"))
        if not curr_key:
            st.warning("⚠️ OpenAI API key required.")
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Key is retained only in this session's memory and is never committed to disk.",
                key=f"openai_key_{'sb' if in_sidebar else 'main'}"
            )
            if api_key_input:
                st.session_state.openai_api_key_input = api_key_input.strip()
                st.rerun()
        else:
            masked = f"{curr_key[:7]}...{curr_key[-4:]}" if len(curr_key) > 12 else "configured"
            st.caption(f"🔑 OpenAI Key: `{masked}`")

    st.caption(f"Active Embedding: `{active_model_id}`")

    # Retrieval Settings
    with st.expander("⚙️ Retrieval Parameters", expanded=False):
        st.session_state.use_hybrid = st.checkbox(
            "Hybrid BM25 + Vector Search",
            value=st.session_state.use_hybrid,
            help="Combines sparse keyword matching with dense embeddings using RRF",
            key=f"hybrid_chk_{'sb' if in_sidebar else 'main'}"
        )
        st.session_state.top_k = st.slider(
            "Evidence Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=st.session_state.top_k,
            key=f"topk_slider_{'sb' if in_sidebar else 'main'}"
        )
        st.session_state.cross_doc_mode = st.checkbox(
            "Search Across All Documents",
            value=st.session_state.cross_doc_mode,
            help="Default is strict per-document isolation. Enable to search across the entire session library.",
            key=f"crossdoc_chk_{'sb' if in_sidebar else 'main'}"
        )

    st.markdown("---")

    # Upload Multiple PDFs with Resource Limits & Safe Validation
    st.markdown("#### Upload Documents")
    st.caption(f"Limits: max {MAX_DOCS_PER_SESSION} docs, {MAX_FILE_SIZE_MB}MB/file, {MAX_PDF_PAGES} pages/doc")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{'sb' if in_sidebar else 'main'}"
    )

    if uploaded_files:
        if st.button("Index Uploaded Files", use_container_width=True, type="primary", key=f"btn_idx_{'sb' if in_sidebar else 'main'}"):
            if embeddings is None or llm is None:
                st.error("AI Engine is not configured. Please provide an OpenAI API Key above to begin indexing.")
            elif len(st.session_state.docs) >= MAX_DOCS_PER_SESSION:
                st.error(f"Session limit reached ({MAX_DOCS_PER_SESSION} documents). Please remove a document first.")
            else:
                progress_bar = st.progress(0, text="Starting ingestion...")
                ingest_graph = build_multi_ingest_graph(embeddings, active_model_id, base_dir=SESSION_VS_DIR)

                for idx, file in enumerate(uploaded_files):
                    if len(st.session_state.docs) >= MAX_DOCS_PER_SESSION:
                        st.warning(f"Reached limit of {MAX_DOCS_PER_SESSION} documents. Skipping {file.name}.")
                        break

                    file_bytes = file.read()
                    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                        st.error(f"'{file.name}' exceeds the {MAX_FILE_SIZE_MB}MB size limit ({len(file_bytes)/(1024*1024):.1f}MB).")
                        continue

                    clean_name = sanitize_filename(file.name)
                    doc_id = str(uuid.uuid4())
                    save_path = os.path.join(SESSION_UPLOADS_DIR, f"{doc_id}_{clean_name}")

                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

                    # Validate PDF readability and extract metadata
                    try:
                        meta = get_pdf_metadata(save_path)
                    except Exception as e:
                        st.error(f"Failed to read '{clean_name}': Invalid or corrupted PDF ({str(e)}).")
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        continue

                    page_count = meta.get("page_count", 0)
                    if page_count > MAX_PDF_PAGES:
                        st.error(f"'{clean_name}' has {page_count} pages, exceeding the {MAX_PDF_PAGES}-page limit for low-cost cloud deployment.")
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        continue

                    if meta.get("is_scanned", False):
                        st.warning(f"Notice: '{clean_name}' contains no extractable digital text (scanned PDF). Lexical search and citations will be limited.")

                    progress_bar.progress(
                        (idx) / len(uploaded_files),
                        text=f"Indexing {clean_name} ({page_count} pages)..."
                    )

                    try:
                        res = ingest_graph.invoke({
                            "path": save_path,
                            "doc_id": doc_id
                        })
                        st.session_state.docs.append({
                            "id": doc_id,
                            "name": clean_name,
                            "path": save_path,
                            "page_count": page_count,
                            "is_indexed": True,
                            "status": "Ready"
                        })
                        st.session_state.chat_history[doc_id] = []
                        if st.session_state.selected_doc_id is None:
                            st.session_state.selected_doc_id = doc_id
                    except Exception as e:
                        st.error(f"Failed to ingest {clean_name}: {str(e)}")
                        if os.path.exists(save_path):
                            os.remove(save_path)

                progress_bar.progress(1.0, text="Ingestion complete.")
                st.success("Documents processed successfully.")
                st.rerun()

    st.markdown("---")

    # Document List & Selection (scrollable container for hygiene)
    st.markdown("#### Library Documents")
    if not st.session_state.docs:
        st.info("No documents uploaded in this session yet. Upload a PDF above to begin.")
    else:
        with st.container(height=340 if not in_sidebar else 420):
            for doc in list(st.session_state.docs):
                is_active = (doc["id"] == st.session_state.selected_doc_id)
                card_class = "doc-card doc-card-active" if is_active else "doc-card"
                badge = "🟢 Ready" if doc.get("is_indexed") else "⚪ Unindexed"
                status_text = f"{badge} • {doc.get('page_count', 1)} pages"
                if is_active:
                    status_text += " • <b>Active Context</b>"

                st.markdown(
                    f'<div class="{card_class}">'
                    f'<div class="doc-title" title="{doc["name"]}">{doc["name"]}</div>'
                    f'<div class="doc-meta">{status_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                col_act, col_del = st.columns([3, 1])
                with col_act:
                    if is_active:
                        st.button("✓ Active", key=f"act_{doc['id']}_{'sb' if in_sidebar else 'main'}", disabled=True, use_container_width=True)
                    else:
                        if st.button("Select", key=f"sel_{doc['id']}_{'sb' if in_sidebar else 'main'}", use_container_width=True):
                            st.session_state.selected_doc_id = doc["id"]
                            st.session_state.active_citation = None
                            set_viewer_page(1)
                            st.rerun()

                with col_del:
                    if st.button("✕", key=f"del_{doc['id']}_{'sb' if in_sidebar else 'main'}", help="Delete document", use_container_width=True):
                        # Remove files and vector store from session sandbox
                        if os.path.exists(doc["path"]):
                            try:
                                os.remove(doc["path"])
                            except Exception:
                                pass
                        vs_dir = os.path.join(SESSION_VS_DIR, f"{doc['id']}_index")
                        if os.path.exists(vs_dir):
                            shutil.rmtree(vs_dir, ignore_errors=True)

                        st.session_state.docs = [d for d in st.session_state.docs if d["id"] != doc["id"]]
                        if doc["id"] in st.session_state.chat_history:
                            del st.session_state.chat_history[doc["id"]]
                        if st.session_state.selected_doc_id == doc["id"]:
                            st.session_state.selected_doc_id = st.session_state.docs[0]["id"] if st.session_state.docs else None
                            st.session_state.active_citation = None
                            set_viewer_page(1)
                        st.rerun()

        # Session reset / cleanup button
        st.markdown("")
        if st.button("🗑️ Reset Session & Clear Files", key=f"btn_reset_session_{'sb' if in_sidebar else 'main'}", help="Clear all documents and indexes from this session", use_container_width=True):
            shutil.rmtree(SESSION_BASE_DIR, ignore_errors=True)
            os.makedirs(SESSION_UPLOADS_DIR, exist_ok=True)
            os.makedirs(SESSION_VS_DIR, exist_ok=True)
            st.session_state.docs = []
            st.session_state.selected_doc_id = None
            st.session_state.chat_history = {}
            st.session_state.active_citation = None
            set_viewer_page(1)
            st.success("Session reset.")
            st.rerun()


def render_ai_assistant():
    """Render AI Research Assistant panel with chat and grounded citations."""
    st.markdown('<div class="panel-header">💬 AI Research Assistant</div>', unsafe_allow_html=True)

    if selected_doc:
        st.caption(f"Active Context: **{selected_doc['name']}** (ID: `{selected_doc['id'][:8]}`)")

        # Verify index model compatibility
        vs_dir = os.path.join(SESSION_VS_DIR, f"{selected_doc['id']}_index")
        if os.path.exists(vs_dir):
            is_compat, compat_msg = check_embedding_compatibility(vs_dir, active_model_id)
            if not is_compat:
                st.warning(compat_msg)
                if st.button("Re-index Document for Active Model", key=f"reindex_{selected_doc['id']}"):
                    if embeddings is None:
                        st.error("Cannot re-index without configured embeddings.")
                    else:
                        with st.spinner("Re-indexing document..."):
                            ingest_graph = build_multi_ingest_graph(embeddings, active_model_id, base_dir=SESSION_VS_DIR)
                            ingest_graph.invoke({"path": selected_doc["path"], "doc_id": selected_doc["id"]})
                        st.success("Re-indexed successfully.")
                        st.rerun()

        # Chat History Container with independent vertical scrolling
        history = st.session_state.chat_history.get(selected_doc["id"], [])

        with st.container(height=520):
            if not history:
                st.info("Ask any factual question about this document. All assertions are linked to source pages.")
            else:
                for msg_idx, msg in enumerate(history):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                        # Render citations with responsive column wrapping
                        citations = msg.get("citations", [])
                        if citations:
                            st.markdown("**Evidence Citations:**")
                            num_cit_cols = min(len(citations), 3)
                            cit_cols = st.columns(num_cit_cols)
                            for c_idx, cit in enumerate(citations):
                                with cit_cols[c_idx % num_cit_cols]:
                                    p_num = cit.get("page_number", 1)
                                    btn_label = f"📍 [{cit.get('citation_id', c_idx+1)}] Page {p_num}"
                                    if st.button(btn_label, key=f"cit_btn_{msg_idx}_{c_idx}", use_container_width=True):
                                        if cit.get("doc_id"):
                                            st.session_state.selected_doc_id = cit["doc_id"]
                                        set_viewer_page(p_num)
                                        st.session_state.active_citation = cit
                                        st.rerun()

                        # Insufficient evidence alert
                        if not msg.get("has_sufficient_evidence", True):
                            st.markdown(
                                '<div class="evidence-alert">⚠️ <b>Notice:</b> The provided document context '
                                'does not contain sufficient verifiable evidence to substantiate this response.</div>',
                                unsafe_allow_html=True
                            )

        # Quick research follow-up prompts
        st.markdown("---")
        quick_cols = st.columns(3)
        quick_query = None
        if quick_cols[0].button("Summary", help="Summarize primary contributions and findings", use_container_width=True, key="quick_contrib"):
            quick_query = "Summarize the primary research contributions and findings of this document."
        if quick_cols[1].button("Methodology", help="What methodology or architecture is described?", use_container_width=True, key="quick_method"):
            quick_query = "What methodology or experimental architecture is described?"
        if quick_cols[2].button("Limitations", help="What limitations, weaknesses, or open issues are identified?", use_container_width=True, key="quick_limits"):
            quick_query = "What limitations, weaknesses, or open issues are identified?"

        # Chat Input
        user_input = st.chat_input("Enter research question...", key="chat_user_query_input")
        effective_query = user_input or quick_query

        if effective_query:
            if embeddings is None or llm is None:
                st.error("AI Assistant is not configured. Please supply an OpenAI API key in the Document Library panel.")
            else:
                if selected_doc["id"] not in st.session_state.chat_history:
                    st.session_state.chat_history[selected_doc["id"]] = []

                st.session_state.chat_history[selected_doc["id"]].append({
                    "role": "user",
                    "content": effective_query
                })

                query_graph = build_multi_query_graph(embeddings, llm, active_model_id, base_dir=SESSION_VS_DIR)

                with st.spinner("Retrieving evidence and formulating answer..."):
                    try:
                        doc_name_map = {d["id"]: d["name"] for d in st.session_state.docs}
                        result = query_graph.invoke({
                            "question": effective_query,
                            "doc_id": selected_doc["id"],
                            "top_k": st.session_state.top_k,
                            "use_hybrid": st.session_state.use_hybrid,
                            "cross_doc_mode": st.session_state.cross_doc_mode,
                            "available_doc_ids": [d["id"] for d in st.session_state.docs],
                            "doc_name_map": doc_name_map,
                            "chat_history": st.session_state.chat_history[selected_doc["id"]]
                        })

                        citations = result.get("citations", [])
                        st.session_state.chat_history[selected_doc["id"]].append({
                            "role": "assistant",
                            "content": result.get("answer", ""),
                            "citations": citations,
                            "has_sufficient_evidence": result.get("has_sufficient_evidence", True),
                            "pages": result.get("pages", [])
                        })

                        # Automatically navigate viewer to first citation if available
                        if citations:
                            first_cit = citations[0]
                            if first_cit.get("doc_id"):
                                st.session_state.selected_doc_id = first_cit["doc_id"]
                            st.session_state.active_citation = first_cit
                            set_viewer_page(first_cit.get("page_number", 1))

                        st.rerun()

                    except Exception as e:
                        st.error(f"Query execution failed: {str(e)}")
    else:
        st.info("Select or upload a document from the library panel to begin research.")


def render_evidence_viewer():
    """Render Precision Evidence Viewer panel with PyMuPDF highlights and page navigation."""
    st.markdown('<div class="panel-header">🔎 Precision Evidence Viewer</div>', unsafe_allow_html=True)

    # Determine which document to display (honors active citation's doc_id)
    doc_to_view = selected_doc
    active_cit = st.session_state.active_citation
    if active_cit and active_cit.get("doc_id"):
        target_doc_id = active_cit["doc_id"]
        for d in st.session_state.docs:
            if d["id"] == target_doc_id:
                doc_to_view = d
                break

    if doc_to_view and os.path.exists(doc_to_view["path"]):
        total_pages = doc_to_view.get("page_count", 1)

        # Navigation callbacks to guarantee bulletproof state synchronization
        def on_prev_page():
            new_p = max(1, st.session_state.viewer_page - 1)
            set_viewer_page(new_p)

        def on_next_page():
            new_p = min(total_pages, st.session_state.viewer_page + 1)
            set_viewer_page(new_p)

        def on_page_input_change():
            if "tb_input_page" in st.session_state:
                st.session_state.viewer_page = int(st.session_state.tb_input_page)

        # Viewer Toolbar (Navigation in Row 1; Zoom & Mode in Row 2)
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1.4, 1], gap="small")
        with nav_col1:
            st.button(
                "◀ Prev",
                disabled=(st.session_state.viewer_page <= 1),
                use_container_width=True,
                key="tb_btn_prev",
                on_click=on_prev_page
            )

        with nav_col2:
            st.number_input(
                f"Page (of {total_pages})",
                min_value=1,
                max_value=max(1, total_pages),
                step=1,
                label_visibility="collapsed",
                key="tb_input_page",
                on_change=on_page_input_change
            )

        with nav_col3:
            st.button(
                "Next ▶",
                disabled=(st.session_state.viewer_page >= total_pages),
                use_container_width=True,
                key="tb_btn_next",
                on_click=on_next_page
            )

        zm_col, mode_col = st.columns([1.2, 2.2], gap="small")
        with zm_col:
            zoom_options = ["100%", "125%", "150%"]
            curr_zoom_label = f"{int(st.session_state.viewer_zoom * 100)}%"
            zoom_idx = zoom_options.index(curr_zoom_label) if curr_zoom_label in zoom_options else 0
            zoom_choice = st.selectbox(
                "Zoom",
                zoom_options,
                index=zoom_idx,
                label_visibility="collapsed",
                key="tb_select_zoom"
            )
            st.session_state.viewer_zoom = float(zoom_choice.replace("%", "")) / 100.0

        with mode_col:
            mode_options = ["Interactive PDF Reader", "Precision Evidence Image"]
            m_idx = mode_options.index(st.session_state.viewer_mode) if st.session_state.viewer_mode in mode_options else 0
            viewer_mode = st.radio(
                "Viewer Mode",
                mode_options,
                index=m_idx,
                horizontal=True,
                label_visibility="collapsed",
                key="tb_radio_mode"
            )
            st.session_state.viewer_mode = viewer_mode

        # Active Citation Notification Banner
        matched_rects = []

        if active_cit:
            cit_page = active_cit.get("page_number", 1)
            cit_id = active_cit.get("citation_id", 1)

            if cit_page == st.session_state.viewer_page:
                raw_boxes = active_cit.get("bounding_boxes", [])
                snippet = active_cit.get("source_text") or active_cit.get("snippet", "")

                # Priority: Resolve coordinates from the retrieved source passage
                resolved = CitationResolutionService.resolve_citation_rects(
                    pdf_path=doc_to_view["path"],
                    page_number=cit_page,
                    snippet=snippet
                )
                if resolved:
                    matched_rects = resolved
                elif raw_boxes:
                    matched_rects = raw_boxes

                status_label = "Highlighted on page" if matched_rects else "Position unverified (displaying snippet below)"
                st.markdown(
                    f'<div class="citation-banner"><b>Citation [{cit_id}]</b> • Page {cit_page} • '
                    f'<i>{status_label}</i></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="citation-banner">Citation [{cit_id}] is on <b>Page {cit_page}</b>.</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Jump to Cited Page {cit_page}", key="jump_to_cited_page", use_container_width=True):
                    set_viewer_page(cit_page)
                    st.rerun()

        # Scrollable Viewer Content Container (independent vertical scrolling & proper aspect ratio)
        with st.container(height=660):
            is_active_page = bool(active_cit and active_cit.get("page_number", 1) == st.session_state.viewer_page)
            curr_cit_id = active_cit.get("citation_id") if (is_active_page and active_cit) else None

            if st.session_state.viewer_mode == "Interactive PDF Reader":
                # High-fidelity vector PDF.js viewer with selectable text & amber bounding boxes
                render_pdfjs_viewer(
                    pdf_path=doc_to_view["path"],
                    page_number=st.session_state.viewer_page,
                    rects=matched_rects if matched_rects else None,
                    zoom=st.session_state.viewer_zoom,
                    citation_id=curr_cit_id,
                    height=640,
                    key=f"pdfjs_{doc_to_view['id']}_{st.session_state.viewer_page}"
                )
            else:
                # Precision Evidence Image mode: PyMuPDF high-DPI rendering with amber rectangle outlines & badges
                page_img = CitationResolutionService.render_page_with_highlights(
                    pdf_path=doc_to_view["path"],
                    page_number=st.session_state.viewer_page,
                    rects=matched_rects if matched_rects else None,
                    zoom=st.session_state.viewer_zoom,
                    base_dpi=150,
                    draw_box=True,
                    citation_id=curr_cit_id
                )

                if page_img:
                    st.image(page_img, use_container_width=True)
                else:
                    st.warning(f"Could not render page {st.session_state.viewer_page}.")

        # Supporting Evidence Snippet Box
        if active_cit:
            snippet_text = active_cit.get("source_text") or active_cit.get("snippet", "")
            chunk_id = active_cit.get("chunk_id", "N/A")
            score = active_cit.get("score", 0.0)
            answer_claim = active_cit.get("answer_claim", "")

            if answer_claim:
                st.markdown(
                    f'<div class="answer-box"><b>🎯 Cited Answer Claim:</b><br>{answer_claim}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"**Supporting Source Snippet (Chunk: `{chunk_id}`, Score: `{score}`):**")
            st.markdown(
                f'<div class="evidence-snippet">{snippet_text}</div>',
                unsafe_allow_html=True
            )
            if st.button("Clear Active Citation", key="btn_clear_citation", use_container_width=True):
                st.session_state.active_citation = None
                st.rerun()

    else:
        st.info("Select a document to inspect pages and precision evidence.")


# =====================================================
# RESPONSIVE THREE-PANEL RENDERING ORCHESTRATION
# =====================================================

if active_layout == "desktop":
    # Wide desktop screens: display all three panels side by side with responsive proportions
    col_lib, col_chat, col_viewer = st.columns([1.05, 1.6, 1.6], gap="medium")
    with col_lib:
        render_document_library(in_sidebar=False)
    with col_chat:
        render_ai_assistant()
    with col_viewer:
        render_evidence_viewer()

elif active_layout == "medium":
    # Medium screens: prioritize AI Assistant and PDF Viewer in main view; move Document Library into collapsible sidebar
    with st.sidebar:
        render_document_library(in_sidebar=True)

    col_chat, col_viewer = st.columns([1.05, 1.0], gap="medium")
    with col_chat:
        render_ai_assistant()
    with col_viewer:
        render_evidence_viewer()

else:  # mobile
    # Mobile screens: navigation tabs providing 100% width and clean navigation
    tab_chat, tab_viewer, tab_lib = st.tabs([
        "💬 AI Assistant",
        "🔎 Precision Evidence Viewer",
        "🗂️ Document Library"
    ])
    with tab_chat:
        render_ai_assistant()
    with tab_viewer:
        render_evidence_viewer()
    with tab_lib:
        render_document_library(in_sidebar=False)
