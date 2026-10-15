import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from models import (
    openai_llm,
    openai_embeddings,
    ollama_llm,
    ollama_embeddings,
)

from rag.ingest_single import build_single_ingest_graph
from rag.query_single import build_single_query_graph


# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Interactive Reader",
    layout="wide"
)

# =====================================================
# Create folders
# =====================================================
os.makedirs("data/single/uploads", exist_ok=True)
os.makedirs("data/single/vectorstore", exist_ok=True)

# =====================================================
# Session State
# =====================================================
if "messages" not in st.session_state: #
    st.session_state.messages = []

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "index_built" not in st.session_state:
    st.session_state.index_built = False


# =====================================================
# Two-column layout
# =====================================================
left_col, right_col = st.columns([1, 1.2])

# =====================================================
# LEFT PANEL
# =====================================================
with left_col:

    st.title("Single PDF Interactive RAG Reader 📚")

    # -----------------------------------------
    # Backend Selection
    # -----------------------------------------
    mode = st.radio(
        "Choose backend",
        ["Ollama", "OpenAI"],
        horizontal=True
    )

    if mode == "Ollama":
        llm = ollama_llm()
        embeddings = ollama_embeddings()
    else:
        llm = openai_llm()
        embeddings = openai_embeddings()

    st.markdown("### Ask a question from the PDF")

    # -----------------------------------------
    # Question Input
    # -----------------------------------------
    question = st.chat_input(
        "Ask anything here and press Enter..."
    )   

    # -----------------------------------------
    # Ask Question
    # -----------------------------------------
    if question and st.session_state.index_built:

        query_graph = build_single_query_graph(
            embeddings,
            llm
        )

        result = query_graph.invoke(
            {"question": question}
        )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    f"{result['answer']}\n\n📄 Pages: {result['pages']}"
            }
        )

    # -----------------------------------------
    # Chat History
    # -----------------------------------------
    chat_container = st.container()

    with chat_container:

        for msg in st.session_state.messages:

            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    st.markdown("---")

    # -----------------------------------------
    # Upload Section
    # -----------------------------------------
    st.subheader("Your documents")

    uploaded = st.file_uploader(
        "Upload your PDF here and click Process",
        type=["pdf"]
    )

    if uploaded:

        pdf_path = os.path.join(
            "data/single/uploads",
            uploaded.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.session_state.pdf_path = pdf_path

        st.success(f"{uploaded.name} uploaded")

        if st.button("Process"):

            with st.spinner("Building index..."):

                ingest_graph = build_single_ingest_graph(
                    embeddings
                )

                ingest_graph.invoke(
                    {"path": pdf_path}
                )

            st.session_state.index_built = True

            st.success(
                "Index built successfully! You can now ask questions."
            )


# =====================================================
# RIGHT PANEL
# =====================================================
with right_col:

    if st.session_state.pdf_path:

        pdf_viewer(
            st.session_state.pdf_path,
            width="100%",
            height=1000
        )

    else:

        st.info(
            "Upload a PDF to see the preview here."
        )
