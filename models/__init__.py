import os
from typing import Optional
from .openai_config import openai_llm, openai_embeddings
from .ollama_config import ollama_llm, ollama_embeddings


def get_deployment_mode() -> str:
    """
    Determine deployment mode: 'cloud' or 'local'.
    Priority:
    1. DEPLOYMENT_MODE in st.secrets (if Streamlit running)
    2. DEPLOYMENT_MODE environment variable
    3. Auto-detection of Streamlit Cloud environment variables
    4. Default: 'local'
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "DEPLOYMENT_MODE" in st.secrets:
            mode = str(st.secrets["DEPLOYMENT_MODE"]).strip().lower()
            if mode in ["cloud", "local"]:
                return mode
    except Exception:
        pass

    env_mode = os.environ.get("DEPLOYMENT_MODE", "").strip().lower()
    if env_mode in ["cloud", "local"]:
        return env_mode

    # Auto-detect Streamlit Community Cloud hosting flags
    if any(k in os.environ for k in ["STREAMLIT_SHARING_HOST", "STREAMLIT_SERVER_BASE_URL_PATH", "IS_STREAMLIT_CLOUD"]):
        return "cloud"

    return "local"


def get_openai_api_key(session_override: Optional[str] = None) -> str:
    """
    Retrieve OpenAI API key with precedence:
    1. Session override from UI text input
    2. Streamlit secrets (st.secrets["OPENAI_API_KEY"])
    3. Environment variable (os.environ["OPENAI_API_KEY"])
    """
    if session_override and session_override.strip():
        return session_override.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            val = str(st.secrets["OPENAI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass

    return os.environ.get("OPENAI_API_KEY", "").strip()


def get_embedding_model_id(embeddings) -> str:
    """
    Return a canonical identifier string for the embeddings instance.
    Examples: 'openai:text-embedding-3-small', 'ollama:nomic-embed-text'
    """
    if embeddings is None:
        return "none"
    if hasattr(embeddings, "model"):
        model_name = getattr(embeddings, "model")
        class_name = embeddings.__class__.__name__.lower()
        if "openai" in class_name:
            return f"openai:{model_name}"
        elif "ollama" in class_name:
            return f"ollama:{model_name}"
        return f"{class_name}:{model_name}"
    return embeddings.__class__.__name__


__all__ = [
    "openai_llm",
    "openai_embeddings",
    "ollama_llm",
    "ollama_embeddings",
    "get_embedding_model_id",
    "get_deployment_mode",
    "get_openai_api_key",
]

