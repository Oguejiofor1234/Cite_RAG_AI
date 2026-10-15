import os
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def openai_llm(api_key: Optional[str] = None, model: str = "gpt-4o-mini", temperature: float = 0.0):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key is missing. Configure it in Streamlit secrets, environment, or UI.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=key)


def openai_embeddings(api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key is missing. Configure it in Streamlit secrets, environment, or UI.")
    return OpenAIEmbeddings(model=model, api_key=key)
