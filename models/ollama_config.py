from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def ollama_llm():
    return Ollama(model="llama3.1:8b", temperature=0)

def ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")   
