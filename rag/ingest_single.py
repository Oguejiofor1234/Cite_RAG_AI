from typing import Dict, Any
from langgraph.graph import StateGraph, END 

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_pages
from utils.vectorstore import build_faiss_from_texts, save_faiss

SINGLE_VS_PATH = "data/single/vectorstore/index"  # 


def build_single_ingest_graph(embeddings): # this function builds and returns a compiled Langgraph for ingesting a single PDF document
    
    def load_node(state: Dict[str, Any]): # this node loads the PDF document from the specified path and extracts its pages as text
        path = state["path"]              # the path to the PDF document is expected to be provided in the state when executing this node
        pages = load_pdf(path)             # the load_pdf function from utils/pdf_loader.py is called to read the PDF and extract its pages as text, returning a list of dictionaries with page numbers and text content
        return {"pages": pages}           # the extracted pages are returned in the state for the next node to process
    
    def chunk_node(state: Dict[str, Any]): # this node takes the extracted pages from the previous node and splits them into smaller chunks of text for better processing and embedding
        pages = state["pages"]            # the list of pages extracted from the PDF is retrieved from the state
        chunks = split_pages(pages)    # the split_pages function from utils/text_splitter.py is called to split the text of each page into smaller chunks, returning a list of dictionaries with page numbers and chunked text content
        return {"chunks": chunks}      # the resulting chunks are returned in the state for the next node to process
    
    def embed_store_node(state: Dict[str, Any]): # this node takes the chunked text from the previous node and creates embeddings for each chunk, then stores them in a vector store
        texts= [c["text"] for c in state["chunks"]]  # the text content of each chunk is extracted into a list of strings to be embedded
        metadatas = [{"page": c["page"]} for c in state["chunks"]] # the metadata for each chunk is created as a list of dictionaries containing the page number, which can be useful for retrieval later
        vs = build_faiss_from_texts(texts, embeddings, metadatas)
        save_faiss(vs, SINGLE_VS_PATH)
        return {"status": "success"}
    
    graph = StateGraph(dict)    #state graph is created with a state type of dictionary to hold the data as it flows through the nodes
    graph.add_node("load", load_node)  # the load_node is added to the graph with the name "load", which will be the entry point for executing the graph
    graph.add_node("chunk", chunk_node)   
    graph.add_node("embed_store", embed_store_node)


    graph.set_entry_point("load")   # the entry point of the graph is set to the "load" node, meaning that when the graph is executed, it will start with the load_node to process the input PDF document
    graph.add_edge("load", "chunk")  #
    graph.add_edge("chunk", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()    
