from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from utils.vectorstore import load_faiss
from rag.ingest_single import SINGLE_VS_PATH


class QueryState(TypedDict):
    question: str
    query_embedding: list
    context: str
    pages: List[int]
    answer: str


def build_single_query_graph(embeddings, llm):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use ONLY the provided context to answer the question."),
            ("human", "{question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

    def embed_node(state: QueryState):
        q = state["question"]
        q_emb = embeddings.embed_query(q)
        return {"query_embedding": q_emb}

    def retrieve_node(state: QueryState):
        vs = load_faiss(SINGLE_VS_PATH, embeddings)
        docs = vs.similarity_search_by_vector(state["query_embedding"], k=3)

        context = "\n\n".join(d.page_content for d in docs)
        pages = sorted({d.metadata.get("page", "?") for d in docs})

        return {"context": context, "pages": pages}

    def answer_node(state: QueryState):
        chain = prompt | llm
        res = chain.invoke(
            {"question": state["question"], "context": state["context"]}
        )
        return {"answer": res}

    graph = StateGraph(QueryState)

    graph.add_node("embed", embed_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
