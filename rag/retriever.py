import os
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from utils.vectorstore import (
    load_faiss,
    load_chunks_manifest,
    check_embedding_compatibility,
)
from rag.ingest_multi import MULTI_BASE_VS_DIR


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 search."""
    return re.findall(r'\b\w+\b', text.lower())


def rewrite_query_for_followup(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    llm: Optional[Any] = None
) -> str:
    """
    Contextualize follow-up questions using recent chat history.
    If LLM is available and history is present, generates a standalone query.
    Falls back gracefully to the original question.
    """
    if not chat_history or not llm:
        return question

    # Look at the last 2 user/assistant interactions
    recent_history = chat_history[-4:]
    if not recent_history:
        return question

    try:
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            "Given the chat history and the follow-up question, rewrite the question into a "
            "standalone search query that captures all necessary context. Do NOT answer the question. "
            "Respond ONLY with the rewritten search query.\n\n"
            "Chat History:\n{history}\n\n"
            "Follow-up Question: {question}\n\n"
            "Standalone Query:"
        )
        formatted_history = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_history
        )
        chain = prompt | llm
        response = chain.invoke({"history": formatted_history, "question": question})
        rewritten = response.content if hasattr(response, "content") else str(response)
        cleaned = rewritten.strip().strip('"').strip("'")
        return cleaned if cleaned else question
    except Exception:
        return question


class HybridRetriever:
    """
    Hybrid retriever combining dense FAISS similarity search and sparse BM25 lexical search
    with Reciprocal Rank Fusion (RRF), cross-page aggregation, and strict document isolation.
    """

    def __init__(
        self,
        embeddings,
        active_model_id: str,
        base_dir: str = MULTI_BASE_VS_DIR
    ):
        self.embeddings = embeddings
        self.active_model_id = active_model_id
        self.base_dir = base_dir

    def retrieve(
        self,
        query: str,
        doc_id: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 4,
        use_hybrid: bool = True,
        cross_doc_mode: bool = False,
        available_doc_ids: Optional[List[str]] = None,
        doc_name_map: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval across target document(s).
        
        Default behavior strictly isolates retrieval to the single selected doc_id.
        Cross-document mode must be explicitly enabled.
        """
        if query_embedding is None:
            query_embedding = self.embeddings.embed_query(query)

        target_doc_ids = [doc_id]
        if cross_doc_mode and available_doc_ids:
            target_doc_ids = list(set([doc_id] + available_doc_ids))

        all_candidates = []

        for target_id in target_doc_ids:
            vs_dir = os.path.join(self.base_dir, f"{target_id}_index")
            if not os.path.exists(vs_dir):
                continue

            # Verify embedding model compatibility to prevent dimension mismatch crashes
            is_compat, msg = check_embedding_compatibility(vs_dir, self.active_model_id)
            if not is_compat:
                raise ValueError(f"Document {target_id}: {msg}")

            if not os.path.exists(os.path.join(vs_dir, "index.faiss")):
                # Document has no indexed vector store (e.g. scanned/empty text)
                continue

            try:
                vs = load_faiss(vs_dir, self.embeddings)
            except Exception as e:
                raise RuntimeError(f"Failed to load vector store for {target_id}: {str(e)}") from e


            doc_name = (doc_name_map or {}).get(target_id, target_id[:8])

            # 1. Dense Search (FAISS)
            dense_docs_with_scores = vs.similarity_search_with_score_by_vector(
                query_embedding,
                k=max(top_k * 2, 6)
            )

            # In FAISS L2 distance, lower is better. In inner product, higher is better.
            dense_ranked = []
            for rank_idx, (doc, dist) in enumerate(dense_docs_with_scores):
                dense_ranked.append({
                    "doc": doc,
                    "dense_rank": rank_idx + 1,
                    "distance": float(dist),
                    "target_id": target_id,
                    "doc_name": doc_name
                })

            # 2. Sparse Search (BM25)
            manifest_chunks = load_chunks_manifest(vs_dir)
            sparse_ranked_map = {}

            if use_hybrid and manifest_chunks:
                corpus_texts = [c.get("text", "") for c in manifest_chunks]
                tokenized_corpus = [tokenize(t) for t in corpus_texts]
                query_tokens = tokenize(query)

                if any(tokenized_corpus) and query_tokens:
                    bm25 = BM25Okapi(tokenized_corpus)
                    bm25_scores = bm25.get_scores(query_tokens)
                    
                    # Sort chunk indices by BM25 score descending
                    sorted_bm25_indices = sorted(
                        range(len(bm25_scores)),
                        key=lambda i: bm25_scores[i],
                        reverse=True
                    )

                    for rank_idx, chunk_idx in enumerate(sorted_bm25_indices[:top_k * 3]):
                        chunk = manifest_chunks[chunk_idx]
                        sparse_ranked_map[chunk.get("chunk_id")] = {
                            "sparse_rank": rank_idx + 1,
                            "bm25_score": float(bm25_scores[chunk_idx]),
                            "chunk": chunk
                        }

            # 3. Reciprocal Rank Fusion (RRF)
            chunk_pool = {}

            # Add dense candidates
            for item in dense_ranked:
                doc = item["doc"]
                chunk_id = doc.metadata.get("chunk_id", f"{target_id}_p{doc.metadata.get('page', 1)}_dense_{item['dense_rank']}")
                dense_r = item["dense_rank"]
                sparse_info = sparse_ranked_map.get(chunk_id, {})
                sparse_r = sparse_info.get("sparse_rank", 100)

                bm25_val = float(sparse_info.get("bm25_score", 0.0))
                rrf_score = (1.0 / (60.0 + dense_r)) + (1.0 / (60.0 + sparse_r)) + (0.002 * bm25_val)

                chunk_pool[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": target_id,
                    "doc_name": item["doc_name"],
                    "page": doc.metadata.get("page", 1),
                    "page_number": doc.metadata.get("page", 1),
                    "text": doc.page_content,
                    "source_text": doc.page_content,
                    "bounding_boxes": doc.metadata.get("bounding_boxes", []),
                    "text_span": doc.metadata.get("text_span", [0, len(doc.page_content)]),
                    "score": round(rrf_score, 6),
                    "dense_distance": round(item["distance"], 4),
                    "bm25_score": round(bm25_val, 4)
                }

            # Add top BM25 candidates that weren't in dense search
            if use_hybrid:
                for chunk_id, info in list(sparse_ranked_map.items())[:top_k]:
                    if chunk_id not in chunk_pool:
                        chunk = info["chunk"]
                        sparse_r = info["sparse_rank"]
                        bm25_val = float(info.get("bm25_score", 0.0))
                        rrf_score = (1.0 / (60.0 + 100)) + (1.0 / (60.0 + sparse_r)) + (0.002 * bm25_val)
                        chunk_pool[chunk_id] = {
                            "chunk_id": chunk_id,
                            "doc_id": target_id,
                            "doc_name": doc_name,
                            "page": chunk.get("page", 1),
                            "page_number": chunk.get("page", 1),
                            "text": chunk.get("text", ""),
                            "source_text": chunk.get("text", ""),
                            "bounding_boxes": chunk.get("bounding_boxes", []),
                            "text_span": chunk.get("text_span", [0, len(chunk.get("text", ""))]),
                            "score": round(rrf_score, 6),
                            "dense_distance": 999.0,
                            "bm25_score": round(bm25_val, 4)
                        }

            all_candidates.extend(chunk_pool.values())

        # Sort all aggregated candidates by fused score descending
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return all_candidates[:top_k]
