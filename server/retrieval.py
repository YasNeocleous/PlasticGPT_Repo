"""Advanced retrieval pipeline with hybrid search and re-ranking.

Features:
- Semantic search (vector similarity) for top K candidates
- Optional BM25 lexical search for keyword matching
- Re-ranking with score fusion or LLM-based scoring
- Lazy chunk text loading (only fetch full text for top N results)
- Maximal Marginal Relevance (MMR) for diversity
"""

from __future__ import annotations

import math
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from .embedding import embed_texts
from .vector_store import get_store, Document
from .chunk_storage import load_chunk_text_by_path


def _bm25_score(query_terms: List[str], doc_terms: List[str], 
                avgdl: float = 100.0, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute BM25 score for a document given query terms.
    
    Args:
        query_terms: List of query terms (words)
        doc_terms: List of document terms (words)
        avgdl: Average document length in corpus
        k1: BM25 parameter controlling term frequency saturation
        b: BM25 parameter controlling length normalization
    
    Returns:
        BM25 score (higher is better)
    """
    doc_len = len(doc_terms)
    doc_tf = Counter(doc_terms)
    score = 0.0
    
    for term in query_terms:
        if term in doc_tf:
            tf = doc_tf[term]
            # Simplified BM25 (no IDF component since we don't have corpus stats)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += numerator / denominator
    
    return score


def _normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range using min-max scaling."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def _mmr_select(candidates: List[Tuple[float, Document]], 
                query_embedding: List[float],
                k: int,
                lambda_param: float = 0.5) -> List[Document]:
    """Select diverse results using Maximal Marginal Relevance.
    
    Args:
        candidates: List of (score, Document) tuples
        query_embedding: Query embedding vector
        k: Number of results to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of selected Documents
    """
    if not candidates:
        return []
    
    selected: List[Document] = []
    remaining = list(candidates)
    
    # First, select the highest-scoring document
    remaining.sort(key=lambda x: x[0], reverse=True)
    selected.append(remaining[0][1])
    remaining.pop(0)
    
    # Iteratively select documents with high relevance and low similarity to selected
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = 0
        
        for idx, (rel_score, doc) in enumerate(remaining):
            # Compute max similarity to already selected documents
            # (Simplified: we don't have embeddings for candidates readily available,
            #  so we'll use a heuristic based on text overlap)
            max_sim = 0.0
            for sel_doc in selected:
                # Simple Jaccard similarity on words
                doc_words = set(doc.page_content.lower().split())
                sel_words = set(sel_doc.page_content.lower().split())
                if doc_words or sel_words:
                    sim = len(doc_words & sel_words) / len(doc_words | sel_words)
                    max_sim = max(max_sim, sim)
            
            # MMR score: lambda * relevance - (1-lambda) * max_similarity
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(remaining[best_idx][1])
        remaining.pop(best_idx)
    
    return selected


class RetrievalPipeline:
    """Advanced retrieval pipeline with hybrid search and re-ranking."""
    
    def __init__(self, 
                 use_bm25: bool = False,
                 use_rerank: bool = False,
                 use_mmr: bool = False,
                 top_k_candidates: int = 20,
                 final_top_k: int = 4):
        """Initialize retrieval pipeline.
        
        Args:
            use_bm25: Enable BM25 lexical search
            use_rerank: Enable re-ranking (simple score fusion for now)
            use_mmr: Enable Maximal Marginal Relevance for diversity
            top_k_candidates: Number of candidates to retrieve initially
            final_top_k: Number of final results to return
        """
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.use_mmr = use_mmr
        self.top_k_candidates = top_k_candidates
        self.final_top_k = final_top_k
        self.store = get_store()
    
    def retrieve(self, query: str, 
                 load_full_text: bool = True) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            load_full_text: If True, load full chunk text for top results
        
        Returns:
            List of result dictionaries with 'text', 'metadata', 'score'
        """
        # 1. Semantic search
        query_embedding = embed_texts([query])[0]
        semantic_results = self.store.similarity_search(
            query_embedding, 
            k=self.top_k_candidates
        )
        
        # Initial candidates with scores (score = 1.0 for now, real scores not exposed)
        candidates: List[Tuple[float, Document]] = [
            (1.0 / (i + 1), doc) for i, doc in enumerate(semantic_results)
        ]
        
        # 2. Optional BM25 lexical search
        if self.use_bm25:
            query_terms = query.lower().split()
            bm25_scores = []
            for score, doc in candidates:
                doc_terms = doc.page_content.lower().split()
                bm25 = _bm25_score(query_terms, doc_terms)
                bm25_scores.append(bm25)
            
            # Normalize and combine scores
            if bm25_scores:
                norm_bm25 = _normalize_scores(bm25_scores)
                norm_semantic = _normalize_scores([s for s, _ in candidates])
                # Weighted average: 0.7 semantic + 0.3 BM25
                combined_scores = [
                    0.7 * sem + 0.3 * bm25 
                    for sem, bm25 in zip(norm_semantic, norm_bm25)
                ]
                candidates = [
                    (score, doc) for score, (_, doc) in zip(combined_scores, candidates)
                ]
                candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Optional re-ranking (placeholder for now)
        if self.use_rerank:
            # In production, use a cross-encoder model or LLM to re-score
            # For now, we'll just keep the combined scores
            pass
        
        # 4. Optional MMR for diversity
        if self.use_mmr:
            selected_docs = _mmr_select(
                candidates, 
                query_embedding, 
                k=self.final_top_k,
                lambda_param=0.7  # Favor relevance over diversity
            )
        else:
            # Just take top K by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected_docs = [doc for _, doc in candidates[:self.final_top_k]]
        
        # 5. Build results and optionally load full text
        results = []
        for doc in selected_docs:
            result = {
                "summary": doc.page_content,  # This is the summary we stored
                "metadata": doc.metadata,
                "score": 1.0,  # Placeholder
            }
            
            if load_full_text and "chunk_path" in doc.metadata:
                chunk_path = doc.metadata["chunk_path"]
                full_text = load_chunk_text_by_path(chunk_path)
                if full_text:
                    result["text"] = full_text
                else:
                    result["text"] = doc.page_content  # Fallback to summary
            else:
                result["text"] = doc.page_content
            
            results.append(result)
        
        return results


def get_retrieval_pipeline(
    use_bm25: Optional[bool] = None,
    use_rerank: Optional[bool] = None,
    use_mmr: Optional[bool] = None,
) -> RetrievalPipeline:
    """Get a configured retrieval pipeline.
    
    Args:
        use_bm25: Override default BM25 setting (from env var RETRIEVAL_USE_BM25)
        use_rerank: Override default rerank setting (from env var RETRIEVAL_USE_RERANK)
        use_mmr: Override default MMR setting (from env var RETRIEVAL_USE_MMR)
    
    Returns:
        Configured RetrievalPipeline instance
    """
    if use_bm25 is None:
        use_bm25 = os.getenv("RETRIEVAL_USE_BM25", "0") in {"1", "true", "True"}
    if use_rerank is None:
        use_rerank = os.getenv("RETRIEVAL_USE_RERANK", "0") in {"1", "true", "True"}
    if use_mmr is None:
        use_mmr = os.getenv("RETRIEVAL_USE_MMR", "0") in {"1", "true", "True"}
    
    top_k_candidates = int(os.getenv("RETRIEVAL_TOP_K_CANDIDATES", "20"))
    final_top_k = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "4"))
    
    return RetrievalPipeline(
        use_bm25=use_bm25,
        use_rerank=use_rerank,
        use_mmr=use_mmr,
        top_k_candidates=top_k_candidates,
        final_top_k=final_top_k,
    )


__all__ = [
    "RetrievalPipeline",
    "get_retrieval_pipeline",
]
