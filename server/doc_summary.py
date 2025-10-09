"""Document-level summary index for hierarchical retrieval.

This module provides utilities to:
1. Generate document-level summaries (abstractive or extractive)
2. Store doc summaries with embeddings in a separate index
3. Perform two-stage retrieval:
   - First stage: search doc summaries to find candidate documents
   - Second stage: search chunk-level embeddings within selected documents

This reduces costs and improves precision for large corpora.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .embedding import embed_texts
from .vector_store import Document


DEFAULT_DOC_SUMMARY_PATH = Path(__file__).parent / "pubmed_data" / "doc_summaries.jsonl"


def generate_doc_summary(full_text: str, 
                         method: str = "extractive",
                         max_length: int = 500) -> str:
    """Generate a document-level summary.
    
    Args:
        full_text: Full document text
        method: 'extractive' (select key sentences) or 'truncate'
        max_length: Maximum summary length in characters
    
    Returns:
        Document summary string
    """
    if method == "truncate":
        # Simple truncation
        if len(full_text) <= max_length:
            return full_text
        return full_text[:max_length].rsplit(" ", 1)[0] + "..."
    
    elif method == "extractive":
        # Extract first few sentences
        sentences = full_text.split(". ")
        summary = ""
        for sent in sentences:
            if len(summary) + len(sent) + 2 > max_length:
                break
            summary += sent + ". "
        
        if not summary and sentences:
            # If first sentence is too long, truncate it
            summary = sentences[0][:max_length] + "..."
        
        return summary.strip()
    
    else:
        raise ValueError(f"Unknown summary method: {method}")


def compute_doc_id(source_id: str, title: str) -> str:
    """Compute deterministic document ID.
    
    Args:
        source_id: Source identifier (e.g., PMID)
        title: Document title
    
    Returns:
        SHA256 hex digest as doc_id
    """
    h = hashlib.sha256()
    h.update(source_id.encode("utf-8"))
    h.update(b"\0")
    h.update(title.encode("utf-8"))
    return h.hexdigest()


class DocSummaryIndex:
    """Manages document-level summaries and their embeddings."""
    
    def __init__(self, summary_path: Path | str | None = None):
        self.summary_path = Path(summary_path) if summary_path else DEFAULT_DOC_SUMMARY_PATH
        self._summaries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Dict[str, Any]]:
        """Load summaries from disk.
        
        Returns:
            Dictionary mapping doc_id -> summary entry
        """
        if self._loaded:
            return self._summaries
        
        self._summaries = {}
        if not self.summary_path.exists():
            print(f"[doc_summary] No existing summaries at {self.summary_path}")
            self._loaded = True
            return self._summaries
        
        with self.summary_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    doc_id = obj.get("doc_id")
                    if doc_id:
                        self._summaries[doc_id] = obj
                except json.JSONDecodeError as e:
                    print(f"[doc_summary] Error parsing line {line_num}: {e}")
        
        print(f"[doc_summary] Loaded {len(self._summaries)} doc summaries")
        self._loaded = True
        return self._summaries
    
    def has_doc(self, doc_id: str) -> bool:
        """Check if a document summary exists."""
        if not self._loaded:
            self.load()
        return doc_id in self._summaries
    
    def get_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get summary entry for a document."""
        if not self._loaded:
            self.load()
        return self._summaries.get(doc_id)
    
    def add_summary(self, 
                    source_id: str,
                    title: str,
                    summary: str,
                    metadata: Dict[str, Any] | None = None) -> str:
        """Add a document summary to the index.
        
        Args:
            source_id: Source identifier (e.g., PMID)
            title: Document title
            summary: Document summary text
            metadata: Additional metadata
        
        Returns:
            The doc_id
        """
        doc_id = compute_doc_id(source_id, title)
        
        if self.has_doc(doc_id):
            print(f"[doc_summary] Doc {doc_id[:12]}... already exists, skipping")
            return doc_id
        
        # Generate embedding for summary
        embedding = embed_texts([summary])[0]
        
        entry = {
            "doc_id": doc_id,
            "source_id": source_id,
            "title": title,
            "summary": summary,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        
        # Ensure directory exists
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file
        with self.summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Update in-memory cache
        if not self._loaded:
            self.load()
        self._summaries[doc_id] = entry
        
        return doc_id
    
    def search_summaries(self, 
                        query: str,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """Search document summaries by semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of top documents to return
        
        Returns:
            List of summary entries sorted by relevance
        """
        if not self._loaded:
            self.load()
        
        if not self._summaries:
            return []
        
        # Embed query
        query_embedding = embed_texts([query])[0]
        
        # Compute cosine similarity with all doc summaries
        scored_docs = []
        for doc_id, entry in self._summaries.items():
            doc_embedding = entry.get("embedding", [])
            if not doc_embedding:
                continue
            
            # Cosine similarity
            similarity = self._cosine(query_embedding, doc_embedding)
            scored_docs.append((similarity, entry))
        
        # Sort by score and return top K
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_docs[:top_k]]
    
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)


def get_doc_summary_index(summary_path: Path | str | None = None) -> DocSummaryIndex:
    """Get or create a DocSummaryIndex instance."""
    return DocSummaryIndex(summary_path)


__all__ = [
    "generate_doc_summary",
    "compute_doc_id",
    "DocSummaryIndex",
    "get_doc_summary_index",
]
