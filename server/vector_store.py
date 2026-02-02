"""Vector store with caching support.

Embeddings are computed once and cached to a JSON file.
On startup, the cache is loaded instead of re-computing embeddings.
Optionally syncs to Google Cloud Storage for production.
"""

from __future__ import annotations

import os
import json
import math
import logging
from typing import List, Dict, Any, Sequence, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass


class Document:
    """A document with content and metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(page_content=data["page_content"], metadata=data.get("metadata", {}))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class CachedVectorStore:
    """Vector store with local file caching and optional Cloud Storage sync."""
    
    def __init__(self, cache_path: str | None = None):
        self._data: List[Tuple[str, List[float], Document]] = []
        self._cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), "vector_cache.json"
        )
        self._gcs_bucket = os.getenv("GCS_CACHE_BUCKET", "")
        self._gcs_path = os.getenv("GCS_CACHE_PATH", "vector_cache.json")

    def add(self, ids: List[str], vectors: List[List[float]], docs: List[Document]):
        """Add documents with their embeddings."""
        for doc_id, vector, doc in zip(ids, vectors, docs):
            self._data.append((doc_id, vector, doc))
        print(f"[vector_store] Added {len(ids)} documents. Total: {len(self._data)}")

    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[Document]:
        """Find the k most similar documents to the query vector."""
        if not self._data:
            return []
        
        scored = [
            (_cosine_similarity(query_vector, vec), doc) 
            for _id, vec, doc in self._data
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _score, doc in scored[:k]]
    
    def save_cache(self) -> bool:
        """Save embeddings to local cache file (and optionally GCS)."""
        if not self._data:
            return False
        
        cache_data = {
            "version": 1,
            "count": len(self._data),
            "items": [
                {"id": doc_id, "vector": vector, "document": doc.to_dict()}
                for doc_id, vector, doc in self._data
            ]
        }
        
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
            print(f"[vector_store] Saved cache: {len(self._data)} items")
            
            if self._gcs_bucket:
                self._upload_to_gcs(cache_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def load_cache(self) -> bool:
        """Load embeddings from cache (tries GCS first, then local)."""
        # Try GCS first
        if self._gcs_bucket and self._download_from_gcs():
            return True
        
        # Fall back to local
        if not os.path.exists(self._cache_path):
            return False
        
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            self._data = []
            for item in cache_data.get("items", []):
                doc = Document.from_dict(item["document"])
                self._data.append((item["id"], item["vector"], doc))
            
            print(f"[vector_store] Loaded {len(self._data)} items from local cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False
    
    def _upload_to_gcs(self, cache_data: Dict[str, Any]):
        """Upload cache to Google Cloud Storage."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self._gcs_bucket)
            blob = bucket.blob(self._gcs_path)
            blob.upload_from_string(json.dumps(cache_data), content_type="application/json")
            print(f"[vector_store] Uploaded to gs://{self._gcs_bucket}/{self._gcs_path}")
        except Exception as e:
            logger.debug(f"GCS upload skipped: {e}")
    
    def _download_from_gcs(self) -> bool:
        """Download cache from Google Cloud Storage."""
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self._gcs_bucket)
            blob = bucket.blob(self._gcs_path)
            
            if not blob.exists():
                return False
            
            cache_data = json.loads(blob.download_as_string())
            self._data = []
            for item in cache_data.get("items", []):
                doc = Document.from_dict(item["document"])
                self._data.append((item["id"], item["vector"], doc))
            
            print(f"[vector_store] Loaded {len(self._data)} items from GCS cache")
            return True
        except Exception as e:
            logger.debug(f"GCS download skipped: {e}")
            return False
    
    def clear(self):
        """Clear all stored documents."""
        self._data = []


# Singleton instance
_store: CachedVectorStore | None = None


def get_store() -> CachedVectorStore:
    """Get the shared vector store instance."""
    global _store
    if _store is None:
        _store = CachedVectorStore()
        if _store.load_cache():
            print("[vector_store] Using cached embeddings (fast startup!)")
        else:
            print("[vector_store] No cache found, will compute embeddings")
    return _store


__all__ = ["Document", "CachedVectorStore", "get_store"]

