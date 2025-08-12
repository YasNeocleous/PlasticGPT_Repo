"""Simple in-memory vector store (placeholder for Pinecone / other DB).

For initial development we keep an in-memory list of (id, vector, metadata,
content). This can later be replaced with Pinecone, Chroma, etc. The interface
is intentionally tiny: add() and similarity_search().
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Sequence, Tuple


class Document:
	def __init__(self, page_content: str, metadata: Dict[str, Any] | None = None):
		self.page_content = page_content
		self.metadata = metadata or {}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
	num = sum(x * y for x, y in zip(a, b))
	da = math.sqrt(sum(x * x for x in a))
	db = math.sqrt(sum(y * y for y in b))
	if da == 0 or db == 0:
		return 0.0
	return num / (da * db)


class InMemoryVectorStore:
	def __init__(self):
		self._data: List[Tuple[str, List[float], Document]] = []

	def add(self, ids: List[str], vectors: List[List[float]], docs: List[Document]):
		for i, v, d in zip(ids, vectors, docs):
			self._data.append((i, v, d))

	def similarity_search(self, query_vector: List[float], k: int = 4) -> List[Document]:
		scored = [(_cosine(query_vector, vec), doc) for _id, vec, doc in self._data]
		scored.sort(key=lambda x: x[0], reverse=True)
		return [d for _s, d in scored[:k]]


_shared_store: InMemoryVectorStore | None = None


def get_store() -> InMemoryVectorStore:
	global _shared_store
	if _shared_store is None:
		_shared_store = InMemoryVectorStore()
	return _shared_store


__all__ = ["Document", "InMemoryVectorStore", "get_store"]

