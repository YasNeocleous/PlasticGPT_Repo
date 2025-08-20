"""Simple in-memory vector store (placeholder for Pinecone / other DB).

For initial development we keep an in-memory list of (id, vector, metadata,
content). This can later be replaced with Pinecone, Chroma, etc. The interface
is intentionally tiny: add() and similarity_search().
"""

from __future__ import annotations

import math
import os
from typing import List, Dict, Any, Sequence, Tuple
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


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


# Optional Pinecone backend
try:  # pragma: no cover - optional dependency
	from pinecone import Pinecone, ServerlessSpec  # type: ignore
except Exception:  # pragma: no cover
	Pinecone = None  # type: ignore
	ServerlessSpec = None  # type: ignore


class PineconeVectorStore:
	def __init__(self, index_name: str = "plasticgpt", dimension: int = 1536):
		if Pinecone is None:
			raise RuntimeError("Pinecone SDK not installed. pip install pinecone-client")
		api_key = os.getenv("PINECONE_API_KEY", "")
		if not api_key:
			raise RuntimeError("PINECONE_API_KEY not set")
		self._pc = Pinecone(api_key=api_key)
		self._index_name = index_name
		self._dimension = dimension
		names = [i["name"] if isinstance(i, dict) else getattr(i, "name", None) for i in self._pc.list_indexes()]  # type: ignore
		if index_name not in names:
			# Default to AWS us-east-1 serverless
			spec = None
			try:
				spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1"))
			except Exception:
				spec = None
			self._pc.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)
		self._index = self._pc.Index(index_name)

	def add(self, ids: List[str], vectors: List[List[float]], docs: List[Document]):
		# Only include small fields in metadata; do not store full text in metadata
		upserts = []
		for i, v, d in zip(ids, vectors, docs):
			meta = {k: v for k, v in d.metadata.items() if k in ["pmid", "title", "authors", "date", "full_text_link"]}
			upserts.append({"id": i, "values": v, "metadata": meta})
		# Upsert in small batches to avoid Pinecone payload limits
		batch_size = 100
		for j in range(0, len(upserts), batch_size):
			self._index.upsert(vectors=upserts[j:j+batch_size])

	def similarity_search(self, query_vector: List[float], k: int = 4) -> List[Document]:
		res = self._index.query(vector=query_vector, top_k=k, include_metadata=True)
		docs: List[Document] = []
		matches = getattr(res, "matches", []) or getattr(res, "data", [])  # be lenient across SDK versions
		for m in matches:
			md = getattr(m, "metadata", None) or m.get("metadata", {})  # type: ignore
			text = (md or {}).get("text", "")
			docs.append(Document(page_content=text, metadata={k: v for k, v in (md or {}).items() if k != "text"}))
		return docs


_shared_store: Any | None = None


def get_store() -> InMemoryVectorStore:
	global _shared_store
	if _shared_store is not None:
		return _shared_store  # type: ignore
	backend = os.getenv("VECTOR_BACKEND", "memory").lower()
	print(f"[vector_store] VECTOR_BACKEND={backend}")
	if backend == "pinecone":
		if Pinecone is None:
			print("[vector_store] Pinecone SDK not installed.")
		else:
			try:
				index_name = os.getenv("PINECONE_INDEX", "plasticgpt")
				dim = int(os.getenv("PINECONE_DIM", "1536"))
				print(f"[vector_store] Attempting PineconeVectorStore index={index_name} dim={dim}")
				_shared_store = PineconeVectorStore(index_name=index_name, dimension=dim)
				print("[vector_store] PineconeVectorStore initialized.")
				return _shared_store  # type: ignore
			except Exception as e:
				print(f"[vector_store] Pinecone init failed: {e}")
	print("[vector_store] Using InMemoryVectorStore.")
	_shared_store = InMemoryVectorStore()
	return _shared_store


__all__ = ["Document", "InMemoryVectorStore", "get_store"]

