"""Embedding utilities with OpenAI-first and fallbacks.

Order of backends:
- OpenAI text-embedding-3-small (if OPENAI_API_KEY available)
- sentence-transformers (all-MiniLM-L6-v2)
- Deterministic hashing stub
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List
import os

# Load .env for local dev
try:  # pragma: no cover
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

try:  # pragma: no cover - optional heavy dependency
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
	SentenceTransformer = None  # type: ignore

try:  # OpenAI SDK new client
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
	OpenAI = None  # type: ignore
	
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def _get_model():
	if SentenceTransformer is None:
		return None
	try:
		return SentenceTransformer(MODEL_NAME)
	except Exception:
		return None


def embed_texts(texts: List[str]) -> List[List[float]]:
	# Always use OpenAI embeddings if VECTOR_BACKEND is pinecone
	vector_backend = os.getenv("VECTOR_BACKEND", "memory").lower()
	api_key = os.getenv("OPENAI_API_KEY", "")
	print(f"[embedding] VECTOR_BACKEND={vector_backend} OPENAI_API_KEY={'set' if api_key else 'unset'} OpenAI={'yes' if OpenAI else 'no'}")
	if OpenAI is not None and api_key and not api_key.lower().startswith("test"):
		def batch_openai_embeddings(texts, batch_size=100):
			client = OpenAI()
			all_embeddings = []
			for i in range(0, len(texts), batch_size):
				batch = texts[i:i+batch_size]
				# Check total tokens in batch, if too large, reduce batch size
				# For simplicity, assume max 3000 tokens per text, adjust as needed
				try:
					resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
					all_embeddings.extend([d.embedding for d in resp.data])
				except Exception as e:
					print(f"[embedding] OpenAI embedding error in batch: {e}")
					raise
			return all_embeddings

		if vector_backend == "pinecone":
			try:
				print("[embedding] Using OpenAI embeddings for Pinecone backend (with batching).")
				return batch_openai_embeddings(texts)
			except Exception as e:
				print(f"[embedding] OpenAI embedding error (pinecone): {e}")
		else:
			try:
				print("[embedding] Using OpenAI embeddings (with batching).")
				return batch_openai_embeddings(texts)
			except Exception as e:
				print(f"[embedding] OpenAI embedding error: {e}")

	# 2) sentence-transformers (only if not using Pinecone)
	if vector_backend != "pinecone":
		model = _get_model()
		if model is not None:
			try:
				print("[embedding] Using sentence-transformers.")
				return model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore
			except Exception as e:
				print(f"[embedding] sentence-transformers error: {e}")

	# 3) Deterministic hashing stub: sha256 -> bytes -> floats 0-1
	print("[embedding] Using deterministic hashing stub.")
	vectors: List[List[float]] = []
	for t in texts:
		h = hashlib.sha256(t.encode("utf-8")).digest()[:512]
		vec = [b / 255.0 for b in h]
		vectors.append(vec)
	return vectors


__all__ = ["embed_texts", "MODEL_NAME"]

