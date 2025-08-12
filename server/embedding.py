"""Embedding utilities wrapping sentence-transformers.

Provides a small helper to create embeddings for text chunks. If the
sentence-transformers library (or model download) is unavailable we fall back
to a deterministic hashing stub so that unit tests can still run quickly.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

try:  # pragma: no cover - optional heavy dependency
	from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
	SentenceTransformer = None  # type: ignore

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
	if SentenceTransformer is None:
		return None
	try:
		return SentenceTransformer(MODEL_NAME)
	except Exception:
		return None


def embed_texts(texts: List[str]) -> List[List[float]]:
	model = _get_model()
	if model is None:
		# Deterministic stub embedding: sha256 -> bytes -> floats 0-1
		vectors: List[List[float]] = []
		for t in texts:
			h = hashlib.sha256(t.encode("utf-8")).digest()[:32]
			vec = [b / 255.0 for b in h]
			vectors.append(vec)
		return vectors
	return model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore


__all__ = ["embed_texts", "MODEL_NAME"]

