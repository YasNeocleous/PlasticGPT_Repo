"""Basic ingestion: simple text chunking & storing embeddings.

Replace / extend with PDF / HTML parsing later. For now we accept a list of
(title, text, metadata) items and chunk them into overlapping windows.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterable, Dict, Any, List

from embedding import embed_texts
from vector_store import Document, get_store


def _chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
	# naive token approximation using words
	words = text.split()
	chunks = []
	step = max_tokens - overlap
	for start in range(0, len(words), step):
		window = words[start : start + max_tokens]
		if not window:
			break
		chunks.append(" ".join(window))
		if start + max_tokens >= len(words):
			break
	return chunks


def ingest(items: Iterable[Dict[str, Any]]):
	store = get_store()
	all_texts: List[str] = []
	docs: List[Document] = []
	ids: List[str] = []
	for item in items:
		title = item.get("title", "")
		text = item.get("text", "")
		base_meta = {k: v for k, v in item.items() if k not in {"text"}}
		for chunk in _chunk_text(text):
			meta = dict(base_meta)
			meta["title"] = title
			doc = Document(page_content=chunk, metadata=meta)
			docs.append(doc)
			all_texts.append(chunk)
			ids.append(str(uuid.uuid4()))
	if not docs:
		return 0
	vectors = embed_texts(all_texts)
	store.add(ids, vectors, docs)
	return len(docs)


__all__ = ["ingest"]

