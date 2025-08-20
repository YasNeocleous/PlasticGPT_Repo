"""Basic ingestion: simple text chunking & storing embeddings.

Replace / extend with PDF / HTML parsing later. For now we accept a list of
(title, text, metadata) items and chunk them into overlapping windows.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterable, Dict, Any, List

from .embedding import embed_texts
from .vector_store import Document, get_store
import os
import csv


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


if __name__ == "__main__":  # simple CLI
	# Allow running: python -m server.ingestion <optional_path_to_csv>
	csv_path = None
	import sys
	if len(sys.argv) > 1:
		csv_path = sys.argv[1]
	if not csv_path:
		csv_path = os.path.join(os.path.dirname(__file__), "vector_db", "pubmed_plastic_surgery.csv")
	if not os.path.exists(csv_path):
		print(f"CSV not found: {csv_path}")
		sys.exit(1)
	items = []
	with open(csv_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			text = row.get("abstract") or row.get("full_text") or ""
			items.append({
				"title": row.get("title", ""),
				"text": text,
				"pmid": row.get("pmid", ""),
				"authors": row.get("authors", ""),
				"date": row.get("date", ""),
				"full_text_link": row.get("full_text_link", ""),
			})
	n = ingest(items)
	print(f"Ingested chunks: {n}")

