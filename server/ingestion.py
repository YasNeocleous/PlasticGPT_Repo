"""Incremental ingestion with persistent manifest and external chunk storage.

Features:
- Deterministic chunk IDs (hash of source_id + chunk_index + text)
- Persistent manifest to track ingested chunks
- External chunk text storage (pubmed_data/chunks/)
- Embedding cache: skip re-embedding existing chunks
- Store only summaries + pointers in vector DB metadata
"""

from __future__ import annotations

import os
import csv
from typing import Iterable, Dict, Any, List

from .embedding import embed_texts
from .vector_store import Document, get_store
from .manifest import compute_chunk_id, get_manifest_manager
from .chunk_storage import save_chunk_text, generate_summary


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


def ingest(items: Iterable[Dict[str, Any]], incremental: bool = True):
	"""Ingest documents with incremental support and external chunk storage.
	
	Args:
		items: Iterable of dicts with 'title', 'text', and metadata fields
		incremental: If True, skip chunks that already exist in manifest
	
	Returns:
		Dictionary with statistics: {
			'total_chunks': int,
			'new_chunks': int,
			'skipped_chunks': int,
			'embedded_chunks': int
		}
	"""
	store = get_store()
	manifest = get_manifest_manager()
	manifest.load()
	
	# Prepare batches for embedding
	chunks_to_embed: List[str] = []
	docs_to_add: List[Document] = []
	ids_to_add: List[str] = []
	
	total_chunks = 0
	new_chunks = 0
	skipped_chunks = 0
	
	for item in items:
		title = item.get("title", "")
		text = item.get("text", "")
		source_id = item.get("pmid") or item.get("source_id") or item.get("title", "unknown")
		base_meta = {k: v for k, v in item.items() if k not in {"text"}}
		
		for chunk_index, chunk_text in enumerate(_chunk_text(text)):
			total_chunks += 1
			
			# Compute deterministic chunk_id
			chunk_id = compute_chunk_id(source_id, chunk_index, chunk_text)
			
			# Check if already exists
			if incremental and manifest.has_chunk(chunk_id):
				skipped_chunks += 1
				print(f"[ingestion] Skipping existing chunk {chunk_id[:12]}... (source={source_id}, idx={chunk_index})")
				continue
			
			new_chunks += 1
			
			# Save chunk text to disk
			chunk_path = save_chunk_text(chunk_id, chunk_text)
			
			# Generate summary for metadata
			summary = generate_summary(chunk_text, max_length=200)
			
			# Build metadata (no full text, just summary + pointer)
			meta = dict(base_meta)
			meta["title"] = title
			meta["source_id"] = source_id
			meta["chunk_index"] = chunk_index
			meta["chunk_path"] = chunk_path
			meta["summary"] = summary
			meta["text_length"] = len(chunk_text)
			
			# Create document
			doc = Document(page_content=summary, metadata=meta)  # Store summary as page_content for retrieval
			docs_to_add.append(doc)
			chunks_to_embed.append(chunk_text)  # Embed full text, not just summary
			ids_to_add.append(chunk_id)
	
	# Embed and upsert new chunks
	embedded_chunks = 0
	if chunks_to_embed:
		print(f"[ingestion] Embedding {len(chunks_to_embed)} new chunks...")
		vectors = embed_texts(chunks_to_embed)
		store.add(ids_to_add, vectors, docs_to_add)
		embedded_chunks = len(chunks_to_embed)
		
		# Update manifest
		for chunk_id, doc in zip(ids_to_add, docs_to_add):
			manifest.append_entry({
				"chunk_id": chunk_id,
				"metadata": doc.metadata,
				"embedding_stored": True,
			})
		
		print(f"[ingestion] Added {embedded_chunks} chunks to vector store and manifest.")
	
	stats = {
		"total_chunks": total_chunks,
		"new_chunks": new_chunks,
		"skipped_chunks": skipped_chunks,
		"embedded_chunks": embedded_chunks,
	}
	
	print(f"[ingestion] Stats: {stats}")
	return stats


__all__ = ["ingest", "_chunk_text"]


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
	
	print(f"[ingestion] Loading data from {csv_path}")
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
				"full_text": row.get("full_text", ""),
				"full_text_link": row.get("full_text_link", ""),
			})
	
	print(f"[ingestion] Found {len(items)} documents to process")
	stats = ingest(items, incremental=True)
	print(f"[ingestion] Complete. Final stats: {stats}")

