"""Vectorize a CSV that already contains a full_text column.

Example (PowerShell):
  $env:VECTOR_BACKEND="pinecone"; \
  $env:PINECONE_API_KEY="YOUR_KEY"; \
  $env:OPENAI_API_KEY="YOUR_OPENAI"; \
  python -m server.vectorize_csv `
    --csv server/pubmed_data/pubmed_plastic_surgery_PMCID_with_fulltext_ingested.csv `
    --id-col pmcid --text-col full_text --chunk-size 1500 --overlap 200 `
    --batch-size 64 --persist server/pubmed_data/vector_manifest.jsonl

If you only want to test locally without Pinecone:
  python -m server.vectorize_csv --csv server/pubmed_data/pubmed_plastic_surgery_PMCID_with_fulltext_ingested.csv --limit 5

Environment variables:
  VECTOR_BACKEND=memory|pinecone (default memory)
  PINECONE_API_KEY=... (required if VECTOR_BACKEND=pinecone)
  PINECONE_INDEX=plasticgpt (optional override)
  PINECONE_DIM=1536 (dimension, must match embedding size for OpenAI model)
  OPENAI_API_KEY=... (to use OpenAI embeddings; else falls back)
  PINECONE_STORE_TEXT=1 (store raw chunk text inside Pinecone metadata; increases storage!)

Manifest JSONL fields (one per chunk):
  chunk_id, source_id, row_index, chunk_index, text_length, chunk_size, overlap, n_chunks_total_est
"""
from __future__ import annotations
import argparse, os, json, math
from typing import List
import pandas as pd

from .embedding import embed_texts
from .vector_store import get_store, Document

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap if overlap > 0 else end
        if start < 0:
            start = 0
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Vectorize CSV with full_text column")
    ap.add_argument('--csv', required=True, help='Path to CSV file with full_text')
    ap.add_argument('--id-col', default='pmcid', help='Column to use as base id (pmcid/pmid)')
    ap.add_argument('--text-col', default='full_text', help='Column containing full text')
    ap.add_argument('--chunk-size', type=int, default=1500)
    ap.add_argument('--overlap', type=int, default=200)
    ap.add_argument('--min-chars', type=int, default=200)
    ap.add_argument('--limit', type=int, default=0, help='Limit number of source rows (for testing)')
    ap.add_argument('--persist', help='Optional JSONL manifest output path')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--dry-run', action='store_true', help='Parse/chunk only; no embeddings or store writes')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"Text column '{args.text_col}' not in CSV")
    if args.id_col not in df.columns:
        raise SystemExit(f"ID column '{args.id_col}' not in CSV")

    store = get_store()
    rows = df.to_dict('records')
    if args.limit > 0:
        rows = rows[:args.limit]

    manifest_fp = open(args.persist, 'w', encoding='utf-8') if args.persist else None
    prepared_texts: List[str] = []
    prepared_docs: List[Document] = []
    prepared_ids: List[str] = []
    total_chunks = 0

    def flush_batch():
        nonlocal prepared_texts, prepared_docs, prepared_ids, total_chunks
        if not prepared_texts:
            return
        print(f"[vectorize_csv] Embedding batch size={len(prepared_texts)} ...")
        vectors = embed_texts(prepared_texts) if not args.dry_run else [[0.0] * 16 for _ in prepared_texts]
        if not args.dry_run:
            store.add(prepared_ids, vectors, prepared_docs)
        total_chunks += len(prepared_texts)
        if manifest_fp:
            for cid, doc in zip(prepared_ids, prepared_docs):
                rec = {"chunk_id": cid, **doc.metadata}
                manifest_fp.write(json.dumps(rec) + "\n")
        prepared_texts = []
        prepared_docs = []
        prepared_ids = []

    for row_index, row in enumerate(rows):
        raw_id = str(row.get(args.id_col, '')).strip()
        text = str(row.get(args.text_col, '') or '').strip()
        if not raw_id or not text or len(text) < args.min_chars:
            continue
        base_id = raw_id.replace('PMC','').replace('pmc','') if raw_id.lower().startswith('pmc') else raw_id
        chunks = split_text(text, args.chunk_size, args.overlap)
        if not chunks:
            continue
        n_est = math.ceil(len(text)/max(1,args.chunk_size))
        for c_idx, chunk in enumerate(chunks):
            chunk_id = f"{base_id}_chunk{c_idx}"
            meta = {
                'source_id': raw_id,
                'row_index': row_index,
                'chunk_index': c_idx,
                'text_length': len(chunk),
                'chunk_size': args.chunk_size,
                'overlap': args.overlap,
                'n_chunks_total_est': n_est,
            }
            prepared_ids.append(chunk_id)
            prepared_texts.append(chunk)
            prepared_docs.append(Document(page_content=chunk, metadata=meta))
        if len(prepared_texts) >= args.batch_size:
            flush_batch()

    flush_batch()
    if manifest_fp:
        manifest_fp.close()
    print(f"[vectorize_csv] Done. Total chunks processed: {total_chunks}")

if __name__ == '__main__':
    main()
