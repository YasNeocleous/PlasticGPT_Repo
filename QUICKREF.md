# RAG Pipeline Quick Reference

## Common Commands

```powershell
# Run incremental ingestion
python -m server.ingestion

# Run smoke tests
python server\test_ingestion_idempotent.py

# Start server (development)
uvicorn server.main:app --reload

# Start server (production)
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Key Concepts

### Chunk ID
- **Purpose:** Deterministic identifier for deduplication
- **Formula:** `sha256(source_id + chunk_index + text)`
- **Benefit:** Same chunk always gets same ID → skip re-embedding

### Manifest
- **File:** `server/pubmed_data/vector_manifest.jsonl`
- **Format:** One JSON object per line
- **Schema:** `{chunk_id, metadata, embedding_stored}`
- **Benefit:** Track what's been ingested, enable incremental updates

### Chunk Storage
- **Dir:** `server/pubmed_data/chunks/`
- **Files:** `<chunk_id>.txt`
- **Benefit:** Separate full text from vector DB, reduce storage costs

### Retrieval Pipeline
- **Stages:** Semantic search → BM25 (optional) → Re-rank (optional) → MMR (optional) → Load text
- **Config:** See `.env` variables `RETRIEVAL_*`
- **Benefit:** Better relevance, diversity, and cost control

## Environment Variables Reference

```bash
# Retrieval
RETRIEVAL_USE_BM25=0              # Lexical search (0/1)
RETRIEVAL_USE_RERANK=0            # Re-ranking (0/1)
RETRIEVAL_USE_MMR=0               # Diversity (0/1)
RETRIEVAL_TOP_K_CANDIDATES=20     # Initial candidates
RETRIEVAL_FINAL_TOP_K=4           # Final results

# Storage
CHUNKS_DIR=server/pubmed_data/chunks

# Vector DB
VECTOR_BACKEND=memory             # or 'pinecone'
PINECONE_API_KEY=...
PINECONE_INDEX=plasticgpt
PINECONE_DIM=1536
PINECONE_STORE_TEXT=0

# Embeddings
OPENAI_API_KEY=...
OPENAI_EMBED_MODEL=text-embedding-3-small
```

## File Structure

```
server/
  manifest.py              # Manifest management
  chunk_storage.py         # Chunk file I/O
  ingestion.py            # Incremental ingestion (UPDATED)
  retrieval.py            # Hybrid retrieval pipeline (NEW)
  doc_summary.py          # Document summaries (NEW)
  main.py                 # FastAPI app (UPDATED)
  test_ingestion_idempotent.py  # Tests (NEW)
  
  pubmed_data/
    vector_manifest.jsonl # Chunk manifest (AUTO-CREATED)
    doc_summaries.jsonl   # Doc summaries (OPTIONAL)
    chunks/               # Chunk text files (AUTO-CREATED)
      <chunk_id>.txt
```

## Ingestion Stats Explained

```python
{
  'total_chunks': 2340,      # Total chunks in documents
  'new_chunks': 245,         # New chunks (not in manifest)
  'skipped_chunks': 2095,    # Existing chunks (in manifest)
  'embedded_chunks': 245     # Chunks that were embedded
}
```

**Good:** `skipped_chunks` > 0 on re-run (dedup working!)  
**Bad:** `new_chunks` == `total_chunks` every time (manifest not persisting)

## Testing Workflow

1. **Run smoke tests first:**
   ```powershell
   python server\test_ingestion_idempotent.py
   ```

2. **Check for:**
   - ✅ All tests pass
   - ✅ Manifest file created
   - ✅ Chunk files created
   - ✅ No duplicate chunks on re-run

3. **Run full ingestion:**
   ```powershell
   python -m server.ingestion
   ```

4. **Verify idempotency:**
   ```powershell
   # Run again - should skip all chunks
   python -m server.ingestion
   ```
   Expect: `skipped_chunks` = all chunks, `embedded_chunks` = 0

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| Manifest not found | Normal on first run (auto-created) |
| High embedding costs | Check `skipped_chunks` stat; should be > 0 on re-run |
| Duplicate chunks | Delete `vector_manifest.jsonl` and re-ingest |
| Irrelevant results | Enable `RETRIEVAL_USE_BM25=1` and/or `RETRIEVAL_USE_MMR=1` |
| Chunk file not found | Check `CHUNKS_DIR` permissions and path |

## Migration Checklist

- [ ] Read `MIGRATION_GUIDE.md`
- [ ] Run smoke tests
- [ ] Back up existing data (if any)
- [ ] Run incremental ingestion
- [ ] Verify manifest created
- [ ] Verify chunk files created
- [ ] Test retrieval with queries
- [ ] Configure optional features (BM25, MMR)
- [ ] Deploy to production

## Performance Targets

| Metric | Before | After |
|--------|--------|-------|
| Re-ingest cost | $5-10 | $0.10 |
| Vector DB size | 100MB | 10MB |
| Retrieval time | 50-100ms | 60-120ms |
| Relevance | Good | Excellent |

---

**Need more details?** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
