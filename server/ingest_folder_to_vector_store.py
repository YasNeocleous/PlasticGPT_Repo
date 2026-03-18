"""
Ingest full text files from a folder directly into the vector store.
Does NOT modify any CSV - just reads .txt files and uploads to the vector store.

Usage:
    # Ingest all new files from full_texts folder
    python -m server.ingest_folder_to_vector_store -v

    # Limit to first 50 files (for testing)
    python -m server.ingest_folder_to_vector_store --max 50 -v

    # Force re-ingest even if already in store
    python -m server.ingest_folder_to_vector_store --overwrite -v
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import re
from typing import List, Dict, Any

from tqdm import tqdm

from .embedding import embed_texts
from .vector_store import Document, get_store

logger = logging.getLogger(__name__)


def derive_pmcid_from_filename(filename: str) -> str | None:
    """Handle pmcid<digits>.txt or PMC<digits>.txt variants."""
    low = filename.lower()
    if not low.endswith('.txt'):
        return None
    stem = low[:-4]  # remove .txt
    # pmcid1234567
    if stem.startswith('pmcid'):
        digits = stem[len('pmcid'):]
        if digits.isdigit():
            return f"PMC{digits}"
    # PMC1234567
    if stem.startswith('pmc'):
        digits = stem[len('pmc'):]
        if digits.isdigit():
            return f"PMC{digits}"
    return None


def _chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    """Naive token approximation using words."""
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


def ingest_folder(
    text_dir: str,
    min_chars: int = 100,
    max_files: int | None = None,
    overwrite: bool = False,
    save_cache: bool = True,
) -> int:
    """
    Ingest text files from folder directly into vector store.
    
    Args:
        text_dir: Directory containing .txt files
        min_chars: Minimum text length to accept
        max_files: Optional limit on files to process
        overwrite: If False, skip files whose PMCID is already in store
        save_cache: Whether to save the cache after ingestion
    
    Returns:
        Number of chunks ingested
    """
    if not os.path.isdir(text_dir):
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    
    store = get_store()
    
    # Build set of existing PMCIDs in store
    existing_pmcids = set()
    if not overwrite:
        for _, _, doc in store._data:
            pmcid = doc.metadata.get("pmcid", "")
            if pmcid:
                existing_pmcids.add(pmcid.upper())
        logger.info(f"Found {len(existing_pmcids)} existing PMCIDs in vector store")
    
    # Get list of text files
    files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    logger.info(f"Found {len(files)} text files in {text_dir}")
    
    if max_files:
        files = files[:max_files]
    
    all_texts: List[str] = []
    docs: List[Document] = []
    ids: List[str] = []
    
    ingested_files = 0
    skipped_existing = 0
    skipped_short = 0
    
    for fname in tqdm(files, desc="Processing files"):
        pmcid = derive_pmcid_from_filename(fname)
        if not pmcid:
            continue
        
        # Skip if already in store
        if not overwrite and pmcid.upper() in existing_pmcids:
            skipped_existing += 1
            continue
        
        path = os.path.join(text_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")
            continue
        
        if len(text) < min_chars:
            skipped_short += 1
            continue
        
        # Chunk the text
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            import uuid
            doc = Document(
                page_content=chunk,
                metadata={
                    "pmcid": pmcid,
                    "title": f"PMC Article {pmcid}",
                    "chunk_index": i,
                    "source_file": fname,
                }
            )
            docs.append(doc)
            all_texts.append(chunk)
            ids.append(str(uuid.uuid4()))
        
        ingested_files += 1
    
    if not docs:
        logger.warning("No new documents to ingest")
        return 0
    
    # Embed and add to store
    logger.info(f"Embedding {len(all_texts)} chunks from {ingested_files} files...")
    vectors = embed_texts(all_texts)
    store.add(ids, vectors, docs)
    
    # Save cache
    if save_cache:
        logger.info("Saving vector store cache...")
        store.save_cache()
    
    logger.info(f"Ingestion complete:")
    logger.info(f"  - Files ingested: {ingested_files}")
    logger.info(f"  - Chunks created: {len(docs)}")
    logger.info(f"  - Skipped (already in store): {skipped_existing}")
    logger.info(f"  - Skipped (too short): {skipped_short}")
    
    return len(docs)


def main():
    p = argparse.ArgumentParser(description='Ingest folder full_text .txt files directly into vector store')
    default_txt = os.path.join(os.path.dirname(__file__), 'pubmed_data', 'full_texts')
    
    p.add_argument('--text-dir', default=default_txt, help='Directory containing pmcid*.txt files')
    p.add_argument('--min-chars', type=int, default=100, help='Minimum text length to accept')
    p.add_argument('--max', type=int, default=None, help='Limit to first N files (for testing)')
    p.add_argument('--overwrite', action='store_true', help='Re-ingest even if PMCID already in store')
    p.add_argument('--no-save', action='store_true', help='Do not save cache after ingestion')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = p.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    try:
        n = ingest_folder(
            text_dir=args.text_dir,
            min_chars=args.min_chars,
            max_files=args.max,
            overwrite=args.overwrite,
            save_cache=not args.no_save,
        )
        print(f"\n✅ Ingested {n} chunks into vector store")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
