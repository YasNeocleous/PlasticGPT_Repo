"""Persistent embedding manifest for deduplication and caching.

The manifest is a JSONL file where each line is a JSON object with:
- chunk_id: sha256 hash of (source_id + chunk_index + text)
- embedding_path: path to saved embedding file (optional, can store inline)
- metadata: {pmid, title, text_length, chunk_path, chunk_index, source_id, summary}

This allows incremental ingestion: compute chunk_id, check if it exists,
skip re-embedding if already processed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


DEFAULT_MANIFEST_PATH = Path(__file__).parent / "pubmed_data" / "vector_manifest.jsonl"


def compute_chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    """Compute deterministic chunk ID from source + index + text.
    
    Args:
        source_id: Unique identifier for the source document (e.g., PMID)
        chunk_index: Index of this chunk within the document
        text: The chunk text content
    
    Returns:
        SHA256 hex digest as chunk_id
    """
    h = hashlib.sha256()
    h.update(source_id.encode("utf-8"))
    h.update(b"\0")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class ManifestManager:
    """Manages the persistent manifest of ingested chunks."""
    
    def __init__(self, manifest_path: Path | str | None = None):
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Dict[str, Any]]:
        """Load manifest from disk into memory.
        
        Returns:
            Dictionary mapping chunk_id -> manifest entry
        """
        if self._loaded:
            return self._entries
        
        self._entries = {}
        if not self.manifest_path.exists():
            print(f"[manifest] No existing manifest at {self.manifest_path}")
            self._loaded = True
            return self._entries
        
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunk_id = obj.get("chunk_id")
                    if chunk_id:
                        self._entries[chunk_id] = obj
                except json.JSONDecodeError as e:
                    print(f"[manifest] Error parsing line {line_num}: {e}")
        
        print(f"[manifest] Loaded {len(self._entries)} entries from {self.manifest_path}")
        self._loaded = True
        return self._entries
    
    def has_chunk(self, chunk_id: str) -> bool:
        """Check if a chunk_id already exists in the manifest."""
        if not self._loaded:
            self.load()
        return chunk_id in self._entries
    
    def get_entry(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get manifest entry for a chunk_id."""
        if not self._loaded:
            self.load()
        return self._entries.get(chunk_id)
    
    def append_entry(self, entry: Dict[str, Any]) -> None:
        """Append a new entry to the manifest.
        
        Args:
            entry: Dictionary with at minimum 'chunk_id', optionally
                   'embedding_path', 'metadata', etc.
        """
        chunk_id = entry.get("chunk_id")
        if not chunk_id:
            raise ValueError("Entry must have 'chunk_id'")
        
        # Ensure directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to file
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Update in-memory cache
        if not self._loaded:
            self.load()
        self._entries[chunk_id] = entry
    
    def count(self) -> int:
        """Return the number of entries in the manifest."""
        if not self._loaded:
            self.load()
        return len(self._entries)
    
    def rebuild_index(self) -> None:
        """Rebuild the manifest by re-reading from disk (useful after external edits)."""
        self._loaded = False
        self._entries = {}
        self.load()


def get_manifest_manager(manifest_path: Path | str | None = None) -> ManifestManager:
    """Get or create a singleton ManifestManager instance."""
    # For simplicity, create a new instance each time; in production you might cache it
    return ManifestManager(manifest_path)


__all__ = [
    "compute_chunk_id",
    "ManifestManager",
    "get_manifest_manager",
    "DEFAULT_MANIFEST_PATH",
]
