"""Chunk text storage utilities.

Store chunk text files externally (pubmed_data/chunks/<chunk_id>.txt)
instead of embedding full text in vector DB metadata.

Also provides utilities to generate short summaries for metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_CHUNKS_DIR = Path(__file__).parent / "pubmed_data" / "chunks"


def get_chunks_directory() -> Path:
    """Get the directory where chunk text files are stored.
    
    Can be overridden via CHUNKS_DIR environment variable.
    """
    chunks_dir_env = os.getenv("CHUNKS_DIR")
    if chunks_dir_env:
        return Path(chunks_dir_env)
    return DEFAULT_CHUNKS_DIR


def save_chunk_text(chunk_id: str, text: str, chunks_dir: Path | None = None) -> str:
    """Save chunk text to disk.
    
    Args:
        chunk_id: Unique chunk identifier (typically a hash)
        text: The chunk text content
        chunks_dir: Directory to save chunks (defaults to pubmed_data/chunks)
    
    Returns:
        Relative path to the saved chunk file (from project root)
    """
    if chunks_dir is None:
        chunks_dir = get_chunks_directory()
    
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunks_dir / f"{chunk_id}.txt"
    
    with chunk_file.open("w", encoding="utf-8") as f:
        f.write(text)
    
    # Return relative path from server directory for portability
    try:
        rel_path = chunk_file.relative_to(Path(__file__).parent)
        return str(rel_path)
    except ValueError:
        # If not relative, return absolute path
        return str(chunk_file)


def load_chunk_text(chunk_id: str, chunks_dir: Path | None = None) -> Optional[str]:
    """Load chunk text from disk.
    
    Args:
        chunk_id: Unique chunk identifier
        chunks_dir: Directory where chunks are stored
    
    Returns:
        The chunk text, or None if not found
    """
    if chunks_dir is None:
        chunks_dir = get_chunks_directory()
    
    chunk_file = chunks_dir / f"{chunk_id}.txt"
    
    if not chunk_file.exists():
        return None
    
    with chunk_file.open("r", encoding="utf-8") as f:
        return f.read()


def load_chunk_text_by_path(chunk_path: str) -> Optional[str]:
    """Load chunk text by relative or absolute path.
    
    Args:
        chunk_path: Path to chunk file (relative or absolute)
    
    Returns:
        The chunk text, or None if not found
    """
    # Try relative to server directory first
    server_dir = Path(__file__).parent
    abs_path = server_dir / chunk_path
    
    if not abs_path.exists():
        # Try as absolute path
        abs_path = Path(chunk_path)
    
    if not abs_path.exists():
        return None
    
    with abs_path.open("r", encoding="utf-8") as f:
        return f.read()


def generate_summary(text: str, max_length: int = 200) -> str:
    """Generate a short summary of chunk text for metadata storage.
    
    This is a simple truncation-based summary. For production, consider:
    - Using an LLM to generate proper summaries
    - Extractive summarization (select key sentences)
    - Abstractive summarization models
    
    Args:
        text: The full chunk text
        max_length: Maximum length of summary in characters
    
    Returns:
        Summary string (truncated with ellipsis if needed)
    """
    text = text.strip()
    if len(text) <= max_length:
        return text
    
    # Try to break at sentence boundary
    truncated = text[:max_length]
    
    # Find last period, question mark, or exclamation within the truncated text
    last_sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("?"),
        truncated.rfind("!")
    )
    
    if last_sentence_end > max_length // 2:
        # If we found a sentence boundary in the second half, use it
        return truncated[:last_sentence_end + 1]
    else:
        # Otherwise, just truncate and add ellipsis
        # Try to break at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            return truncated[:last_space] + "..."
        return truncated + "..."


def chunk_exists(chunk_id: str, chunks_dir: Path | None = None) -> bool:
    """Check if a chunk file exists on disk.
    
    Args:
        chunk_id: Unique chunk identifier
        chunks_dir: Directory where chunks are stored
    
    Returns:
        True if chunk file exists
    """
    if chunks_dir is None:
        chunks_dir = get_chunks_directory()
    
    chunk_file = chunks_dir / f"{chunk_id}.txt"
    return chunk_file.exists()


__all__ = [
    "save_chunk_text",
    "load_chunk_text",
    "load_chunk_text_by_path",
    "generate_summary",
    "chunk_exists",
    "get_chunks_directory",
    "DEFAULT_CHUNKS_DIR",
]
