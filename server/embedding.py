"""Google AI embeddings - Simple wrapper for text-embedding-004.

Uses Google's GenAI SDK for embeddings.
"""

from __future__ import annotations

import os
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

# Default model - 768 dimensions
EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "text-embedding-004")
EMBEDDING_DIMENSION = 768

# Import Google GenAI (new SDK)
_client = None
try:
    from google import genai
    
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        _client = genai.Client(api_key=api_key)
        logger.info(f"Google AI embeddings configured, model: {EMBED_MODEL}")
    else:
        logger.warning("GOOGLE_API_KEY not set - embeddings will fail")
except ImportError:
    logger.warning("google-genai not installed. Run: pip install google-genai")


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embed a list of texts using Google AI.
    
    Args:
        texts: List of text strings to embed
        batch_size: Texts per API call (for batching)
        
    Returns:
        List of embedding vectors (768 dimensions each)
    """
    if not _client:
        logger.error("Google AI not available")
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]
    
    all_embeddings: List[List[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        for text in batch:
            try:
                result = _client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                )
                all_embeddings.append(result.embeddings[0].values)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                all_embeddings.append([0.0] * EMBEDDING_DIMENSION)
    
    print(f"[embedding] Embedded {len(texts)} texts using Google AI")
    return all_embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query (optimized for search)."""
    if not _client:
        return [0.0] * EMBEDDING_DIMENSION
    
    try:
        result = _client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Query embedding error: {e}")
        return [0.0] * EMBEDDING_DIMENSION


__all__ = ["embed_texts", "embed_query", "EMBEDDING_DIMENSION"]

