"""Google Gemini chat client using the new google-genai SDK.

Requirements:
    pip install google-genai

Environment Variables:
    GOOGLE_API_KEY: Your Google AI API key
    GOOGLE_AI_MODEL: Model name (default: gemini-2.0-flash)
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

# Default model - Gemini 2.0 Flash is fast and cost-effective
DEFAULT_MODEL = os.getenv("GOOGLE_AI_MODEL", "gemini-2.0-flash")

# Import Google GenAI (new SDK)
_genai = None
try:
    from google import genai
    _genai = genai
except ImportError:
    logger.warning("google-genai not installed. Run: pip install google-genai")


class GoogleAIChatClient:
    """Chat client for Google's Gemini models."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.stub = False
        self._client = None
        
        api_key = os.getenv("GOOGLE_API_KEY", "")
        
        if api_key and _genai:
            self._client = _genai.Client(api_key=api_key)
            logger.info(f"Using Google AI with API key, model: {model}")
        else:
            self.stub = True
            if not api_key:
                logger.warning("GOOGLE_API_KEY not set")
            elif not _genai:
                logger.warning("google-genai not installed. Run: pip install google-genai")
    
    def generate(self, system: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """Generate a chat completion.
        
        Parameters
        ----------
        system : str
            System prompt/instruction.
        messages : list of {role, content}
            Chat history. Roles should be 'user' or 'assistant'.
        temperature : float
            Sampling temperature (0.0-1.0).
        
        Returns
        -------
        str
            The model's response text.
        """
        if self.stub:
            user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"(STUB:{self.model}) System:{system[:40]}... User:{user[:200]}"
        
        try:
            # Build the contents for the API
            # Prepend system instruction to the conversation
            contents = []
            
            for msg in messages:
                role = msg["role"]
                # Map 'assistant' to 'model' for Gemini
                if role == "assistant":
                    role = "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "system_instruction": system,
                    "temperature": temperature,
                    "max_output_tokens": 2048,
                }
            )
            
            return response.text
                
        except Exception as e:
            logger.error(f"Google AI generation error: {e}")
            raise


_shared_client: GoogleAIChatClient | None = None


def get_chat_client(model: str = DEFAULT_MODEL) -> GoogleAIChatClient:
    """Get a shared GoogleAIChatClient instance."""
    global _shared_client
    if _shared_client is None or _shared_client.model != model:
        _shared_client = GoogleAIChatClient(model=model)
    return _shared_client


def get_status() -> dict:
    """Return diagnostic information about the Google AI client status."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_set = bool(api_key)
    genai_available = _genai is not None
    
    stub = not (api_key_set and genai_available)
    
    if api_key_set and genai_available:
        reason = "ok"
    elif not api_key_set:
        reason = "no_api_key"
    else:
        reason = "sdk_missing"
    
    return {
        "stub": stub,
        "reason": reason,
        "api_key_set": api_key_set,
        "google_genai_available": genai_available,
        "model": DEFAULT_MODEL,
    }


__all__ = ["GoogleAIChatClient", "get_chat_client", "get_status", "DEFAULT_MODEL"]
