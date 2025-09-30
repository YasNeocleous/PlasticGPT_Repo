"""OpenAI chat client wrapper with optional local stub.

This module provides a thin abstraction so the rest of the codebase does not
depend directly on the OpenAI SDK. If the environment variable OPENAI_API_KEY
is missing or starts with the word "test" we run in a deterministic stub mode
that returns canned responses (useful for unit tests / offline dev).
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import logging
from pathlib import Path

# Prefer loading the .env file located next to this module (server/.env)
# so the uvicorn process sees variables even when started from the repo root.
try:
	from dotenv import load_dotenv  # type: ignore
	env_path = Path(__file__).parent / ".env"
	if env_path.exists():
		load_dotenv(dotenv_path=env_path)
		logging.getLogger(__name__).info(f"Loaded environment from {env_path}")
	else:
		logging.getLogger(__name__).debug("No server/.env file found to load")
except Exception:
	logging.getLogger(__name__).debug("python-dotenv not installed; server will not auto-load server/.env. Install with: pip install python-dotenv")

logger = logging.getLogger(__name__)
=======
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


# Load environment variables from a local .env if present
try:  # pragma: no cover - convenience for local dev
	from dotenv import load_dotenv, find_dotenv  # type: ignore
	load_dotenv(find_dotenv(usecwd=True), override=False)  # find nearest .env
except Exception:  # pragma: no cover
	pass


try:
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - library may not be installed yet
	OpenAI = None  # type: ignore


# Allow overriding the chat model via env var
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class OpenAIChatClient:
	def __init__(self, model: str = DEFAULT_MODEL):
		self.model = model
		self.api_key = os.getenv("OPENAI_API_KEY", "")
		self.stub = (not self.api_key) or self.api_key.lower().startswith("test") or OpenAI is None
		if not self.stub:
			self._client = OpenAI()
		else:
			self._client = None
			# Log why we're running in stub mode to make debugging easier
			if not self.api_key:
				logger.warning("OpenAI API key not found. Running in stub mode — set OPENAI_API_KEY to enable real responses.")
			elif self.api_key.lower().startswith("test"):
				logger.warning("OPENAI_API_KEY starts with 'test' — running in stub mode (test key).")
			elif OpenAI is None:
				logger.warning("OpenAI SDK not available (import failed). Install the 'openai' package to enable real responses.")

	def generate(self, system: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
		"""Generate a chat completion.

		Parameters
		----------
		system: str
			System prompt string.
		messages: list of {role, content}
			Messages excluding system; will be prepended.
		temperature: float
			Sampling temperature.
		"""
		if self.stub:
			# Deterministic stub: echo last user content with prefix.
			user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
			return f"(STUB:{self.model}) System:{system[:40]}... User:{user[:200]}"

		completion = self._client.chat.completions.create(
			model=self.model,
			temperature=temperature,
			messages=[{"role": "system", "content": system}] + messages,
		)
		return completion.choices[0].message.content  # type: ignore


_shared_client: OpenAIChatClient | None = None


def get_chat_client(model: str = DEFAULT_MODEL) -> OpenAIChatClient:
	global _shared_client
	if _shared_client is None or _shared_client.model != model:
		_shared_client = OpenAIChatClient(model=model)
	return _shared_client


__all__ = ["OpenAIChatClient", "get_chat_client", "DEFAULT_MODEL"]


def get_status() -> dict:
	"""Return diagnostic information about whether the module is running
	in stub mode and why. Useful to query at runtime from a web endpoint.
	"""
	import os

	api_key = os.getenv("OPENAI_API_KEY", "")
	api_key_set = bool(api_key)
	sdk_available = OpenAI is not None
	stub = (not api_key_set) or api_key.lower().startswith("test") or not sdk_available
	reason = None
	if not api_key_set:
		reason = "no_api_key"
	elif api_key.lower().startswith("test"):
		reason = "test_key"
	elif not sdk_available:
		reason = "sdk_missing"
	else:
		reason = "ok"
	return {
		"stub": stub,
		"reason": reason,
		"api_key_set": api_key_set,
		"openai_sdk_available": sdk_available,
		"model": DEFAULT_MODEL,
	}

__all__.extend(["get_status"])

