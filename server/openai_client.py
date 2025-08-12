"""OpenAI chat client wrapper with optional local stub.

This module provides a thin abstraction so the rest of the codebase does not
depend directly on the OpenAI SDK. If the environment variable OPENAI_API_KEY
is missing or starts with the word "test" we run in a deterministic stub mode
that returns canned responses (useful for unit tests / offline dev).
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

try:
	from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - library may not be installed yet
	OpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIChatClient:
	def __init__(self, model: str = DEFAULT_MODEL):
		self.model = model
		self.api_key = os.getenv("OPENAI_API_KEY", "")
		self.stub = (not self.api_key) or self.api_key.lower().startswith("test") or OpenAI is None
		if not self.stub:
			self._client = OpenAI()
		else:
			self._client = None

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

