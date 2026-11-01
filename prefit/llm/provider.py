"""Pluggable LLM provider interface.

The interpreter depends only on this interface. Shipped implementations:
Anthropic, Gemini, and a local OpenAI-compatible endpoint (Ollama, llama.cpp,
vLLM) for fully local use. Subclass Provider to plug in anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Minimal contract: send a system + user prompt, get text back."""

    model: str = "?"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's text response for the given system+user prompt."""
        raise NotImplementedError
