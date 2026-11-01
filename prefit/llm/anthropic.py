"""Anthropic implementation of the Provider interface.

`anthropic` is an optional dependency (the `[llm]` extra), imported lazily so
`import prefit` works with only the core dependencies installed.
"""

from __future__ import annotations

import os

from .provider import Provider

_DEFAULT_MODEL = "claude-sonnet-5-5"
_INSTALL_HINT = "Anthropic support requires the 'llm' extra: pip install 'prefit[llm]'"


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API. The key comes from
    api_key=... or ANTHROPIC_API_KEY; the library never ships one."""

    def __init__(self, api_key: str | None = None, model: str = _DEFAULT_MODEL, max_tokens: int = 1500) -> None:
        try:
            import anthropic  # lazy: only needed when the LLM layer is used
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise ImportError(_INSTALL_HINT) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=... . "
                             "The report is complete without an LLM; llm= is optional.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
