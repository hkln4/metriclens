"""Anthropic implementation of the Provider interface.

Key design point: `anthropic` is an OPTIONAL dependency (the `[llm]` extra). The
import is lazy — it happens inside __init__, not at module top level — so that
`import metriclens` always works with only the core deps installed. The SDK is
only required when a user actually opts into LLM enrichment.
"""

from __future__ import annotations

import os

from .provider import Provider

_DEFAULT_MODEL = "claude-sonnet-4-6"
_INSTALL_HINT = (
    "LLM support requires the 'llm' extra. "
    "Install it with: pip install metriclens[llm]"
)


class AnthropicProvider(Provider):
    """Provider backed by the Anthropic Messages API.

    The API key is brought by the user — via the ANTHROPIC_API_KEY environment
    variable or passed explicitly. The library never ships or owns a key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic  # lazy: only needed when LLM is actually used
        except ImportError as exc:  # pragma: no cover - exercised via extras
            raise ImportError(_INSTALL_HINT) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass "
                "api_key=... . Alternatively run with use_llm=False for the "
                "deterministic roadmap."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # response.content is a list of blocks; collect the text ones.
        parts = [block.text for block in response.content
                 if getattr(block, "type", None) == "text"]
        return "\n".join(parts).strip()
