"""Gemini implementation of the Provider interface. google-genai is optional and
imported lazily. The key comes from api_key=... or GEMINI_API_KEY / GOOGLE_API_KEY."""

from __future__ import annotations

import os

from .provider import Provider

_DEFAULT_MODEL = "gemini-2.5-flash"
_INSTALL_HINT = "Gemini support requires the 'gemini' extra: pip install 'metriclens[gemini]'"


class GeminiProvider(Provider):
    def __init__(self, api_key: str | None = None, model: str | None = None, max_tokens: int = 2048) -> None:
        try:
            from google import genai  # lazy: only needed when Gemini is used
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from exc
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No Gemini API key found. Set GEMINI_API_KEY or pass api_key=... . "
                             "The report is complete without an LLM; llm= is optional.")
        self._client = genai.Client(api_key=key)
        self.model = model or _DEFAULT_MODEL
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self._max_tokens,
                # 2.5-* are thinking models; thinking tokens would eat the output budget
                thinking_config=types.ThinkingConfig(thinking_budget=0)))
        return (response.text or "").strip()
