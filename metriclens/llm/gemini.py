"""Gemini implementation of the Provider interface. Same contract as
AnthropicProvider, different vendor. google-genai is an OPTIONAL dependency,
imported lazily inside __init__. Key comes from api_key=... or GEMINI_API_KEY."""
from __future__ import annotations
import os
from .provider import Provider

_DEFAULT_MODEL = "gemini-2.5-flash"
_INSTALL_HINT = ("Gemini support requires the google-genai SDK. "
                 "Install it with: pip install google-genai")

class GeminiProvider(Provider):
    def __init__(self, api_key=None, model=None, max_tokens=2048):
        try:
            from google import genai  # lazy: only needed when Gemini is used
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        key = (api_key or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
        if not key:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY or pass api_key=... . "
                "Alternatively run with use_llm=False for the deterministic roadmap.")
        self._client = genai.Client(api_key=key)
        self._model = model or _DEFAULT_MODEL
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self._max_tokens,
                # 2.5-* are thinking models: by default "thinking" tokens eat the
                # output budget and the visible answer gets truncated (MAX_TOKENS).
                # We only need prose here, so disable thinking to spend the whole
                # budget on the narrative.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (response.text or "").strip()