"""Local provider: any OpenAI-compatible chat endpoint (Ollama, llama.cpp server,
vLLM, LM Studio). Uses only the standard library, so nothing leaves the machine
when base_url points at localhost (the default)."""

from __future__ import annotations

import json
import os
import urllib.request

from .provider import Provider

_DEFAULT_URL = "http://localhost:11434/v1"   # Ollama


class LocalProvider(Provider):
    def __init__(self, base_url: str | None = None, model: str = "llama3.1", api_key: str | None = None,
                 max_tokens: int = 1500, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.environ.get("PREFIT_LLM_URL") or _DEFAULT_URL).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL.")
        self.model = model
        self._key = api_key
        self._max_tokens = max_tokens
        self._timeout = timeout

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.model, "max_tokens": self._max_tokens, "temperature": 0,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        # scheme validated in __init__ (http/https only)
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - user-configured URL
            data = json.loads(resp.read().decode())
        return (data["choices"][0]["message"]["content"] or "").strip()
