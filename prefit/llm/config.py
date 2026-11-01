"""What the LLM layer may see and do.

Privacy defaults: nothing is sent unless an LLMConfig (or provider) is passed.
Raw rows are never sent. Column names can be hidden behind aliases
(`share_column_names=False`), category labels are never sent, and the free-text
context is only sent when `include_context=True`. `Report.llm_payload()` shows
the exact payload before anything leaves the machine.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any = "anthropic"          # "anthropic" | "gemini" | "local" | Provider instance
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None          # for provider="local" (OpenAI-compatible endpoint)
    share_column_names: bool = True
    include_context: bool = True
    max_tokens: int = 1500


def coerce_config(llm: Any) -> LLMConfig:
    from .provider import Provider

    if isinstance(llm, LLMConfig):
        return llm
    if isinstance(llm, (str, Provider)):
        return LLMConfig(provider=llm)
    raise TypeError("llm must be an LLMConfig, a provider name ('anthropic', 'gemini', 'local') or a Provider.")
