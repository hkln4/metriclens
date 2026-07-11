"""Public entry points and orchestration.

`analyze` wires the pipeline together:
    profile -> rules -> (optional) llm interpretation -> roadmap
The pipeline is shared across modalities; only the profiler differs.

The LLM is opt-in and degrades gracefully: if use_llm is False, or no API key is
available, or the `[llm]` extra is not installed, the deterministic roadmap is
returned with a warning rather than an error — because the roadmap is already
complete before the LLM is ever consulted.
"""

from __future__ import annotations

import warnings
from typing import Any

from .llm.interpreter import interpret
from .llm.provider import Provider
from .profiling.dispatcher import profile_data
from .roadmap import Roadmap, assemble_roadmap
from .rules.engine import run_rules
from .enums import Modality, Task


def _resolve_provider_class(name: str):
    """Map a provider name to its class (imported lazily, SDK not required)."""
    key = name.lower()
    if key == "anthropic":
        from .llm.anthropic import AnthropicProvider
        return AnthropicProvider
    if key == "gemini":
        from .llm.gemini import GeminiProvider
        return GeminiProvider
    raise ValueError(
        f"Unknown provider '{name}'. Known: 'anthropic', 'gemini'. "
        "You can also pass a Provider instance directly.")

def _build_provider(use_llm, provider, model, api_key):
    """provider may be a name ('anthropic'/'gemini') or a ready Provider instance."""
    if not use_llm:
        return None
    if isinstance(provider, Provider):
        return provider
    try:
        cls = _resolve_provider_class(provider)
        return cls(api_key=api_key, model=model)
    except (ImportError, ValueError) as exc:
        warnings.warn(f"Falling back to the deterministic roadmap (no LLM): {exc}",
                      stacklevel=2)
        return None


def analyze(
    data: Any,
    target: str | None = None,
    task: Task | None = None,
    context: str | None = None,
    modality: Modality = Modality.TABULAR,
    use_llm: bool = False,
    provider: str | Provider = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Roadmap:
    """Profile a dataset and produce a reasoned evaluation roadmap.

    Parameters
    ----------
    data : the dataset (a DataFrame-like object for tabular).
    target : name of the target column/field.
    task : the learning task; inferred and flagged for confirmation if omitted.
    context : free-text description of the project, used only by the LLM layer
        to prioritize and explain (e.g. "clinical EF estimation, interpretability
        prioritized").
    modality : data modality; only TABULAR is implemented in v0.
    use_llm : opt into LLM-written narrative. Requires the `[llm]` extra and a
        user-provided API key. Falls back to deterministic prose otherwise.
    api_key : Anthropic key; if omitted, ANTHROPIC_API_KEY is used.
    """
    profile = profile_data(
        data, target=target, modality=modality, task=task, **kwargs
    )
    findings = run_rules(profile)
    llm = _build_provider(use_llm, provider, model, api_key)
    narrative, source = interpret(profile, findings, context, llm)

    return assemble_roadmap(findings, narrative, source, profile)


def review(
    y_true: Any,
    y_pred: Any,
    roadmap: Roadmap | None = None,
    task: Task | None = None,
) -> Any:
    """Second pass: check whether the init roadmap's hypothesis held.

    Looks at the actual error distribution after training and compares it to what
    `analyze` predicted (e.g. "skewed target -> a few large errors dominate; prefer
    MAE"). Deterministic, like the rest of the library; sees only y_true / y_pred.
    """
    from .error_analysis import review as _review

    return _review(y_true, y_pred, roadmap=roadmap, task=task)
