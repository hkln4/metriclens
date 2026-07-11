"""Turns deterministic findings into a reasoned narrative. The interpreter sits
ABOVE the trust boundary: it sees only the computed profile/findings/context,
never raw data, never computes a statistic. If no provider, or the call fails,
it falls back deterministically — the structured roadmap is already complete."""
from __future__ import annotations
import warnings
from ..profiling.profile import DataProfile
from ..rules.finding import Finding
from .prompt import build_messages
from .provider import Provider

def deterministic_narrative(profile, findings):
    if not findings:
        return (f"No notable issues detected for a {profile.task.value} task on "
                f"{profile.modality.value} data with {profile.core.n_samples} samples.")
    parts = [f"Detected {len(findings)} point(s) worth attention:"]
    for f in findings:
        parts.append(f"- [{f.severity.value}] {f.message}")
    return "\n".join(parts)

def interpret(profile, findings, context, provider):
    """Return (narrative, source). source is 'llm' or 'deterministic'."""
    if provider is None:
        return deterministic_narrative(profile, findings), "deterministic"
    try:
        system, user = build_messages(profile, findings, context)
        text = provider.complete(system, user)
        if not text:
            raise ValueError("empty response from provider")
        return text, "llm"
    except Exception as exc:  # any failure must not crash analyze
        warnings.warn(
            f"LLM interpretation failed, using deterministic narrative: {exc}",
            stacklevel=2,
        )
        return deterministic_narrative(profile, findings), "deterministic"