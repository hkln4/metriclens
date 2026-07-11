"""DataProfile -> list[Finding]. After this layer a complete roadmap exists, no
LLM required. Findings ordered critical-first."""
from __future__ import annotations
from ..profiling.profile import DataProfile
from ..enums import Severity
from .finding import Finding
from .registry import CATALOG, matches
from . import catalog  # noqa: F401  (importing registers every rule)

_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}

def run_rules(profile: DataProfile) -> list[Finding]:
    findings: list[Finding] = []
    for r in CATALOG:
        if not matches(profile, r.applies_to):
            continue
        result = r.fn(profile)
        if result is not None:
            findings.append(result)
    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 99))
    return findings