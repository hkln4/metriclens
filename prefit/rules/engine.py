"""DataProfile + context -> evidence, findings, decisions, questions, metric plan.

Deterministic: the same profile and answers always produce the same output, in
the same order. Re-running it is how `Report.answer()` applies new answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import Severity
from ..profiling.profile import DataProfile
from ..report.model import Decision, Evidence, Finding, MetricPlan, Question
from . import catalog  # noqa: F401  (importing registers every rule)
from .metrics import metric_policy
from .registry import CATALOG, Emitter, Rule, RuleContext

_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}


@dataclass
class RuleOutput:
    evidence: dict[str, Evidence]
    findings: list[Finding]
    decisions: list[Decision]
    questions: list[Question]
    metric_plan: MetricPlan


def rule_versions() -> dict[str, str]:
    return {r.id: r.version for r in CATALOG} | {"metric_policy": "1"}


def rules() -> list[Rule]:
    return list(CATALOG)


def run_rules(profile: DataProfile, ctx: RuleContext) -> RuleOutput:
    out = Emitter()
    for r in CATALOG:
        if r.tasks is not None and profile.task not in r.tasks:
            continue
        out.rule = r
        r.fn(profile, ctx, out)
    out.rule = Rule("metric_policy", "1", lambda *a: None, {}, None, "")
    plan = metric_policy(profile, ctx, out, {f.id for f in out.findings})

    # critical first; sorted() is stable, so catalog order breaks ties
    findings = sorted(out.findings, key=lambda f: _SEVERITY_ORDER[f.severity])
    return RuleOutput(out.evidence, findings, out.decisions, out.questions, plan)
