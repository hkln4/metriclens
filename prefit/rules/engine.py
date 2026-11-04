"""DataProfile + context -> evidence, findings, decisions, questions, metric plan.

Deterministic: the same profile and answers always produce the same output, in
the same order. Re-running it is how `Report.answer()` applies new answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..enums import Severity
from ..frozen import evolve
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
    return {r.id: r.version for r in CATALOG} | {"metric_policy": "2"}


def rules() -> list[Rule]:
    return list(CATALOG)


def run_rules(profile: DataProfile, ctx: RuleContext) -> RuleOutput:
    out = Emitter()
    task_specific: tuple[set[str], set[str]] = (set(), set())  # finding ids, decision ids
    for r in CATALOG:
        if r.tasks is not None and profile.task not in r.tasks:
            continue
        out.rule = r
        nf, nd = len(out.findings), len(out.decisions)
        r.fn(profile, ctx, out)
        if r.tasks is not None:
            task_specific[0].update(f.id for f in out.findings[nf:])
            task_specific[1].update(d.id for d in out.decisions[nd:])
    out.rule = Rule("metric_policy", "2", lambda *a: None, {}, None, "")
    nd = len(out.decisions)
    plan = metric_policy(profile, ctx, out, {f.id for f in out.findings})
    task_specific[1].update(d.id for d in out.decisions[nd:])  # metric options follow the task
    out.findings = [evolve(f, task_dependent=True) if f.id in task_specific[0] else f for f in out.findings]
    out.decisions = [evolve(d, task_dependent=True) if d.id in task_specific[1] else d for d in out.decisions]

    # critical first; sorted() is stable, so catalog order breaks ties
    findings = sorted(out.findings, key=lambda f: _SEVERITY_ORDER[f.severity])
    return RuleOutput(out.evidence, findings, out.decisions, out.questions, plan)
