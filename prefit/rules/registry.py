"""Rule registry and the emitter rules write into.

A rule is `fn(profile, ctx, out) -> None`. It reads the profile, and when its
trigger is met it emits Evidence, a Finding (which must cite evidence), and one or
more Decisions. Each rule declares a version and its thresholds; both are shown
in the report next to every decision the rule produced. Bump the version whenever
a threshold or the decision logic changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..enums import DecisionClass, DecisionStatus, Severity, Task, Trigger
from ..profiling.profile import DataProfile
from ..report.model import Decision, Evidence, Finding, Option, Question, Step


@dataclass
class RuleContext:
    answers: dict[str, str] = field(default_factory=dict)
    groups: str | None = None          # column name or "<array>"
    time_col: str | None = None
    has_estimator: bool = False
    estimator_params: set[str] = field(default_factory=set)
    n_splits: int = 5
    n_repeats: int = 2
    split_columns_as_features: bool = False
    external_holdout: dict | None = None


@dataclass
class Rule:
    id: str
    version: str
    fn: Callable[[DataProfile, RuleContext, Emitter], None]
    thresholds: dict[str, float]
    tasks: tuple[Task, ...] | None
    description: str


CATALOG: list[Rule] = []


def rule(id: str, version: str, thresholds: dict[str, float] | None = None,
         tasks: tuple[Task, ...] | None = None, description: str = ""):
    def deco(fn):
        CATALOG.append(Rule(id, version, fn, thresholds or {}, tasks, description))
        return fn
    return deco


def trigger(value: float | None, ci: tuple[float, float] | None, threshold: float) -> Trigger | None:
    """FIRED when value (and its whole CI) is at/above threshold, UNCERTAIN when the
    CI straddles the threshold, None when clearly below."""
    if value is None:
        return None
    if ci is None:
        return Trigger.FIRED if value >= threshold else None
    lo, hi = ci
    if lo >= threshold:
        return Trigger.FIRED
    if hi >= threshold:
        return Trigger.UNCERTAIN
    return None


def abs_ci(ci: tuple[float, float] | None) -> tuple[float, float] | None:
    if ci is None:
        return None
    lo, hi = ci
    if lo <= 0 <= hi:
        return (0.0, max(abs(lo), abs(hi)))
    a, b = sorted((abs(lo), abs(hi)))
    return (a, b)


class Emitter:
    """Collects what one rule produces, stamping rule id and version."""

    def __init__(self) -> None:
        self.evidence: dict[str, Evidence] = {}
        self.findings: list[Finding] = []
        self.decisions: list[Decision] = []
        self.questions: list[Question] = []
        self.rule: Rule | None = None

    def ev(self, id: str, label: str, value: Any, method: str,
           ci: tuple[float, float] | None = None, columns: list[str] | tuple[str, ...] = ()) -> str:
        eid = f"ev.{id}"
        self.evidence.setdefault(eid, Evidence(id=eid, label=label, value=value, ci=ci,
                                               method=method, columns=list(columns)))
        return eid

    def finding(self, id: str, severity: Severity, trig: Trigger, message: str,
                evidence_ids: list[str], columns: list[str] | tuple[str, ...] = ()) -> str:
        assert self.rule is not None
        fid = f"f.{id}"
        self.findings.append(Finding(
            id=fid, rule_id=self.rule.id, rule_version=self.rule.version,
            severity=severity if trig is Trigger.FIRED else Severity.INFO,
            trigger=trig, message=message, evidence_ids=evidence_ids,
            thresholds=dict(self.rule.thresholds), columns=list(columns)))
        return fid

    def decision(self, id: str, title: str, cls: DecisionClass, stage: str, action: str,
                 rationale: str, *, status: DecisionStatus | None = None, risk: str = "",
                 verify_hint: str = "", finding_ids: list[str] | tuple[str, ...] = (), evidence_ids: list[str] | tuple[str, ...] = (),
                 step: Step | None = None, question_id: str | None = None,
                 trig: Trigger | None = None, note: str = "", why: str = "", do: str = "",
                 standard: bool = False, alternatives: list[str] | tuple[str, ...] = ()) -> Decision:
        assert self.rule is not None
        if status is None:
            status = {
                DecisionClass.PROCEDURAL: DecisionStatus.APPLIED,
                DecisionClass.CONDITIONAL: (DecisionStatus.UNCERTAIN if trig is Trigger.UNCERTAIN
                                            else DecisionStatus.PROPOSED),
                DecisionClass.CONTEXTUAL: DecisionStatus.AWAITING_USER,
                DecisionClass.REVIEW: DecisionStatus.NEEDS_REVIEW,
            }[cls]
        d = Decision(id=f"d.{id}", title=title, cls=cls, status=status, stage=stage,
                     action=action, rationale=rationale, risk=risk, verify_hint=verify_hint,
                     why_short=why or first_sentence(rationale), do_short=do or first_sentence(action),
                     rule_id=self.rule.id, finding_ids=list(finding_ids),
                     evidence_ids=list(evidence_ids), step=step, question_id=question_id,
                     trigger=trig, note=note, standard=standard, alternatives=list(alternatives))
        self.decisions.append(d)
        return d

    def question(self, id: str, text: str, options: list[tuple],
                 decision_ids: list[str] | tuple[str, ...] = (), answer: str | None = None,
                 keyword: str = "", column: str | None = None) -> Question:
        """`options` are (value, label, effect) or (value, label, effect, {"aliases": [...], "flag": bool})."""
        opts = [Option(value=o[0], label=o[1], effect=o[2], **(o[3] if len(o) > 3 else {})) for o in options]
        q = Question(id=f"q.{id}", text=text, options=opts, decision_ids=list(decision_ids), answer=answer,
                     keyword=keyword or id, column=column)
        self.questions.append(q)
        return q


def contextual(out: Emitter, ctx: RuleContext, *, qid: str, question: str,
               options: list[tuple], confirm: str, decision_id: str,
               title: str, stage: str, action: str, rationale: str, keyword: str = "",
               column: str | None = None, **kw: Any) -> Decision:
    """Emit a contextual decision tied to a question; status follows the answer.

    `confirm` is the option value that confirms the decision's action."""
    full_qid = f"q.{qid}"
    answer = ctx.answers.get(full_qid)
    if answer is None:
        status = DecisionStatus.AWAITING_USER
    elif answer == confirm:
        status = DecisionStatus.CONFIRMED
    else:
        status = DecisionStatus.DECLINED
    d = out.decision(decision_id, title, DecisionClass.CONTEXTUAL, stage, action, rationale,
                     status=status, question_id=full_qid, **kw)
    out.question(qid, question, options, decision_ids=[d.id], answer=answer, keyword=keyword, column=column)
    return d


def first_sentence(text: str) -> str:
    """The first sentence of `text` (abbreviations like 'e.g.' are not split)."""
    import re

    parts = re.split(r"(?<![ .][a-z]\.)(?<=[.!?])\s+(?=[A-Z])", text.strip(), maxsplit=1)
    return parts[0]
