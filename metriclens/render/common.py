"""Formatting helpers shared by the text and HTML renderers."""

from __future__ import annotations

from typing import Any

from ..assemble import status_word
from ..enums import DecisionClass, DecisionStatus, Severity

GOOD = {DecisionStatus.APPLIED, DecisionStatus.CONFIRMED, DecisionStatus.VERIFIED}
ATTENTION = {DecisionStatus.AWAITING_USER, DecisionStatus.NEEDS_REVIEW, DecisionStatus.UNCERTAIN}
BAD = {DecisionStatus.REFUTED}


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3g}"
        return f"{v:.{nd}g}"
    if isinstance(v, (list, tuple)):
        if len(v) > 8:
            return ", ".join(fmt(x, nd) for x in v[:8]) + f", … (+{len(v) - 8})"
        return ", ".join(fmt(x, nd) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}={fmt(x, nd)}" for k, x in v.items())
    return str(v)


def fmt_ci(ci: Any) -> str:
    if not ci:
        return ""
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def tone(status: DecisionStatus) -> str:
    if status in GOOD:
        return "good"
    if status in BAD:
        return "critical"
    if status in ATTENTION:
        return "warning"
    return "neutral"


ICON = {"good": "✓", "warning": "!", "critical": "✕", "neutral": "·", "serious": "▲"}
SEV_TONE = {Severity.CRITICAL: "critical", Severity.WARN: "serious", Severity.INFO: "neutral"}

CLASS_LABEL = {
    DecisionClass.PROCEDURAL: "procedural",
    DecisionClass.CONDITIONAL: "conditional",
    DecisionClass.CONTEXTUAL: "needs your context",
    DecisionClass.REVIEW: "review",
}

CLASS_HELP = {
    DecisionClass.PROCEDURAL: "About how to evaluate; correct regardless of the data. Applied.",
    DecisionClass.CONDITIONAL: "May help or hurt depending on data and model; only an experiment can support it.",
    DecisionClass.CONTEXTUAL: "Cannot be decided from data; applied only after you answer.",
    DecisionClass.REVIEW: "Needs human inspection; no automatic action.",
}


def summary_counts(report: Any) -> dict[str, int]:
    ds = report.decisions
    return {
        "applied": sum(d.status in GOOD for d in ds),
        "awaiting": sum(d.status is DecisionStatus.AWAITING_USER for d in ds),
        "review": sum(d.status is DecisionStatus.NEEDS_REVIEW for d in ds),
        "conditional": sum(d.cls is DecisionClass.CONDITIONAL for d in ds),
        "critical": sum(f.severity is Severity.CRITICAL for f in report.findings),
    }


def priorities(report: Any, k: int = 6) -> list[Any]:
    """Decisions that need the user's attention, most important first."""
    sev_rank = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}
    fsev = {f.id: sev_rank[f.severity] for f in report.findings}

    def key(d):
        s = min((fsev.get(f, 3) for f in d.finding_ids), default=3)
        attention = 0 if d.status in ATTENTION or d.status in BAD else 1
        return (attention, s)

    cand = [d for d in report.decisions if d.status not in (DecisionStatus.APPLIED,)
            and not (d.id == "d.task" and d.status is DecisionStatus.CONFIRMED)]
    return sorted(cand, key=key)[:k]


def headline(report: Any) -> str:
    c = summary_counts(report)
    p = report.profile
    parts = [f"{p.dataset.n_rows:,} development rows, {p.dataset.n_features} features, "
             f"{report.task.value}{' (inferred)' if p.task_inferred else ''}."]
    parts.append(f"{len(report.decisions)} decisions: {c['applied']} applied/confirmed, "
                 f"{c['awaiting']} awaiting your answer, {c['review']} to review, "
                 f"{c['conditional']} conditional.")
    if c["critical"]:
        parts.append(f"{c['critical']} critical finding(s).")
    return " ".join(parts)


__all__ = ["fmt", "fmt_ci", "tone", "ICON", "SEV_TONE", "CLASS_LABEL", "CLASS_HELP", "summary_counts",
           "priorities", "headline", "status_word"]
