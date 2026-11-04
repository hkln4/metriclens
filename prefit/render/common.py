"""Formatting helpers shared by the text and HTML renderers."""

from __future__ import annotations

from typing import Any

from ..assemble import status_word
from ..enums import DecisionClass, DecisionStatus, Severity

GOOD = {DecisionStatus.APPLIED, DecisionStatus.CONFIRMED, DecisionStatus.VERIFIED, DecisionStatus.REVIEWED}
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
    critical_f = {f.id for f in report.details.findings if f.severity is Severity.CRITICAL}
    return {
        "applied": sum(d.status in GOOD for d in ds),
        "awaiting": sum(d.status is DecisionStatus.AWAITING_USER for d in ds),
        "review": sum(d.status is DecisionStatus.NEEDS_REVIEW for d in ds),
        "conditional": sum(d.cls is DecisionClass.CONDITIONAL for d in ds),
        "untested": sum(d.cls is DecisionClass.CONDITIONAL and d.status in UNTESTED for d in ds),
        "critical": len(critical_f),
        "critical_or_review": sum(d.status is DecisionStatus.NEEDS_REVIEW or bool(set(d.finding_ids) & critical_f)
                                  for d in ds),
        "questions": len(report.pending_questions()),
    }


UNTESTED = {DecisionStatus.PROPOSED, DecisionStatus.UNCERTAIN}
ATTENTION_STATUSES = {DecisionStatus.AWAITING_USER, DecisionStatus.NEEDS_REVIEW}


def attention(report: Any) -> list[Any]:
    """Decisions that need the user now: awaiting an answer, needing review, or
    conditional decisions an experiment settled (verified / refuted). Most severe first."""
    sev_rank = {Severity.CRITICAL: 0, Severity.WARN: 1, Severity.INFO: 2}
    fsev = {f.id: sev_rank[f.severity] for f in report.details.findings}
    keep = [d for d in report.decisions
            if d.status in ATTENTION_STATUSES
            or (d.cls is DecisionClass.CONDITIONAL and d.status in (DecisionStatus.VERIFIED, DecisionStatus.REFUTED))]
    return sorted(keep, key=lambda d: (min((fsev.get(f, 3) for f in d.finding_ids), default=3),
                                       d.status is not DecisionStatus.AWAITING_USER))


def next_step(report: Any) -> str:
    c = summary_counts(report)
    if c["questions"]:
        return (f"Answer {c['questions']} open question(s) with report.answer({{...}}); contextual decisions stay out "
                "of the model recipe until you do.")
    if c["untested"]:
        return (f"Run report.verify(df) to test {c['untested']} conditional decision(s) with your estimator; "
                "untested ones are not exported.")
    if c["review"]:
        return f"Inspect {c['review']} review item(s), then export the model recipe with report.export(estimator)."
    return "Export the model recipe with report.export(estimator) and evaluate it once on the held-out data."


STATE_LABEL = {"done": "done", "needs_you": "needs you", "tested": "tested"}
STATE_ICON = {"done": "✓", "needs_you": "!", "tested": "⚗"}


def states(report: Any) -> dict[str, list[Any]]:
    """Decisions grouped by the three states the user sees (standard practice counts as done)."""
    out: dict[str, list[Any]] = {"needs_you": [], "tested": [], "done": []}
    for d in report.decisions:
        out[d.state].append(d)
    return out


def state_line(report: Any) -> str:
    g = states(report)
    return " · ".join(f"{STATE_ICON[k]} {len(v)} {STATE_LABEL[k]}" for k, v in g.items())


def _literal(v: Any) -> str:
    return repr(v) if not isinstance(v, str) else '"' + v.replace('"', '\\"') + '"'


def answer_line(q: Any) -> str:
    """The full line that answers question q, its first option filled in, the others in a comment."""
    def short(o: Any) -> str:
        return o.aliases[0] if o.aliases else o.value

    first = q.options[0]
    kw = q.keyword or q.id[2:]
    if q.column is not None and first.flag is not None:
        call = f"report = report.answer({kw}={{{_literal(q.column)}: {first.flag}}})"
        alts = " | ".join(f"{o.flag}: {o.label}" for o in q.options)
    else:
        call = f"report = report.answer({kw}={_literal(short(first))})"
        alts = "or " + ", ".join(_literal(short(o)) for o in q.options[1:])
    return f"{call}  # {alts}"


def next_lines(report: Any) -> list[str]:
    """One runnable line per open item, then the remaining steps of the flow."""
    has_data = getattr(report, "_data", None) is not None
    df_arg = "" if has_data else "df"
    lines = [answer_line(q) for q in report.pending_questions()]
    for d in report.decisions:
        if d.cls is DecisionClass.REVIEW and d.status is DecisionStatus.NEEDS_REVIEW:
            lines.append(f'report = report.resolve("{d.id}", "what you found and decided")  # {d.title}')
    if any(d.cls is DecisionClass.CONDITIONAL and d.status in UNTESTED for d in report.decisions):
        lines.append(f"report = report.verify({df_arg})  # test the conditional decisions with your model")
    est = "" if getattr(report, "_estimator", None) is not None else "estimator"
    lines.append(f"fitted = report.export({est}).fit({df_arg})  # the recipe: applied, confirmed and verified steps")
    if report.details.profile.holdout.get("used"):
        lines.append("review = report.review(fitted)  # once, on the internal holdout")
    else:
        lines.append("review = report.review(fitted, test_df)  # once, on your untouched test data")
    return lines


def key_evidence(report: Any, d: Any) -> str:
    """The decisive measurement behind a decision, as 'label = value [ci]'."""
    for eid in d.evidence_ids:
        e = report.details.evidence.get(eid)
        if e is not None:
            ci = f" [{fmt(e.ci[0])}, {fmt(e.ci[1])}]" if e.ci else ""
            return f"{e.label} = {fmt(e.value)}{ci}"
    return ""


def effect(e: Any) -> tuple[float, tuple[float, float] | None, str]:
    """An experiment's change in natural metric units: (change, ci, direction).
    Error metrics (sklearn's neg_*) are shown as errors, where lower is better."""
    if e.metric.startswith("neg_"):
        ci = (-e.ci[1], -e.ci[0]) if e.ci else None
        return -e.mean_diff, ci, "lower = better"
    return e.mean_diff, (tuple(e.ci) if e.ci else None), "higher = better"


def experiment_summary(report: Any, d: Any) -> str:
    from ..rules.metrics import label

    exps = [e for e in report.experiments if e.decision_id == d.id]
    parts = []
    for e in exps:
        p = f", Holm p={fmt(e.p_adjusted)}" if e.p_adjusted is not None else ", no p-value (time-ordered folds)"
        if e.p_equivalence is not None:
            p += f", equivalence p={fmt(e.p_equivalence)}"
        change, _, direction = effect(e)
        parts.append(f"{e.family} ({e.estimator}): {label(e.metric)} {change:+.4g} ({direction}), "
                     f"{e.folds_improved}/{len(e.baseline_scores)} folds better{p} -> {e.outcome}")
    return "; ".join(parts)


def task_unconfirmed(report: Any) -> bool:
    return report.details.profile.task_needs_answer and report.details.context.answers.get("q.task") is None


def task_dependence_note(report: Any, d: Any) -> str:
    if not (d.task_dependent and task_unconfirmed(report)):
        return ""
    return f"Assumes the inferred task ({report.task.value}); it changes if you answer q.task otherwise."


def partition_note(report: Any) -> str:
    p = report.details.profile
    h = p.holdout
    ext = report.details.manifest.external_holdout
    dev = f"{p.dataset.n_rows:,} development rows"
    if p.sampling.get("sampled"):
        dev += f" (a seeded sample of {p.sampling['of_rows']:,})"
    base = f"Every statistic, figure, decision and experiment in this report comes from {dev}."
    if h.get("used"):
        dup = h.get("duplicate_groups_kept_together")
        dup_txt = (f" {dup:,} set(s) of exact duplicate rows were kept whole, so no copy sits on both sides."
                   if dup else "")
        return (f"{base} The internal holdout ({h['n_holdout']:,} rows, {h['method']}) was set aside before "
                f"profiling and was never read; get it with report.holdout(df) and use it once, at the end.{dup_txt}")
    if ext:
        ov = ext.get("overlap_rows")
        ov_txt = "" if ov is None else (f" Exact row overlap with the development data: {ov} row(s)." if ov
                                        else " No exact row overlap with the development data.")
        return (f"{base} Prefit did not reserve an internal holdout; the {ext['name']} ({ext['n_rows']:,} rows) "
                f"was registered (metadata only, never profiled).{ov_txt}")
    return (f"{base} Prefit did not reserve an internal holdout. If you split off a test set before analyze, "
            "register it with report.register_external_holdout(test_df); otherwise estimate final performance "
            "with nested cross-validation.")


def task_note(report: Any) -> str:
    p = report.details.profile
    if not p.task_inferred:
        return ""
    if not p.task_needs_answer:
        return f" (from the target: {p.target.task_evidence.get('reason', 'inferred')})"
    return " (inferred, confirmed)" if report.details.context.answers.get("q.task") else " (inferred)"


def headline(report: Any) -> str:
    p = report.details.profile
    c = summary_counts(report)
    return (f"{report.task.value}{task_note(report)} on '{report.target}' · "
            f"{p.dataset.n_rows:,} development rows, {p.dataset.n_features} features · "
            f"{c['questions']} question(s) open, {c['critical_or_review']} critical/review decision(s).")


__all__ = ["states", "state_line", "answer_line", "next_lines", "STATE_LABEL", "STATE_ICON", "effect", "task_unconfirmed", "task_dependence_note", "partition_note", "fmt", "fmt_ci", "tone", "ICON", "SEV_TONE", "CLASS_LABEL", "CLASS_HELP",
           "summary_counts", "attention", "next_step", "key_evidence", "experiment_summary", "headline",
           "status_word"]
