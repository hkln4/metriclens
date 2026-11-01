"""Public entry points and orchestration.

    validate -> reserve holdout -> profile (dev rows) -> rules -> figures
             -> [verify: bounded experiments] -> [llm: optional interpretation]
             -> Report

Everything up to and including the figures is deterministic for a given seed.
"""

from __future__ import annotations

import platform
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .assemble import build_limitations, build_roadmap, load_track_record
from .config import Budget, ProfileOptions
from .enums import DecisionClass, DecisionStatus, Task
from .figures import _link, build_figures, experiment_figure
from .guard import reserve_holdout, sample_rows, validate_inputs
from .profiling.tabular import profile_tabular
from .profiling.task_inference import infer_task
from .report.model import ContextState, Manifest, Report
from .rules.engine import RuleOutput, rule_versions, run_rules
from .rules.registry import RuleContext


def analyze(
    df: pd.DataFrame,
    target: str,
    *,
    task: Task | str | None = None,
    estimator: Any = None,
    groups: str | Any = None,
    time_col: str | None = None,
    holdout: float | None = 0.2,
    context: str | None = None,
    answers: dict[str, str] | None = None,
    verify: bool = False,
    budget: Budget | None = None,
    llm: Any = None,
    seed: int = 0,
    profile_options: ProfileOptions | None = None,
) -> Report:
    """Profile a dataset and produce an evidence-backed decision roadmap.

    Parameters
    ----------
    df : the full dataset as a pandas DataFrame.
    target : name of the target column.
    task : "classification" / "regression"; inferred (and flagged for
        confirmation) when omitted.
    estimator : your scikit-learn estimator. Experiments use it, and only it can
        *verify* a conditional decision. Without it, proxy models give a
        sensitivity analysis.
    groups : column name (or array) identifying entities whose rows must stay on
        one side of every split. Group-aware splitting is only used when given.
    time_col : column defining time order; switches to a time-ordered holdout and
        TimeSeriesSplit.
    holdout : fraction reserved before anything is computed, or None.
    context : free-text description of the application (used by the LLM layer).
    answers : answers to report questions, e.g. {"metric_priority": "miss_costly"}.
    verify : run the bounded experiments for conditional decisions (needs the
        `verify` extra: scikit-learn + scipy).
    budget : experiment limits (see `metriclens.Budget`).
    llm : optional `metriclens.LLMConfig` (or a provider name / Provider). Off by
        default; see `Report.llm_payload()` for exactly what would be sent.
    """
    options = profile_options or ProfileOptions()
    budget = budget or Budget()
    groups_s = validate_inputs(df, target, groups, time_col)

    if task is None:
        task_t, task_ev = infer_task(df[target])
        inferred = True
    else:
        task_t = Task(task)
        task_ev = {"reason": "given by the user", "dtype": str(df[target].dtype),
                   "n_unique": int(df[target].nunique())}
        inferred = False
    if task_t is Task.UNKNOWN:
        raise ValueError("Could not infer the task from the target; pass task='classification' or 'regression'.")

    dev_pos, hold_pos, hold_info = reserve_holdout(df, target, task_t, holdout, groups_s, time_col, seed)
    dev = df.iloc[dev_pos]
    roles: dict[str, str] = {}
    if isinstance(groups, str):
        roles[groups] = "groups"
    if time_col:
        roles[time_col] = "time_col"
    strat = dev[target].astype(str) if task_t is Task.CLASSIFICATION else None
    dev_s, sampling = sample_rows(dev, options.max_rows, seed, strat)
    if sampling.get("sampled"):
        sampling["of_rows"] = len(dev)

    profile = profile_tabular(dev_s, target, task_t, inferred, task_ev, roles, options, seed,
                              n_rows_input=len(df), sampling=sampling, holdout=hold_info)
    ctx_state = ContextState(
        text=context, answers=_normalize_answers(answers or {}),
        groups=groups if isinstance(groups, str) else ("<array>" if groups is not None else None),
        time_col=time_col, estimator=_describe(estimator))
    out = run_rules(profile, _rule_context(ctx_state, estimator, budget))
    _validate_answers(ctx_state.answers, out)
    figures = build_figures(dev_s, profile, out.evidence, out.decisions)

    manifest = _manifest(seed, options, None)
    manifest.history = [{"revision": 0, "action": "analyze", "target": target, "task": task_t.value,
                         "task_inferred": inferred, "answers": dict(ctx_state.answers),
                         "groups": ctx_state.groups, "time_col": time_col, "holdout": holdout,
                         "estimator": ctx_state.estimator}]
    report = _make_report(profile, ctx_state, out, figures, experiments=[], baseline={}, manifest=manifest)
    report._holdout_positions = hold_pos
    report._estimator = estimator
    if verify:
        report = _verify(report, df, estimator=estimator, budget=budget, groups=groups_s, seed=seed)
    if llm is not None:
        report = report.interpret(llm)
    return report


def review(y_true: Any, y_pred: Any, report: Report | None = None, *, y_score: Any = None,
           segments: Any = None, task: Task | str | None = None) -> Any:
    """Post-training review against the roadmap's assumptions (see `metriclens.review`)."""
    from .posttrain import review as _review

    return _review(y_true, y_pred, report=report, y_score=y_score, segments=segments, task=task)


# --------------------------------------------------------------------------- #
# internals shared by Report.answer / Report.verify
# --------------------------------------------------------------------------- #

def _describe(estimator: Any) -> str | None:
    if estimator is None:
        return None
    return type(estimator).__name__


def _rule_context(ctx: ContextState, estimator: Any, budget: Budget) -> RuleContext:
    params: set[str] = set()
    if estimator is not None and hasattr(estimator, "get_params"):
        params = set(estimator.get_params())
    return RuleContext(answers=dict(ctx.answers), groups=ctx.groups, time_col=ctx.time_col,
                       has_estimator=estimator is not None, estimator_params=params, n_splits=budget.n_splits)


def _normalize_answers(answers: dict[str, str]) -> dict[str, str]:
    return {(k if k.startswith("q.") else f"q.{k}"): str(v) for k, v in answers.items()}


def _validate_answers(answers: dict[str, str], out: RuleOutput) -> None:
    qs = {q.id: q for q in out.questions}
    for k, v in answers.items():
        if k not in qs:
            raise KeyError(f"Unknown question '{k}'. Open questions: {sorted(qs)}")
        allowed = [o.value for o in qs[k].options]
        if v not in allowed:
            raise ValueError(f"Answer '{v}' is not an option for '{k}'. Options: {allowed}")


def _packages() -> dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "pydantic", "scikit-learn", "scipy", "plotly"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def _manifest(seed: int, options: ProfileOptions, budget: Budget | None) -> Manifest:
    tr = load_track_record()
    return Manifest(metriclens_version=__version__, python=platform.python_version(), packages=_packages(),
                    seed=seed, rule_versions=rule_versions(), profile_options=options.model_dump(),
                    budget=budget.model_dump() if budget else None, benchmark=tr.get("manifest_version"))


def _make_report(profile, ctx_state, out: RuleOutput, figures, experiments, baseline, manifest,
                 llm=None) -> Report:
    has_conditional = any(d.cls is DecisionClass.CONDITIONAL for d in out.decisions)
    tr = load_track_record()
    return Report(
        target=profile.target.name, task=profile.task, profile=profile, evidence=out.evidence,
        findings=out.findings, decisions=out.decisions, questions=out.questions, metric_plan=out.metric_plan,
        roadmap=build_roadmap(out.decisions), figures=figures, experiments=experiments, baseline=baseline,
        llm=llm, context=ctx_state, manifest=manifest,
        limitations=build_limitations({"sampling": profile.sampling, "experiments": experiments,
                                       "has_conditional": has_conditional}),
        rule_track_record={k: v for k, v in tr.get("rules", {}).items()
                           if k in {f.rule_id for f in out.findings}},
    )


def _next_revision(manifest: Manifest, action: str, **details: Any) -> Manifest:
    """A new manifest whose history records this step; earlier reports keep theirs."""
    rev = manifest.revision + 1
    entry = {"revision": rev, "action": action, **details}
    return manifest.model_copy(update={"revision": rev, "history": [*manifest.history, entry]})


def _carry(report: Report, new: Report) -> Report:
    new._holdout_positions = report._holdout_positions
    new._estimator = report._estimator
    return new


def _apply_experiments(out: RuleOutput, experiments, notes: dict[str, str], prev: dict[str, Any]) -> None:
    from .verify import status_from_experiments

    by_dec: dict[str, list] = {}
    for e in experiments:
        by_dec.setdefault(e.decision_id, []).append(e)
    for d in out.decisions:
        if d.cls is not DecisionClass.CONDITIONAL:
            continue
        if d.id in by_dec:
            exps = by_dec[d.id]
            status, note = status_from_experiments(d, exps, user=any(e.family == "user" for e in exps))
            if d.trigger is not None and d.trigger.value == "uncertain":
                note = (note + " " if note else "") + "The trigger itself was near its threshold."
            d.status, d.note = status, note
            d.experiment_ids = [e.id for e in exps]
            if notes.get(d.id) == "skipped_budget":
                d.note += " The budget ran out before every model family was tested."
            if exps[0].metric != out.metric_plan.dev_metric:
                d.note += (f" Experiments used {exps[0].metric}; the metric is now {out.metric_plan.dev_metric}: "
                           "re-run report.verify(df).")
        elif d.id in notes:
            n = notes[d.id]
            if n == "skipped_budget":
                d.status = DecisionStatus.SKIPPED_BUDGET
            elif n == "not_applicable":
                d.status, d.note = DecisionStatus.NOT_APPLICABLE, "The estimator has no parameter for this change."
            else:
                d.note = n
        elif d.id in prev:
            d.status, d.note = prev[d.id]


def _apply_answers(report: Report, answers: dict[str, str]) -> Report:
    answers = _normalize_answers(answers)
    q_task = answers.get("q.task")
    if q_task is not None and q_task != report.task.value:
        raise ValueError(f"You answered that the task is {q_task}: re-run ml.analyze(df, target, task='{q_task}').")
    ctx_state = report.context.model_copy(update={"answers": {**report.context.answers, **answers}})
    out = run_rules(report.profile, _rule_context(ctx_state, report._estimator, Budget(**(report.manifest.budget or {}))))
    _validate_answers(ctx_state.answers, out)
    prev = {d.id: (d.status, d.note) for d in report.decisions
            if d.cls is DecisionClass.CONDITIONAL and d.status in
            (DecisionStatus.SKIPPED_BUDGET, DecisionStatus.NOT_APPLICABLE)}
    _apply_experiments(out, report.experiments, {}, prev)
    figures = [f.model_copy(deep=True) for f in report.figures]
    _link(figures, out.evidence, out.decisions)
    manifest = _next_revision(report.manifest, "answer", answers=answers)
    new = _make_report(report.profile, ctx_state, out, figures, report.experiments, report.baseline,
                       manifest, llm=report.llm)
    return _carry(report, new)


def _verify(report: Report, df: pd.DataFrame, estimator: Any = None, budget: Budget | None = None,
            groups: Any = None, seed: int | None = None) -> Report:
    from .verify import run_verification

    budget = budget or Budget()
    seed = report.manifest.seed if seed is None else seed
    estimator = estimator if estimator is not None else report._estimator
    if report._holdout_positions is not None:
        dev, _ = report.split(df)
        dev_pos = np.setdiff1d(np.arange(len(df)), np.asarray(report._holdout_positions, dtype=int))
    else:
        import warnings

        warnings.warn("No holdout information on this report; treating `df` as development data.", stacklevel=3)
        dev, dev_pos = df, np.arange(len(df))
    g: Any = None
    if report.context.groups and report.context.groups != "<array>":
        g = report.context.groups
    elif report.context.groups == "<array>":
        if groups is None:
            raise ValueError("This report was built with a groups array; pass the same array: report.verify(df, groups=...).")
        g = np.asarray(groups).ravel()[dev_pos] if len(np.asarray(groups).ravel()) == len(df) else groups

    experiments, baseline, notes = run_verification(report, dev, estimator, budget, g, seed)
    ctx_state = report.context.model_copy(update={"estimator": _describe(estimator)})
    out = run_rules(report.profile, _rule_context(ctx_state, estimator, budget))
    _apply_experiments(out, experiments, notes, {})
    figures = [f.model_copy(deep=True) for f in report.figures if not f.id.startswith("fig.exp.")]
    for d in out.decisions:
        exps = [e for e in experiments if e.decision_id == d.id]
        if exps:
            figures.append(experiment_figure(d, exps))
    _link(figures, out.evidence, out.decisions)
    figures.sort(key=lambda f: (f.priority, f.id))
    manifest = _next_revision(report.manifest.model_copy(update={"budget": budget.model_dump()}), "verify",
                              estimator=_describe(estimator) or "proxies (linear + tree)",
                              experiments=len(experiments), budget=budget.model_dump(),
                              protocol=experiments[0].protocol if experiments else None)
    new = _make_report(report.profile, ctx_state, out, figures, experiments, baseline, manifest, llm=report.llm)
    new = _carry(report, new)
    new._estimator = estimator
    return new
