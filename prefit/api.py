"""Public entry points and orchestration.

    pf.split(df, ...)            -> (dev_df, test_df)      data preparation, explicit
    pf.analyze(dev_df, ...)      -> Report                 profile -> rules -> figures
    report.answer / verify / interpret / register_external_holdout -> new Report revision
    report.export(...)           -> model recipe
    pf.review(...)               -> post-training check

`analyze` never splits unless asked (`holdout=0.2`); then the holdout is set
aside before anything is computed and can be re-derived with `report.holdout(df)`.
Everything up to and including the figures is deterministic for a given seed.
"""

from __future__ import annotations

import platform
import warnings
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd

from ._version import __version__
from .assemble import build_limitations, build_roadmap, load_track_record
from .config import Budget, ProfileOptions
from .enums import DecisionClass, DecisionStatus, Task
from .figures import _link, build_figures, experiment_figure
from .fingerprint import check as check_fingerprint
from .fingerprint import fingerprint, row_hashes
from .frozen import evolve
from .guard import reserve_holdout, sample_rows, validate_inputs
from .profiling.tabular import profile_tabular
from .profiling.task_inference import infer_task
from .report.model import ContextState, Manifest, Report, ReportDetails
from .rules.engine import RuleOutput, rule_versions, run_rules
from .rules.registry import RuleContext

SPLIT_STRATEGIES = ("stratified", "random", "group", "time")


def split(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    strategy: str = "stratified",
    groups: str | None = None,
    time_col: str | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (dev_df, test_df) before any analysis.

    strategy
        "stratified"  keep class ratios (classification targets only)
        "random"      plain random rows
        "group"       whole groups (needs `groups=` column) on one side
        "time"        the latest `test_size` share by `time_col` becomes the test set
    Pass dev_df to `pf.analyze` and keep test_df untouched until the end;
    `report.register_external_holdout(test_df, dev_df=dev_df)` records it.
    """
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"strategy must be one of {SPLIT_STRATEGIES}, got {strategy!r}.")
    if strategy == "group" and groups is None:
        raise ValueError("strategy='group' needs groups=<column name>.")
    if strategy == "time" and time_col is None:
        raise ValueError("strategy='time' needs time_col=<column name>.")
    if strategy in ("stratified", "random") and (groups or time_col):
        raise ValueError(f"strategy={strategy!r} does not use groups/time_col; use strategy='group' or 'time'.")
    g = validate_inputs(df, target, groups, time_col)
    if strategy == "stratified":
        y = df[target].dropna()
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 50:
            raise ValueError("strategy='stratified' needs a discrete target; use strategy='random' for a "
                             "continuous target.")
        task = Task.CLASSIFICATION
    else:
        task = Task.REGRESSION  # no stratification
    dev, test, info = reserve_holdout(df, target, task, test_size, g, time_col, random_state)
    if not info["used"]:
        raise ValueError(f"Cannot split: {info['reason']}")
    return df.iloc[dev], df.iloc[test]


def analyze(
    df: pd.DataFrame,
    target: str,
    *,
    task: Task | str | None = None,
    estimator: Any = None,
    groups: str | Any = None,
    time_col: str | None = None,
    holdout: float | None = None,
    split_columns_as_features: bool = False,
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
    df : the development data (typically dev_df from `pf.split`).
    target : name of the target column.
    task : "classification" / "regression"; inferred (and flagged for
        confirmation) when omitted.
    estimator : your scikit-learn estimator. Experiments use it, and only it can
        *verify* a conditional decision. Without it, proxy models give a
        sensitivity analysis.
    groups : column name (or array) identifying entities whose rows must stay on
        one side of every split. Group-aware splitting is only used when given.
    time_col : column defining time order; switches to TimeSeriesSplit.
    holdout : None (default) analyzes all of `df`. A fraction (e.g. 0.2) asks
        prefit to set that share aside *before* profiling; get it back with
        `report.holdout(df)`.
    split_columns_as_features : by default the groups / time_col columns are
        kept out of the model features; True includes them (with a warning).
    context : free-text description of the application (used by the LLM layer).
    answers : answers to report questions, e.g. {"metric_priority": "miss_costly"}.
    verify : run the bounded experiments for conditional decisions (needs the
        `verify` extra: scikit-learn + scipy).
    budget : experiment limits (see `prefit.Budget`).
    llm : optional `prefit.LLMConfig` (or a provider name / Provider). Off by
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
    if not split_columns_as_features:
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
        time_col=time_col, estimator=_describe(estimator), split_columns_as_features=split_columns_as_features)
    out = run_rules(profile, _rule_context(ctx_state, estimator, budget, None))
    _validate_answers(ctx_state.answers, out)
    figures = build_figures(dev_s, profile, out.evidence, out.decisions)

    tr = load_track_record()
    manifest = Manifest(
        prefit_version=__version__, python=platform.python_version(), packages=_packages(), seed=seed,
        rule_versions=rule_versions(), profile_options=options.model_dump(), budget=budget.model_dump(),
        benchmark=tr.get("manifest_version"), source_fingerprint=fingerprint(df, target),
        history=[{"revision": 0, "action": "analyze", "target": target, "task": task_t.value,
                  "task_inferred": inferred, "answers": dict(ctx_state.answers), "groups": ctx_state.groups,
                  "time_col": time_col, "holdout": holdout, "estimator": ctx_state.estimator}])
    report = _make_report(profile, ctx_state, out, figures, (), {}, manifest)
    report._holdout_positions = hold_pos
    report._estimator = estimator
    report._row_hashes = row_hashes(df)
    if verify:
        report = _verify(report, df, estimator=estimator, budget=budget, groups=groups)
    if llm is not None:
        report = report.interpret(llm)
    return report


def load_report(source: str) -> Report:
    """Load a report saved with `report.to_json(path)` (a path or a JSON string).

    Holdout rows and the estimator are not stored; `report.holdout(df)` and
    `report.verify(df)` re-derive what they need from `df` after checking its
    fingerprint."""
    text = source
    if not source.lstrip().startswith("{"):
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
    return Report.model_validate_json(text)


def review(y_true: Any, y_pred: Any, report: Report | None = None, *, y_score: Any = None,
           segments: Any = None, task: Task | str | None = None) -> Any:
    """Post-training review against the roadmap's assumptions (see `prefit.posttrain`)."""
    from .posttrain import review as _review

    return _review(y_true, y_pred, report=report, y_score=y_score, segments=segments, task=task)


# --------------------------------------------------------------------------- #
# internals shared by the Report flow methods
# --------------------------------------------------------------------------- #

def _describe(estimator: Any) -> str | None:
    return None if estimator is None else type(estimator).__name__


def _budget_of(report: Report) -> Budget:
    return Budget(**(report.details.manifest.budget or {}))


def _rule_context(ctx: ContextState, estimator: Any, budget: Budget, external: dict | None) -> RuleContext:
    params: set[str] = set()
    if estimator is not None and hasattr(estimator, "get_params"):
        params = set(estimator.get_params())
    return RuleContext(answers=dict(ctx.answers), groups=ctx.groups, time_col=ctx.time_col,
                       has_estimator=estimator is not None, estimator_params=params, n_splits=budget.n_splits,
                       n_repeats=budget.n_repeats, split_columns_as_features=ctx.split_columns_as_features,
                       external_holdout=dict(external) if external else None)


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


def _make_report(profile, ctx_state, out: RuleOutput, figures, experiments, baseline, manifest,
                 llm=None) -> Report:
    has_conditional = any(d.cls is DecisionClass.CONDITIONAL for d in out.decisions)
    tr = load_track_record()
    details = ReportDetails(
        findings=out.findings, evidence=out.evidence, profile=profile, figures=figures, manifest=manifest,
        context=ctx_state, llm=llm,
        limitations=build_limitations({"sampling": profile.sampling, "experiments": experiments,
                                       "has_conditional": has_conditional}),
        rule_track_record={k: v for k, v in tr.get("rules", {}).items()
                           if k in {f.rule_id for f in out.findings}})
    return Report(target=profile.target.name, task=profile.task, decisions=out.decisions,
                  questions=out.questions, metric_plan=out.metric_plan, roadmap=build_roadmap(out.decisions),
                  experiments=experiments, baseline=baseline, details=details)


def _next_revision(manifest: Manifest, action: str, **details: Any) -> Manifest:
    """A new manifest whose history records this step; earlier reports keep theirs."""
    rev = manifest.revision + 1
    return evolve(manifest, revision=rev, history=[*manifest.history, {"revision": rev, "action": action, **details}])


def _carry(report: Report, new: Report, estimator: Any = None) -> Report:
    new._holdout_positions = report._holdout_positions
    new._estimator = report._estimator if estimator is None else estimator
    new._row_hashes = report._row_hashes
    return new


def _apply_experiments(out: RuleOutput, experiments, notes: dict[str, str], prev: dict[str, Any]) -> None:
    from .verify import status_from_experiments

    by_dec: dict[str, list] = {}
    for e in experiments:
        by_dec.setdefault(e.decision_id, []).append(e)
    for i, d in enumerate(out.decisions):
        if d.cls is not DecisionClass.CONDITIONAL:
            continue
        if d.id in by_dec:
            exps = by_dec[d.id]
            status, note = status_from_experiments(d, exps, user=any(e.family == "user" for e in exps))
            if d.trigger is not None and d.trigger.value == "uncertain":
                note = (note + " " if note else "") + "The trigger itself was near its threshold."
            if notes.get(d.id) == "skipped_budget":
                note += " The budget ran out before every model family was tested."
            if exps[0].metric != out.metric_plan.dev_metric:
                note += (f" Experiments used {exps[0].metric}; the metric is now {out.metric_plan.dev_metric}: "
                         "re-run report.verify(df).")
            out.decisions[i] = evolve(d, status=status, note=note.strip(), experiment_ids=[e.id for e in exps])
        elif d.id in notes:
            n = notes[d.id]
            if n == "skipped_budget":
                out.decisions[i] = evolve(d, status=DecisionStatus.SKIPPED_BUDGET)
            elif n == "not_applicable":
                out.decisions[i] = evolve(d, status=DecisionStatus.NOT_APPLICABLE,
                                          note="The estimator has no parameter for this change.")
            else:
                out.decisions[i] = evolve(d, note=n)
        elif d.id in prev:
            out.decisions[i] = evolve(d, status=prev[d.id][0], note=prev[d.id][1])


def _rederive(report: Report, *, ctx_state: ContextState, manifest: Manifest, estimator: Any,
              experiments: Any = None, baseline: Any = None, notes: dict[str, str] | None = None,
              llm: Any = "keep") -> Report:
    """Re-run the rules on the stored profile and rebuild the report around them."""
    d = report.details
    out = run_rules(d.profile, _rule_context(ctx_state, estimator, Budget(**(manifest.budget or {})),
                                             manifest.external_holdout))
    _validate_answers(ctx_state.answers, out)
    experiments = report.experiments if experiments is None else experiments
    prev = {x.id: (x.status, x.note) for x in report.decisions
            if x.cls is DecisionClass.CONDITIONAL and x.status in
            (DecisionStatus.SKIPPED_BUDGET, DecisionStatus.NOT_APPLICABLE)} if notes is None else {}
    _apply_experiments(out, experiments, notes or {}, prev)
    figures = [f for f in d.figures if not f.id.startswith("fig.exp.")]
    for dec in out.decisions:
        exps = [e for e in experiments if e.decision_id == dec.id]
        if exps:
            figures.append(experiment_figure(dec, exps))
    figures.sort(key=lambda f: (f.priority, f.id))
    _link(figures, out.evidence, out.decisions)
    new = _make_report(d.profile, ctx_state, out, figures, experiments,
                       report.baseline if baseline is None else baseline, manifest,
                       llm=d.llm if llm == "keep" else llm)
    return _carry(report, new, estimator)


def _apply_answers(report: Report, answers: dict[str, str]) -> Report:
    answers = _normalize_answers(answers)
    q_task = answers.get("q.task")
    if q_task is not None and q_task != report.task.value:
        raise ValueError(f"You answered that the task is {q_task}: re-run pf.analyze(df, target, task='{q_task}').")
    ctx = report.details.context
    ctx_state = evolve(ctx, answers={**ctx.answers, **answers})
    manifest = _next_revision(report.details.manifest, "answer", answers=answers)
    return _rederive(report, ctx_state=ctx_state, manifest=manifest, estimator=report._estimator)


def _holdout_positions(report: Report, df: pd.DataFrame) -> np.ndarray:
    """Positions of the internal holdout; re-derived deterministically when not in memory."""
    if report._holdout_positions is not None:
        return np.asarray(report._holdout_positions, dtype=int)
    h = report.details.profile.holdout
    if not h.get("used"):
        return np.array([], dtype=int)
    ctx = report.details.context
    if ctx.groups == "<array>":
        raise ValueError("This report was built with a groups array; holdout positions cannot be re-derived "
                         "after loading. Re-run pf.analyze.")
    g = df[ctx.groups] if ctx.groups else None
    _, hold, _ = reserve_holdout(df, report.target, report.task, h["fraction"], g, ctx.time_col,
                                 report.details.manifest.seed)
    return hold


def _holdout(report: Report, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = report.details.manifest
    check_fingerprint(df, m.source_fingerprint, report.target, "report.holdout", report._row_hashes)
    if not report.details.profile.holdout.get("used"):
        raise ValueError("This report has no internal holdout (analyze ran with holdout=None). If you split with "
                         "pf.split, your test frame is already separate; record it with "
                         "report.register_external_holdout(test_df).")
    hold = _holdout_positions(report, df)
    mask = np.zeros(len(df), dtype=bool)
    mask[hold] = True
    return df.iloc[~mask], df.iloc[mask]


def _register_external_holdout(report: Report, holdout_df: Any, dev_df: Any = None,
                               name: str = "external holdout") -> Report:
    if not isinstance(holdout_df, pd.DataFrame):
        raise TypeError("holdout_df must be a pandas DataFrame.")
    if report.target not in holdout_df.columns:
        raise ValueError(f"holdout_df has no target column '{report.target}'.")
    m = report.details.manifest
    fp = fingerprint(holdout_df, report.target)
    meta: dict[str, Any] = {
        "name": name, "n_rows": fp["n_rows"], "n_columns": fp["n_columns"],
        "same_columns_as_dev": fp["column_schema_digest"] == m.source_fingerprint.get("column_schema_digest"),
        "content_digest": fp["content_digest"], "column_schema_digest": fp["column_schema_digest"],
        "profiled": False, "overlap_rows": None,
    }
    if dev_df is not None:
        check_fingerprint(dev_df, m.source_fingerprint, report.target, "register_external_holdout(dev_df=...)",
                          report._row_hashes)
        common = [c for c in dev_df.columns if c in holdout_df.columns]
        dev_rows = set(row_hashes(dev_df[common]).tolist())
        hold_rows = row_hashes(holdout_df[common])
        overlap = int(sum(h in dev_rows for h in hold_rows.tolist()))
        meta["overlap_rows"] = overlap
        meta["overlap_columns_compared"] = len(common)
        if overlap:
            warnings.warn(f"Exact row overlap detected between development and {name} ({overlap} rows). "
                          "This may inflate evaluation; inspect whether the duplicated observations are legitimate "
                          "repeats.", stacklevel=3)
    manifest = evolve(_next_revision(m, "register_external_holdout", name=name, n_rows=fp["n_rows"],
                                     overlap_rows=meta["overlap_rows"]), external_holdout=meta)
    return _rederive(report, ctx_state=report.details.context, manifest=manifest, estimator=report._estimator)


def _verify(report: Report, df: pd.DataFrame, estimator: Any = None, budget: Budget | None = None,
            groups: Any = None) -> Report:
    from .verify import run_verification

    m = report.details.manifest
    check_fingerprint(df, m.source_fingerprint, report.target, "report.verify", report._row_hashes)
    budget = budget or _budget_of(report)
    estimator = estimator if estimator is not None else report._estimator
    hold = _holdout_positions(report, df)
    mask = np.zeros(len(df), dtype=bool)
    mask[hold] = True
    dev = df.iloc[~mask]
    ctx = report.details.context
    g: Any = None
    if ctx.groups and ctx.groups != "<array>":
        g = ctx.groups
    elif ctx.groups == "<array>":
        if groups is None or isinstance(groups, str):
            raise ValueError("This report was built with a groups array; pass the same array: "
                             "report.verify(df, groups=...).")
        arr = np.asarray(groups).ravel()
        if len(arr) != len(df):
            raise ValueError("groups must have one value per row of df.")
        g = arr[~mask]

    experiments, baseline, notes = run_verification(report, dev, estimator, budget, g, m.seed)
    ctx_state = evolve(ctx, estimator=_describe(estimator))
    manifest = _next_revision(evolve(m, budget=budget.model_dump()), "verify",
                              estimator=_describe(estimator) or "proxies (linear + tree)",
                              experiments=len(experiments), budget=budget.model_dump(),
                              protocol=experiments[0].protocol if experiments else None,
                              cv=baseline.get("cv"))
    return _rederive(report, ctx_state=ctx_state, manifest=manifest, estimator=estimator,
                     experiments=experiments, baseline=baseline, notes=notes)
