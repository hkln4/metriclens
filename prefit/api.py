"""Public entry points and orchestration.

    pf.split(df, ...)            -> (dev_df, test_df)      data preparation, explicit
    pf.analyze(dev_df, ...)      -> Report                 profile -> rules -> figures
    report.answer / resolve / change_task / verify / interpret / register_external_holdout
                                 -> new Report revision
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
    With "stratified" and "random", exact duplicates stay on one side: identical
    rows (target included) move together, so a copy never tests its twin.
    "group" keeps your groups whole instead (and warns when copies span groups);
    "time" ignores duplicates.
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
    keep_data: bool = True,
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
        `report.holdout(df)`. Exact duplicates stay on one side (unless groups
        or time_col define the split), and cross-validation keeps them in one
        fold.
    split_columns_as_features : by default the groups / time_col columns are
        kept out of the model features; True includes them (with a warning).
    context : free-text description of the application (used by the LLM layer).
    answers : answers to report questions, e.g. {"metric_priority": "miss_costly"}.
    verify : run the bounded experiments for conditional decisions (needs the
        `verify` extra: scikit-learn + scipy).
    budget : experiment limits (see `prefit.Budget`).
    llm : optional `prefit.LLMConfig` (or a provider name / Provider). Off by
        default; see `Report.llm_payload()` for exactly what would be sent.
    keep_data : keep a reference to `df` (no copy) so `verify()`, `holdout()`,
        `change_task()`, `answer(task=...)`, `export().fit()` and `review(model)`
        need no frame. It is fingerprint-checked on every use (an in-place edit
        raises) and never serialized. False keeps no reference.
    """
    return _analyze(df, target, task=task, estimator=estimator, groups=groups, time_col=time_col, holdout=holdout,
                    split_columns_as_features=split_columns_as_features, context=context, answers=answers,
                    verify=verify, budget=budget, llm=llm, seed=seed, profile_options=profile_options,
                    keep_data=keep_data)


def _analyze(df: pd.DataFrame, target: str, *, task: Task | str | None, estimator: Any, groups: Any,
             time_col: str | None, holdout: float | None, split_columns_as_features: bool, context: str | None,
             answers: dict[str, str] | None, verify: bool, budget: Budget | None, llm: Any, seed: int,
             profile_options: ProfileOptions | None, keep_data: bool, fixed_holdout: Any = None,
             session: dict | None = None) -> Report:
    """analyze(); `fixed_holdout` = (dev_pos, hold_pos, info) reuses a partition (change_task)."""
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

    if fixed_holdout is not None:
        dev_pos, hold_pos, hold_info = fixed_holdout
    else:
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
    report._data = df if keep_data else None
    report._session = session if session is not None else {"test_reviews": {}}
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
           segments: Any = None, task: Task | str | None = None, X: Any = None, n_bootstrap: int = 1000,
           seed: int = 0) -> Any:
    """Post-training review against the roadmap's assumptions (see `prefit.posttrain`).

    Headline metrics get 95% bootstrap intervals; pass the test features as `X=` so
    exact duplicate rows are resampled together. After `report.verify`, the review
    also checks that the CV estimate lies inside the test interval."""
    from .posttrain import review as _review

    return _review(y_true, y_pred, report=report, y_score=y_score, segments=segments, task=task, X=X,
                   n_bootstrap=n_bootstrap, seed=seed)


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
            raise KeyError(f"Unknown question '{k}'. Questions in this report: {sorted(qs)}. Review items are "
                           "closed with report.resolve(decision_id, note), not answered.")
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
    new._data = report._data
    new._session = report._session if report._session is not None else {"test_reviews": {}}
    return new


def _frame(report: Report, df: Any, where: str) -> pd.DataFrame:
    """The development frame: `df` when given, else the one kept in memory; fingerprint-checked."""
    if df is None:
        df = report._data
        if df is None:
            raise ValueError(f"{where} needs the DataFrame passed to analyze: this report keeps none (it was loaded "
                             f"from JSON or built with keep_data=False). Pass it: {where}(df).")
    check_fingerprint(df, report.details.manifest.source_fingerprint, report.target, where, report._row_hashes)
    return df


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
    _apply_resolutions(out, ctx_state)
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


def _apply_resolutions(out: RuleOutput, ctx: ContextState) -> None:
    for i, d in enumerate(out.decisions):
        if d.cls is DecisionClass.REVIEW and d.id in ctx.resolutions:
            out.decisions[i] = evolve(d, status=DecisionStatus.REVIEWED, review_note=ctx.resolutions[d.id])


def _resolve(report: Report, decision_id: str, note: str) -> Report:
    did = decision_id if decision_id.startswith("d.") else f"d.{decision_id}"
    d = report.decision(did)
    if d.cls is not DecisionClass.REVIEW:
        raise ValueError(f"{did} is a {d.cls.value} decision; only review items are resolved with a note. "
                         + ("Answer its question with report.answer(...)." if d.question_id else ""))
    if not isinstance(note, str) or not note.strip():
        raise ValueError("resolve needs a note saying what you decided and why.")
    ctx = report.details.context
    ctx_state = evolve(ctx, resolutions={**ctx.resolutions, did: note.strip()})
    manifest = _next_revision(report.details.manifest, "resolve", decision=did, note=note.strip())
    return _rederive(report, ctx_state=ctx_state, manifest=manifest, estimator=report._estimator)


def _change_task(report: Report, df: Any, task: Task | str) -> Report:
    new_task = Task(getattr(task, "value", task))
    m = report.details.manifest
    df = _frame(report, df, "report.change_task")
    ctx = report.details.context
    if ctx.groups == "<array>":
        raise ValueError("This report was built with a groups array, which is not stored; re-run "
                         "pf.analyze(df, target, task=..., groups=...).")
    first = m.history[0] if m.history else {}
    common: dict[str, Any] = dict(
        task=new_task, estimator=report._estimator, groups=ctx.groups, time_col=ctx.time_col,
        holdout=first.get("holdout"), split_columns_as_features=ctx.split_columns_as_features, context=ctx.text,
        budget=Budget(**(m.budget or {})), seed=m.seed, profile_options=ProfileOptions(**m.profile_options))
    # the same holdout rows: they were set aside under the old task and must stay unread
    hold = _holdout_positions(report, df)
    mask = np.zeros(len(df), dtype=bool)
    mask[hold] = True
    fixed = (np.flatnonzero(~mask), np.flatnonzero(mask), dict(report.details.profile.holdout))
    fresh = _analyze(df, report.target, split_columns_as_features=common.pop("split_columns_as_features"),
                     answers=None, verify=False, llm=None, keep_data=report._data is not None, fixed_holdout=fixed,
                     session=report._session, **common)
    qs = {q.id: {o.value for o in q.options} for q in fresh.questions}
    kept = {k: v for k, v in ctx.answers.items() if k != "q.task" and v in qs.get(k, set())}
    dropped = sorted(set(ctx.answers) - set(kept) - {"q.task"})
    review_ids = {d.id for d in fresh.decisions if d.cls is DecisionClass.REVIEW}
    ctx_state = evolve(fresh.details.context, answers=kept,
                       resolutions={k: v for k, v in ctx.resolutions.items() if k in review_ids})
    rev = m.revision + 1
    manifest = evolve(fresh.details.manifest, revision=rev, external_holdout=m.external_holdout, history=[
        *m.history, {"revision": rev, "action": "change_task", "from": report.task.value, "task": new_task.value,
                     "answers_kept": sorted(kept), "answers_dropped": dropped,
                     "experiments_dropped": len(report.experiments)}])
    return _rederive(fresh, ctx_state=ctx_state, manifest=manifest, estimator=report._estimator, experiments=(),
                     baseline={}, notes={})


def _option_value(q: Any, v: Any) -> str:
    """An answer -> the option value: the value itself, an alias ("mae"), or True/False for column questions."""
    if isinstance(v, (bool, np.bool_)):
        for o in q.options:
            if o.flag is bool(v):
                return o.value
        raise ValueError(f"{q.id} ({q.text}) takes one of {[o.value for o in q.options]}, not {v!r}.")
    text = str(getattr(v, "value", v))
    for o in q.options:
        if text == o.value or text.lower() in {a.lower() for a in o.aliases}:
            return o.value
    opts = ", ".join(o.value + (f" ({'/'.join(o.aliases)})" if o.aliases else "") for o in q.options)
    raise ValueError(f"{text!r} is not an answer to {q.id} ({q.text}). Options: {opts}.")


def _keywords(report: Report) -> list[str]:
    return sorted({q.keyword for q in report.questions if q.keyword} | {"task"})


def _resolve_keywords(report: Report, answers: dict[str, Any], by_keyword: dict[str, Any]) -> dict[str, str]:
    """{question id: option value} from question-id answers and keyword answers."""
    out: dict[str, str] = {}
    qs = {q.id: q for q in report.questions}
    for raw, v in answers.items():
        k = raw if raw.startswith("q.") else f"q.{raw}"
        q = qs.get(k)
        out[k] = _option_value(q, v) if q is not None else str(v)
    for kw, val in by_keyword.items():
        if kw == "task":
            if "q.task" in qs:
                out["q.task"] = _option_value(qs["q.task"], val)
            continue
        matches = [q for q in report.questions if q.keyword == kw or q.id == f"q.{kw}"]
        if not matches:
            raise KeyError(f"Unknown answer keyword {kw!r}. This report takes: {', '.join(_keywords(report))}. "
                           "Review items are closed with report.resolve(decision_id, note), not answered.")
        by_col = {q.column: q for q in matches if q.column is not None}
        if not by_col:
            out[matches[0].id] = _option_value(matches[0], val)
            continue
        items = (val.items() if isinstance(val, dict)
                 else [(c, True) for c in ([val] if isinstance(val, str) else val)])
        for col, v in items:
            if col not in by_col:
                raise KeyError(f"{kw}: no open question about column {col!r}; there are questions about "
                               f"{sorted(by_col)}.")
            out[by_col[col].id] = _option_value(by_col[col], v)
    return out


def _apply_answers(report: Report, answers: dict[str, Any], by_keyword: dict[str, Any] | None = None) -> Report:
    by_keyword = dict(by_keyword or {})
    answers = dict(answers)
    new_task = by_keyword.pop("task", None)
    for k in ("task", "q.task"):
        if k in answers and new_task is None:
            new_task = answers.pop(k)
    if new_task is not None:
        t = Task(getattr(new_task, "value", new_task))
        if t is not report.task:
            # a different task re-runs the analysis (same frame, same holdout), then the other answers apply
            try:
                report = _change_task(report, None, t)
            except ValueError as exc:
                if report._data is not None:
                    raise
                raise ValueError(f"answer(task={t.value!r}) re-runs the analysis and needs the frame passed to "
                                 f"analyze; this report keeps none. Use report.change_task({t.value!r}, df), then "
                                 "answer the rest.") from exc
            if not answers and not by_keyword:
                return report
        else:
            by_keyword["task"] = t.value
    resolved = _resolve_keywords(report, answers, by_keyword)
    ctx = report.details.context
    ctx_state = evolve(ctx, answers={**ctx.answers, **resolved})
    manifest = _next_revision(report.details.manifest, "answer", answers=resolved)
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
    reserved_as = Task(h.get("task", report.task.value))
    _, hold, _ = reserve_holdout(df, report.target, reserved_as, h["fraction"], g, ctx.time_col,
                                 report.details.manifest.seed)
    return hold


def _holdout(report: Report, df: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _frame(report, df, "report.holdout")
    if not report.details.profile.holdout.get("used"):
        raise ValueError("This report has no internal holdout (analyze ran with holdout=None). If you split with "
                         "pf.split, your test frame is already separate; record it with "
                         "report.register_external_holdout(test_df).")
    hold = _holdout_positions(report, df)
    mask = np.zeros(len(df), dtype=bool)
    mask[hold] = True
    return df.iloc[~mask], df.iloc[mask]


def _review_model(report: Report, model: Any, test_df: Any, *, segments: Any, n_bootstrap: int,
                  seed: int) -> Any:
    from .posttrain import review as _review

    if test_df is None:
        if not report.details.profile.holdout.get("used"):
            ext = report.details.manifest.external_holdout
            hint = (f"pass the {ext['name']} you registered: report.review(model, test_df)" if ext else
                    "pass your test frame: report.review(model, test_df), or analyze with holdout=0.2")
            raise ValueError(f"This report has no internal holdout; {hint}.")
        _, test_df = _holdout(report, None)
        which = "internal holdout"
    else:
        if not isinstance(test_df, pd.DataFrame):
            raise TypeError("test_df must be a pandas DataFrame.")
        ext = report.details.manifest.external_holdout
        digest = fingerprint(test_df, report.target)["content_digest"]
        if ext and ext.get("content_digest") != digest:
            warnings.warn(f"test_df is not the {ext['name']} registered with this report (its content digest "
                          "differs).", stacklevel=3)
        which = digest
    uses = report._session.setdefault("test_reviews", {}) if report._session is not None else {}
    uses[which] = uses.get(which, 0) + 1
    if uses[which] > 1:
        warnings.warn(f"This test data has been reviewed {uses[which]} times in this session. A test set you "
                      "look at more than once (to change the model in between) no longer gives an unbiased "
                      "estimate; report the first review.", stacklevel=3)
    if report.target not in test_df.columns:
        raise ValueError(f"test_df has no target column '{report.target}'.")
    cols = [str(c) for c in getattr(model, "feature_names_in_", [])]
    if not cols:
        from .export import export_pipeline

        cols = export_pipeline(report).feature_columns
    test = test_df[test_df[report.target].notna()]
    X = test[cols]
    y_score = None
    if report.task is Task.CLASSIFICATION and hasattr(model, "predict_proba") and report.metric_plan.positive_label:
        from .export import positive_label

        classes = list(getattr(model, "classes_", []))
        pos = positive_label(report.details.profile, report.metric_plan.positive_label)
        if len(classes) == 2 and pos in classes:
            y_score = model.predict_proba(X)[:, classes.index(pos)]
    return _review(test[report.target], model.predict(X), report=report, y_score=y_score, segments=segments,
                   X=X, n_bootstrap=n_bootstrap, seed=seed)


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
    if dev_df is None:
        dev_df = report._data
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
                          "repeats. Use pf.split, which keeps exact duplicates on one side.", stacklevel=3)
    manifest = evolve(_next_revision(m, "register_external_holdout", name=name, n_rows=fp["n_rows"],
                                     overlap_rows=meta["overlap_rows"]), external_holdout=meta)
    return _rederive(report, ctx_state=report.details.context, manifest=manifest, estimator=report._estimator)


def _verify(report: Report, df: Any, estimator: Any = None, budget: Budget | None = None,
            groups: Any = None) -> Report:
    from .verify import run_verification

    m = report.details.manifest
    df = _frame(report, df, "report.verify")
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
