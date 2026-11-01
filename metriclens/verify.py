"""Bounded verification experiments for CONDITIONAL decisions.

Design constraints (so this never becomes AutoML):
- each conditional decision is tested once per model family, as
  "base setup" vs "base setup + this one change", on identical folds;
- the base setup is the procedural + user-confirmed decisions only;
- a hard budget caps experiment count, wall-clock time and rows;
- differences are judged with a paired, Nadeau-Bengio corrected t-test, and
  Holm-corrected across every experiment in the run;
- the user's estimator, when given, is the only thing that can VERIFY a
  decision; proxy models produce a sensitivity analysis (SUPPORTED_BY_PROXY).
All of it is development evidence on the dev rows — never the holdout.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .config import Budget
from .enums import DecisionClass, DecisionStatus, Task
from .export import build_model, build_scorer, positive_label, proxy_estimators
from .report.model import Decision, Experiment

_TESTABLE = (DecisionStatus.PROPOSED, DecisionStatus.UNCERTAIN)


def _splits(report: Any, X: pd.DataFrame, y: pd.Series, groups: Any, budget: Budget, seed: int) -> list:
    from sklearn import model_selection as ms

    task = report.task
    if report.context.time_col:
        cv = ms.TimeSeriesSplit(n_splits=budget.n_splits)
        return list(cv.split(X))
    if groups is not None:
        cv = (ms.StratifiedGroupKFold(budget.n_splits, shuffle=True, random_state=seed)
              if task is Task.CLASSIFICATION else ms.GroupKFold(budget.n_splits))
        return list(cv.split(X, y, groups))
    if task is Task.CLASSIFICATION:
        k = int(max(2, min(budget.n_splits, y.value_counts().min())))
        cv = ms.RepeatedStratifiedKFold(n_splits=k, n_repeats=budget.n_repeats, random_state=seed)
    else:
        cv = ms.RepeatedKFold(n_splits=budget.n_splits, n_repeats=budget.n_repeats, random_state=seed)
    return list(cv.split(X, y))


def corrected_ttest(base: np.ndarray, cand: np.ndarray, test_train_ratio: float) -> tuple[float, tuple[float, float], float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores.
    Returns (mean_diff, 95% CI, two-sided p)."""
    from scipy import stats

    d = cand - base
    k = len(d)
    mean = float(d.mean())
    var = float(d.var(ddof=1)) if k > 1 else 0.0
    se = float(np.sqrt((1.0 / k + test_train_ratio) * var))
    if se == 0:
        return mean, (mean, mean), (1.0 if mean == 0 else 0.0)
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), k - 1))
    half = float(stats.t.ppf(0.975, k - 1) * se)
    return mean, (mean - half, mean + half), p


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [float(a) for a in adj]


def _score(model: Any, X: pd.DataFrame, y: pd.Series, splits: list, scorer: Any) -> np.ndarray:
    from sklearn.model_selection import cross_val_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_score(model, X, y, cv=splits, scoring=scorer, error_score="raise")


def run_verification(report: Any, df: pd.DataFrame, estimator: Any, budget: Budget,
                     groups: Any, seed: int) -> tuple[list[Experiment], dict[str, Any], dict[str, str]]:
    """Return (experiments, baseline, notes) — notes map decision id -> status note."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import get_scorer

    p = report.profile
    tgt = report.target
    # work on positions so a groups Series stays aligned through filtering/sampling
    df = df.reset_index(drop=True)
    if groups is not None and not isinstance(groups, str):
        groups = pd.Series(np.asarray(groups).ravel())
    df = df[df[tgt].notna()]
    if len(df) > budget.max_rows:
        from .guard import sample_rows

        strat = df[tgt] if report.task is Task.CLASSIFICATION else None
        df, _ = sample_rows(df, budget.max_rows, seed, strat)
    if report.context.time_col:
        df = df.sort_values(report.context.time_col, kind="stable")
    g = None if groups is None else (df[groups] if isinstance(groups, str) else groups.loc[df.index])
    X, y = df.drop(columns=[tgt]), df[tgt]

    plan = report.metric_plan
    scorer = build_scorer(plan.dev_metric, p, positive_label(p, plan.positive_label))
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)
    splits = _splits(report, X, y, g, budget, seed)
    ratio = float(np.mean([len(te) / len(tr) for tr, te in splits]))

    base_steps = [d.step for d in report.decisions
                  if d.step and d.step.kind != "cv" and d.cls is not DecisionClass.CONDITIONAL and d.eligible()]
    families = [("user", estimator)] if estimator is not None else proxy_estimators(report.task)

    t0 = time.perf_counter()
    baseline: dict[str, Any] = {"metric": plan.dev_metric, "n_rows": int(len(df)), "n_folds": len(splits),
                                "provisional_metric": plan.dev_metric_provisional, "families": {}}
    dummy = DummyClassifier(strategy="prior") if report.task is Task.CLASSIFICATION else DummyRegressor(strategy="median")
    from sklearn.pipeline import Pipeline

    try:
        ds = _score(Pipeline([("model", dummy)]), X, y, splits, scorer)
        baseline["dummy"] = {"mean": float(ds.mean()), "std": float(ds.std(ddof=1))}
    except Exception as exc:  # pragma: no cover - scorer incompatible with dummy
        baseline["dummy"] = {"error": str(exc)}
    base_scores: dict[str, np.ndarray] = {}
    for fam, est in families:
        s = _score(build_model(p, base_steps, est), X, y, splits, scorer)
        base_scores[fam] = s
        baseline["families"][fam] = {"estimator": type(est).__name__, "mean": float(s.mean()),
                                     "std": float(s.std(ddof=1))}

    candidates = [d for d in report.decisions if d.cls is DecisionClass.CONDITIONAL and d.status in _TESTABLE]
    candidates.sort(key=lambda d: (d.status is DecisionStatus.UNCERTAIN))
    experiments: list[Experiment] = []
    notes: dict[str, str] = {}
    runs = 0
    for d in candidates:
        for fam, est in families:
            if runs >= budget.max_experiments or time.perf_counter() - t0 > budget.max_seconds:
                notes[d.id] = "skipped_budget"
                break
            if d.step.kind == "class_weight" and "class_weight" not in est.get_params():
                notes.setdefault(d.id, "not_applicable")
                continue
            ts = time.perf_counter()
            try:
                cand = _score(build_model(p, base_steps + [d.step], est), X, y, splits, scorer)
            except Exception as exc:
                notes[d.id] = f"error: {exc}"
                continue
            mean, ci, pval = corrected_ttest(base_scores[fam], cand, ratio)
            runs += 1
            experiments.append(Experiment(
                id=f"exp.{d.id[2:]}.{fam}", decision_id=d.id, family=fam, estimator=type(est).__name__,
                metric=plan.dev_metric, n_rows=int(len(df)), n_splits=budget.n_splits,
                n_repeats=1 if (g is not None or report.context.time_col) else budget.n_repeats,
                baseline_scores=[float(x) for x in base_scores[fam]], candidate_scores=[float(x) for x in cand],
                mean_diff=mean, ci=ci, p_value=pval, seconds=round(time.perf_counter() - ts, 2)))
    if experiments:
        for e, a in zip(experiments, holm([e.p_value for e in experiments])):
            e.p_adjusted = a
            e.outcome = ("improved" if e.mean_diff > 0 else "worse") if a < budget.alpha else "no_difference"
    baseline["seconds"] = round(time.perf_counter() - t0, 2)
    return experiments, baseline, notes


def status_from_experiments(d: Decision, exps: list[Experiment], user: bool) -> tuple[DecisionStatus, str]:
    outcomes = {e.family: e.outcome for e in exps}
    if user:
        o = outcomes.get("user", "")
        return ({"improved": DecisionStatus.VERIFIED, "worse": DecisionStatus.REFUTED}
                .get(o, DecisionStatus.INCONCLUSIVE), "")
    vals = set(outcomes.values())
    if "improved" in vals and "worse" in vals:
        return DecisionStatus.INCONCLUSIVE, "Model-family dependent: helps one proxy, hurts the other."
    if "improved" in vals:
        fams = [f for f, o in outcomes.items() if o == "improved"]
        return DecisionStatus.SUPPORTED_BY_PROXY, f"Supported by {', '.join(fams)} only (sensitivity analysis)."
    if vals == {"worse"}:
        return DecisionStatus.REFUTED, "Hurts every proxy model tested."
    return DecisionStatus.INCONCLUSIVE, "No difference beyond fold noise for the proxies tested."
