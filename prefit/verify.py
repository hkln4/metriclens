"""Bounded verification experiments for CONDITIONAL decisions.

Design constraints (so this never becomes AutoML):
- each conditional decision is tested once per model family, as
  "base setup" vs "base setup + this one change", on identical folds;
- the base setup is the procedural + user-confirmed decisions only;
- a hard budget caps experiment count, wall-clock time and rows;
- the significance test depends on the split protocol: repeated (or grouped)
  k-fold uses the corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert
  & Frank 2004), Holm-corrected across the run; time-ordered folds are not
  exchangeable, so no p-value is computed there;
- "improved" needs direction, significance AND a practical effect threshold
  (Budget.min_effect / min_relative_effect); smaller significant differences
  are "negligible";
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


def verification_cv(report: Any, y: pd.Series, budget: Budget, seed: int) -> tuple[Any, str, str]:
    """The splitter experiments and baselines use: the export splitter (the CV
    decision), repeated `budget.n_repeats` times when it is a plain k-fold.
    Returns (splitter, protocol, description)."""
    from sklearn import model_selection as ms

    step = next(d.step for d in report.decisions if d.id == "d.cv" and d.step is not None)
    params = dict(step.params)
    name = params.pop("name")
    dup_aware = bool(params.pop("duplicate_aware", False))
    if name == "TimeSeriesSplit":
        return ms.TimeSeriesSplit(**params), "time_series", f"TimeSeriesSplit(n_splits={params['n_splits']})"
    if name in ("GroupKFold", "StratifiedGroupKFold"):
        return getattr(ms, name)(**params), "grouped_kfold", f"{name}(n_splits={params['n_splits']})"
    k = int(params["n_splits"])
    if report.task is Task.CLASSIFICATION:
        k = int(max(2, min(k, y.value_counts().min())))  # the dev subsample may have fewer minority rows
    repeats = int(params.get("n_repeats", budget.n_repeats))
    stratified = name in ("StratifiedKFold", "RepeatedStratifiedKFold")
    if dup_aware:
        from .splitters import DuplicateAwareKFold

        cv = DuplicateAwareKFold(n_splits=k, n_repeats=repeats, stratify=stratified, random_state=seed)
        return (cv, "repeated_kfold", f"{cv!r} — the export splitter, repeated; exact duplicates kept in one fold")
    cls = ms.RepeatedStratifiedKFold if stratified else ms.RepeatedKFold
    return (cls(n_splits=k, n_repeats=repeats, random_state=seed), "repeated_kfold",
            f"{cls.__name__}(n_splits={k}, n_repeats={repeats}) — the export splitter, repeated")


def _protocol_method(protocol: str, n_folds: int) -> str:
    if protocol == "time_series":
        return (f"{n_folds} expanding-window time-series folds. Training sets overlap and grow and folds are not "
                "exchangeable, so no significance test is valid here; the outcome is left to you from the per-fold "
                "differences.")
    if protocol == "grouped_kfold":
        return (f"one run of {n_folds}-fold grouped CV on identical folds (groups cannot be re-shuffled without "
                "changing the split); corrected resampled t-test (Nadeau & Bengio 2003), df = k-1, low power with "
                "few folds; Holm across this run.")
    return (f"{n_folds} folds of repeated k-fold CV on identical folds; corrected repeated k-fold t-test (Nadeau & "
            "Bengio 2003; Bouckaert & Frank 2004): variance x (1/n + n_test/n_train), df = n-1; Holm across this run.")


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


def classify_outcome(mean_diff: float, p_adjusted: float | None, min_effect: float, alpha: float) -> str:
    """Direction + significance + practical size. Without a valid test: not_tested."""
    if p_adjusted is None:
        return "not_tested"
    if p_adjusted >= alpha:
        return "no_difference"
    if mean_diff > 0 and mean_diff >= min_effect:
        return "improved"
    if mean_diff < 0 and -mean_diff >= min_effect:
        return "worse"
    return "negligible"


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
    from sklearn.pipeline import Pipeline

    p = report.details.profile
    ctx = report.details.context
    tgt = report.target
    # work on positions so a groups array stays aligned through filtering/sampling
    df = df.reset_index(drop=True)
    if groups is not None and not isinstance(groups, str):
        groups = pd.Series(np.asarray(groups).ravel())
    df = df[df[tgt].notna()]
    if len(df) > budget.max_rows:
        from .guard import sample_rows

        strat = df[tgt] if report.task is Task.CLASSIFICATION else None
        df, _ = sample_rows(df, budget.max_rows, seed, strat)
    if ctx.time_col:
        df = df.sort_values(ctx.time_col, kind="stable")
    g = None if groups is None else (df[groups] if isinstance(groups, str) else groups.loc[df.index])
    X, y = df.drop(columns=[tgt]), df[tgt]

    plan = report.metric_plan
    scorer = build_scorer(plan.dev_metric, p, positive_label(p, plan.positive_label))
    if isinstance(scorer, str):
        scorer = get_scorer(scorer)
    cv, protocol, cv_desc = verification_cv(report, y, budget, seed)
    splits = list(cv.split(X, y, g) if g is not None else cv.split(X, y))
    ratio = float(np.mean([len(te) / len(tr) for tr, te in splits]))
    method = _protocol_method(protocol, len(splits))
    n_rep = int(getattr(cv, "n_repeats", 1))
    n_spl = len(splits) // n_rep

    base_steps = [d.step for d in report.decisions
                  if d.step and d.step.kind != "cv" and d.cls is not DecisionClass.CONDITIONAL and d.eligible()]
    families = [("user", estimator)] if estimator is not None else proxy_estimators(report.task)

    t0 = time.perf_counter()
    baseline: dict[str, Any] = {"metric": plan.dev_metric, "n_rows": int(len(df)), "n_folds": len(splits),
                                "cv": cv_desc, "protocol": protocol,
                                "provisional_metric": plan.dev_metric_provisional, "families": {}}
    dummy = DummyClassifier(strategy="prior") if report.task is Task.CLASSIFICATION else DummyRegressor(strategy="median")
    try:
        ds = _score(Pipeline([("model", dummy)]), X, y, splits, scorer)
        baseline["dummy"] = {"mean": float(ds.mean()), "std": float(ds.std(ddof=1))}
    except Exception as exc:  # pragma: no cover - scorer incompatible with dummy
        baseline["dummy"] = {"error": str(exc)}
    base_scores: dict[str, np.ndarray] = {}
    for fam, est in families:
        sc = _score(build_model(p, base_steps, est), X, y, splits, scorer)
        base_scores[fam] = sc
        baseline["families"][fam] = {"estimator": type(est).__name__, "mean": float(sc.mean()),
                                     "std": float(sc.std(ddof=1))}

    candidates = [d for d in report.decisions if d.cls is DecisionClass.CONDITIONAL and d.status in _TESTABLE]
    candidates.sort(key=lambda d: (d.status is DecisionStatus.UNCERTAIN))
    raw: list[dict[str, Any]] = []
    notes: dict[str, str] = {}
    for d in candidates:
        assert d.step is not None
        for fam, est in families:
            if len(raw) >= budget.max_experiments or time.perf_counter() - t0 > budget.max_seconds:
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
            base = base_scores[fam]
            min_eff = (budget.min_effect if budget.min_effect is not None
                       else budget.min_relative_effect * abs(float(base.mean())))
            diffs = cand - base
            if protocol == "time_series":
                mean, ci, pval = float(diffs.mean()), None, None
            else:
                mean, ci, pval = corrected_ttest(base, cand, ratio)
            raw.append(dict(
                id=f"exp.{d.id[2:]}.{fam}", decision_id=d.id, family=fam, estimator=type(est).__name__,
                metric=plan.dev_metric, n_rows=int(len(df)), n_splits=n_spl, n_repeats=n_rep, cv=cv_desc,
                baseline_scores=[float(x) for x in base], candidate_scores=[float(x) for x in cand],
                mean_diff=mean, ci=ci, p_value=pval, min_effect=float(min_eff),
                folds_improved=int((diffs > 0).sum()), protocol=protocol, method=method,
                seconds=round(time.perf_counter() - ts, 2)))
    tested = [r for r in raw if r["p_value"] is not None]
    for r, a in zip(tested, holm([float(r["p_value"]) for r in tested]) if tested else []):
        r["p_adjusted"] = a
    experiments = [Experiment(**r, outcome=classify_outcome(r["mean_diff"], r.get("p_adjusted"), r["min_effect"],
                                                            budget.alpha)) for r in raw]
    baseline["seconds"] = round(time.perf_counter() - t0, 2)
    return experiments, baseline, notes


def status_from_experiments(d: Decision, exps: list[Experiment], user: bool) -> tuple[DecisionStatus, str]:
    outcomes = {e.family: e.outcome for e in exps}
    if "not_tested" in outcomes.values():
        return (DecisionStatus.INCONCLUSIVE,
                "Time-ordered folds admit no significance test; judge from the per-fold differences.")
    if "negligible" in outcomes.values() and "improved" not in outcomes.values() and "worse" not in outcomes.values():
        return (DecisionStatus.INCONCLUSIVE,
                "The difference is statistically detectable but below the practical threshold (Budget.min_effect).")
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
