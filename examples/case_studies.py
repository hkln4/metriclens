"""Quantified case studies: does following the roadmap change real outcomes?

Every number printed here is computed in this run — nothing is hardcoded. Each
study prints a compact before/after table. If a rule that SHOULD fire does not,
the script stops with a STOP message instead of working around it.

Run:  python examples/case_studies.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

import prefit as pf


def _stop(msg: str) -> None:
    print(f"\n*** STOP: {msg}\n"
          "A rule that should have fired did not. Reporting instead of patching.")
    sys.exit(1)


def _finding(report, rule_id):
    return next((f for f in report.details.findings if f.rule_id == rule_id), None)


def _evidence(report, finding):
    return {report.details.evidence[e].label: report.details.evidence[e].value for e in finding.evidence_ids}


def _table(title: str, subtitle: str, rows: list[tuple[str, str, str]],
           col_before: str, col_after: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{subtitle}\n{'-' * 72}")
    w = max(len(r[0]) for r in rows)
    print(f"{'':{w}}   {col_before:>16}   {col_after:>16}")
    for label, before, after in rows:
        print(f"{label:{w}}   {before:>16}   {after:>16}")


# --------------------------------------------------------------------------- #
# Case 1 — Skewed target (California Housing)
# --------------------------------------------------------------------------- #
def case1() -> dict:
    df = fetch_california_housing(as_frame=True).frame
    # holdout=None: these studies run their own CV on the full data
    report = pf.analyze(df, target="MedHouseVal", task="regression", holdout=None)
    fnd = _finding(report, "target_skew")
    if fnd is None:
        _stop("case 1: expected 'target_skew' to fire on California Housing.")

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"].to_numpy()
    cv = report.export().cv  # the splitter chosen by the procedural CV decision

    def model():
        return HistGradientBoostingRegressor(random_state=0)

    # (a) naive: raw target, plain regressor
    pred_naive = cross_val_predict(model(), X, y, cv=cv)
    # (b) roadmap-guided: log-transform the target (the preprocessing rec)
    guided = TransformedTargetRegressor(
        regressor=model(), func=np.log1p, inverse_func=np.expm1)
    pred_guided = cross_val_predict(guided, X, y, cv=cv)

    mae_n = mean_absolute_error(y, pred_naive)
    mae_g = mean_absolute_error(y, pred_guided)
    rmse_n = root_mean_squared_error(y, pred_naive)
    rmse_g = root_mean_squared_error(y, pred_guided)
    r2_n = r2_score(y, pred_naive)
    r2_g = r2_score(y, pred_guided)
    rel = (mae_n - mae_g) / mae_n * 100

    rev_n = pf.review(y, pred_naive, report)
    rev_g = pf.review(y, pred_guided, report)
    held_n = next(c.held for c in rev_n.checks if c.finding_id)
    held_g = next(c.held for c in rev_g.checks if c.finding_id)
    verdict = {True: "HELD", False: "did not hold", None: "n/a"}

    _table(
        "CASE 1 — Skewed target (California Housing)",
        f"rule fired: {fnd.rule_id}  evidence={_evidence(report, fnd)}",
        [("MAE (orig scale)", f"{mae_n:.4f}", f"{mae_g:.4f}"),
         ("RMSE", f"{rmse_n:.4f}", f"{rmse_g:.4f}"),
         ("R^2", f"{r2_n:.4f}", f"{r2_g:.4f}"),
         ("review: tails dominate", verdict[held_n], verdict[held_g])],
        "naive", "roadmap-guided")
    print(f"\nMAE relative improvement (naive -> guided): {rel:+.1f}%")
    return {"mae_naive": mae_n, "mae_guided": mae_g, "rel_improvement_pct": rel,
            "held_naive": held_n, "held_guided": held_g, "evidence": _evidence(report, fnd)}


# --------------------------------------------------------------------------- #
# Case 2 — Duplicate-row leakage (Breast Cancer)
# --------------------------------------------------------------------------- #
def _straddling(cv, X, y, hashes) -> int:
    """Rows whose exact copy sits on the other side of a fold boundary (summed over folds)."""
    return sum(int(np.isin(hashes[te], hashes[tr]).sum()) for tr, te in cv.split(X, y))


def case2() -> dict:
    df = load_breast_cancer(as_frame=True).frame
    rng = np.random.default_rng(0)
    dup = rng.choice(len(df), size=int(0.2 * len(df)), replace=False)
    corrupt = pd.concat([df, df.iloc[dup]], ignore_index=True)

    report = pf.analyze(corrupt, target="target", task="classification", holdout=None)
    fnd = _finding(report, "duplicate_rows")
    if fnd is None:
        _stop("case 2: expected 'duplicate_rows' to fire on the corrupted copy.")
    step = report.decision("d.cv").step
    if not step.params.get("duplicate_aware"):
        _stop("case 2: expected the CV decision to keep exact duplicates in one fold.")

    model = RandomForestClassifier(n_estimators=200, random_state=0)
    Xc, yc = corrupt.drop(columns="target"), corrupt["target"]
    hashes = pd.util.hash_pandas_object(corrupt, index=False).to_numpy()

    # old (prefit 0.1): the same splitter, row by row; copies can straddle folds
    sk = {k: v for k, v in step.params.items() if k not in ("name", "duplicate_aware")}
    cv_old = getattr(model_selection, step.params["name"])(**sk)
    # new (prefit 0.2): report.export().cv keeps exact duplicates in one fold
    cv_new = report.export().cv
    acc_old = cross_val_score(model, Xc, yc, cv=cv_old, scoring="accuracy").mean()
    acc_new = cross_val_score(model, Xc, yc, cv=cv_new, scoring="accuracy").mean()

    # reference: the review item resolved by de-duplicating (the copies are artefacts here)
    clean = corrupt.drop_duplicates().reset_index(drop=True)
    acc_clean = cross_val_score(model, clean.drop(columns="target"), clean["target"], cv=cv_old,
                                scoring="accuracy").mean()

    _table(
        "CASE 2 — Duplicate-row leakage (Breast Cancer + 20% copies)",
        f"rule fired: {fnd.rule_id}  evidence={_evidence(report, fnd)}\n"
        f"old: {type(cv_old).__name__} row by row · new: {type(cv_new).__name__} (report.export().cv)",
        [("CV accuracy", f"{acc_old:.4f}", f"{acc_new:.4f}"),
         ("copies across a fold", str(_straddling(cv_old, Xc, yc, hashes)),
          str(_straddling(cv_new, Xc, yc, hashes)))],
        "old (0.1)", "new (0.2)")
    print(f"\nInflation removed by the duplicate-aware CV: {(acc_old - acc_new) * 100:+.2f} accuracy points")
    print(f"Reference, de-duplicated data ({len(clean)} rows): CV accuracy {acc_clean:.4f}")
    return {"acc_old": acc_old, "acc_new": acc_new, "acc_clean": acc_clean,
            "gap_points": (acc_old - acc_new) * 100, "evidence": _evidence(report, fnd)}


# --------------------------------------------------------------------------- #
# Case 3 — Target leakage (Diabetes + injected feature)
# --------------------------------------------------------------------------- #
def case3() -> dict:
    df = load_diabetes(as_frame=True).frame  # target column: 'target'
    rng = np.random.default_rng(1)
    noise = rng.normal(0, df["target"].std() * 0.01, len(df))
    leaked = df.copy()
    leaked["leaky"] = leaked["target"] + noise  # |r| ~ 0.9999

    report = pf.analyze(leaked, target="target", task="regression", holdout=None)
    fnd = _finding(report, "leakage_hint")
    if fnd is None:
        _stop("case 3: expected 'leakage_hint' to fire on the injected feature.")
    if fnd.severity.value != "critical":
        _stop(f"case 3: 'leakage_hint' fired but severity={fnd.severity.value}, expected critical.")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_leak = cross_val_score(LinearRegression(), leaked.drop(columns="target"),
                              leaked["target"], cv=cv, scoring="r2").mean()
    r2_honest = cross_val_score(LinearRegression(), df.drop(columns="target"),
                                df["target"], cv=cv, scoring="r2").mean()
    gap = (r2_leak - r2_honest) * 100

    corr = report.details.evidence["ev.col.leaky.pearson"].value
    _table(
        "CASE 3 — Target leakage (Diabetes + injected feature)",
        f"rule fired: {fnd.rule_id} [{fnd.severity.value}]  |r|={abs(corr):.4f}",
        [("CV R^2", f"{r2_leak:.4f}", f"{r2_honest:.4f}")],
        "with leak", "leak removed")
    print(f"\nApparent R^2 inflation from leakage: {gap:+.1f} points")
    return {"r2_leak": r2_leak, "r2_honest": r2_honest, "gap_points": gap,
            "corr": corr}


if __name__ == "__main__":
    r1 = case1()
    r2 = case2()
    r3 = case3()
    print(f"\n{'=' * 72}\nSUMMARY (numbers from this run)\n{'-' * 72}")
    print(f"1. Skewed target : MAE {r1['mae_naive']:.4f} -> {r1['mae_guided']:.4f} "
          f"({r1['rel_improvement_pct']:+.1f}%); tails-dominate warning "
          f"{ {True:'held',False:'did not hold',None:'n/a'}[r1['held_naive']] } "
          f"naive vs { {True:'held',False:'did not hold',None:'n/a'}[r1['held_guided']] } guided")
    print(f"2. Duplicate leak: CV accuracy {r2['acc_old']:.4f} (0.1 splitter) -> {r2['acc_new']:.4f} "
          f"(0.2, duplicate-aware; {r2['gap_points']:+.2f} pts inflation removed); de-duplicated {r2['acc_clean']:.4f}")
    print(f"3. Target leak   : R^2 {r3['r2_leak']:.4f} -> {r3['r2_honest']:.4f} "
          f"({r3['gap_points']:+.1f} pts inflation)")
