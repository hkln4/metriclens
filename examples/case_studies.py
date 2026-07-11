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
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                              load_diabetes)
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score)

import metriclens as ml


def _stop(msg: str) -> None:
    print(f"\n*** STOP: {msg}\n"
          "A rule that should have fired did not. Reporting instead of patching.")
    sys.exit(1)


def _finding(roadmap, rule_id):
    return next((f for f in roadmap.findings if f.id == rule_id), None)


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
    roadmap = ml.analyze(df, target="MedHouseVal", use_llm=False)
    fnd = _finding(roadmap, "target_skew")
    if fnd is None:
        _stop("case 1: expected 'target_skew' to fire on California Housing.")

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"].to_numpy()
    cv = roadmap.make_sklearn()["cv"]  # KFold recommended by the roadmap

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

    rev_n = ml.review(y, pred_naive, roadmap=roadmap)
    rev_g = ml.review(y, pred_guided, roadmap=roadmap)
    held_n = next(c.held for c in rev_n.checks)
    held_g = next(c.held for c in rev_g.checks)
    verdict = {True: "HELD", False: "did not hold", None: "n/a"}

    _table(
        "CASE 1 — Skewed target (California Housing)",
        f"rule fired: {fnd.id}  evidence={fnd.evidence}",
        [("MAE (orig scale)", f"{mae_n:.4f}", f"{mae_g:.4f}"),
         ("RMSE", f"{rmse_n:.4f}", f"{rmse_g:.4f}"),
         ("R^2", f"{r2_n:.4f}", f"{r2_g:.4f}"),
         ("review: tails dominate", verdict[held_n], verdict[held_g])],
        "naive", "roadmap-guided")
    print(f"\nMAE relative improvement (naive -> guided): {rel:+.1f}%")
    return {"mae_naive": mae_n, "mae_guided": mae_g, "rel_improvement_pct": rel,
            "held_naive": held_n, "held_guided": held_g, "evidence": fnd.evidence}


# --------------------------------------------------------------------------- #
# Case 2 — Duplicate-row leakage (Breast Cancer)
# --------------------------------------------------------------------------- #
def case2() -> dict:
    df = load_breast_cancer(as_frame=True).frame
    rng = np.random.default_rng(0)
    dup = rng.choice(len(df), size=int(0.2 * len(df)), replace=False)
    corrupt = pd.concat([df, df.iloc[dup]], ignore_index=True)

    roadmap = ml.analyze(corrupt, target="target", use_llm=False)
    fnd = _finding(roadmap, "duplicate_rows")
    if fnd is None:
        _stop("case 2: expected 'duplicate_rows' to fire on the corrupted copy.")

    model = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    Xc, yc = corrupt.drop(columns="target"), corrupt["target"]
    acc_naive = cross_val_score(model, Xc, yc, cv=cv, scoring="accuracy").mean()

    clean = corrupt.drop_duplicates().reset_index(drop=True)  # follow the rec
    Xk, yk = clean.drop(columns="target"), clean["target"]
    acc_honest = cross_val_score(model, Xk, yk, cv=cv, scoring="accuracy").mean()
    gap = (acc_naive - acc_honest) * 100

    _table(
        "CASE 2 — Duplicate-row leakage (Breast Cancer)",
        f"rule fired: {fnd.id}  evidence={fnd.evidence}",
        [("CV accuracy", f"{acc_naive:.4f}", f"{acc_honest:.4f}"),
         ("rows", str(len(corrupt)), str(len(clean)))],
        "naive (dupes)", "de-duplicated")
    print(f"\nInflation from duplicate leakage: {gap:+.2f} accuracy points")
    return {"acc_naive": acc_naive, "acc_honest": acc_honest,
            "gap_points": gap, "evidence": fnd.evidence}


# --------------------------------------------------------------------------- #
# Case 3 — Target leakage (Diabetes + injected feature)
# --------------------------------------------------------------------------- #
def case3() -> dict:
    df = load_diabetes(as_frame=True).frame  # target column: 'target'
    rng = np.random.default_rng(1)
    noise = rng.normal(0, df["target"].std() * 0.01, len(df))
    leaked = df.copy()
    leaked["leaky"] = leaked["target"] + noise  # |r| ~ 0.9999

    roadmap = ml.analyze(leaked, target="target", use_llm=False)
    fnd = _finding(roadmap, "target_leakage")
    if fnd is None:
        _stop("case 3: expected 'target_leakage' to fire on the injected feature.")
    if fnd.severity.value != "critical":
        _stop(f"case 3: 'target_leakage' fired but severity={fnd.severity.value}, expected critical.")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_leak = cross_val_score(LinearRegression(), leaked.drop(columns="target"),
                              leaked["target"], cv=cv, scoring="r2").mean()
    r2_honest = cross_val_score(LinearRegression(), df.drop(columns="target"),
                                df["target"], cv=cv, scoring="r2").mean()
    gap = (r2_leak - r2_honest) * 100

    corr = fnd.evidence["leakage_hints"][0]["correlation"]
    _table(
        "CASE 3 — Target leakage (Diabetes + injected feature)",
        f"rule fired: {fnd.id} [{fnd.severity.value}]  |r|={abs(corr):.4f}",
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
    print(f"2. Duplicate leak: accuracy {r2['acc_naive']:.4f} -> {r2['acc_honest']:.4f} "
          f"({r2['gap_points']:+.2f} pts inflation)")
    print(f"3. Target leak   : R^2 {r3['r2_leak']:.4f} -> {r3['r2_honest']:.4f} "
          f"({r3['gap_points']:+.1f} pts inflation)")
