"""Does prefit work? Plain flow vs. prefit flow on real datasets, several seeds.

    python benchmarks/real_data.py                 # all datasets (downloads once, ~15 min)
    python benchmarks/real_data.py wine_red telco  # a subset

The one honest question is whether the score a workflow *reports* is the score
the model *gets*. Per dataset and seed:

1. A final evaluation set E (20%) is set aside first, with exact duplicates kept
   on one side, so no row of E has a copy in the rest. Neither flow ever sees E.
2. On the rest R, each flow does what it would do on its own:
   plain   train_test_split (75/25) -> KFold / StratifiedKFold(5) on the train
           part -> fit -> score its own test part. Preprocessing: numeric columns
           (by dtype) median-imputed, text columns with <= 50 distinct values
           one-hot, other text columns dropped.
   prefit  analyze(R, holdout=0.25, estimator) -> answer the task and metric ->
           verify -> export().fit() -> score the internal holdout. Nothing else is
           answered or resolved: open questions stay out of the recipe.
3. Both models are scored on E.

Reported per flow: the CV estimate, its own test score, the score on E, and the
optimism of each estimate = how much better it looks than the score on E (in
metric units, positive = optimistic). Same estimator (HistGradientBoosting,
random_state=0) everywhere; prefit does not choose models.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from real_datasets import load  # noqa: E402

import prefit as pf  # noqa: E402

SEEDS = {"wine_red": 10, "breast_cancer": 10, "telco": 5, "california": 3, "adult": 3}
OUT = ROOT / "benchmarks" / "results"
METRIC = {"typical_error": ("mae", False), "ranking": ("roc_auc", True), "symmetric": ("balanced_accuracy", True)}


def _estimator(task: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    return (HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor)(
        random_state=0)


def _score(name: str, model, X, y, pos) -> float:
    from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score

    if name == "mae":
        return float(mean_absolute_error(y, model.predict(X)))
    if name == "roc_auc":
        classes = list(model.classes_)
        return float(roc_auc_score(y == pos, model.predict_proba(X)[:, classes.index(pos)]))
    return float(balanced_accuracy_score(y, model.predict(X)))


def _plain(R: pd.DataFrame, E: pd.DataFrame, target: str, task: str, metric: str, seed: int, pos) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    X = R.drop(columns=target)
    num = list(X.select_dtypes("number").columns)
    txt = [c for c in X.columns if c not in num and X[c].nunique() <= 50]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                                   OneHotEncoder(handle_unknown="ignore", sparse_output=False)), txt)])
    model = Pipeline([("pre", pre), ("model", _estimator(task))])
    strat = R[target] if task == "classification" else None
    tr, te = train_test_split(R, test_size=0.25, random_state=seed, stratify=strat)
    cv = (StratifiedKFold if task == "classification" else KFold)(5, shuffle=True, random_state=seed)
    scoring = {"mae": "neg_mean_absolute_error", "roc_auc": None, "balanced_accuracy": "balanced_accuracy"}[metric]
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score

        def scoring(est, Xv, yv):
            return roc_auc_score(yv == pos, est.predict_proba(Xv)[:, list(est.classes_).index(pos)])
    cvs = cross_val_score(model, tr.drop(columns=target), tr[target], cv=cv, scoring=scoring)
    cv_mean = float(-cvs.mean()) if metric == "mae" else float(cvs.mean())
    model.fit(tr.drop(columns=target), tr[target])
    own = _score(metric, model, te.drop(columns=target), te[target], pos)
    # exact copies of test rows in the training part (what a plain split allows)
    h_tr = set(pd.util.hash_pandas_object(tr, index=False))
    seen = int(sum(x in h_tr for x in pd.util.hash_pandas_object(te, index=False)))
    return {"cv": cv_mean, "own_test": own, "final": _score(metric, model, E.drop(columns=target), E[target], pos),
            "test_rows_seen_in_training": seen, "n_test": len(te), "n_train": len(tr),
            "distinct_train_rows": int((~tr.duplicated()).sum())}


def _prefit(R: pd.DataFrame, E: pd.DataFrame, target: str, task: str, answer: str, metric: str, seed: int,
            pos) -> dict:
    rep = pf.analyze(R, target, estimator=_estimator(task), holdout=0.25, seed=seed)
    rep = rep.answer(task=task, metric=answer)
    rep = rep.verify(budget=pf.Budget(max_experiments=4, n_repeats=2))
    fitted = rep.export().fit()
    _, hold = rep.holdout()
    cols = list(fitted.feature_names_in_)
    est = rep.cv_estimate()
    dev, _ = rep.holdout()
    return {"cv": float(est["mean"]), "own_test": _score(metric, fitted, hold[cols], hold[target], pos),
            "final": _score(metric, fitted, E[cols], E[target], pos), "n_test": len(hold), "n_train": len(dev),
            "distinct_train_rows": int((~dev.duplicated()).sum()),
            "decisions": {d.id: d.status.value for d in rep.decisions if not d.standard},
            "exported": [x["id"] for x in rep.export().included]}


def run(name: str, seeds: int) -> list[dict]:
    ds = load(name)
    df = ds.frame
    answer = ds.metric
    metric, higher = METRIC[answer]
    pos = None
    if metric == "roc_auc":
        pos = df[ds.target].value_counts().idxmin()
    out = []
    for s in range(seeds):
        t0 = time.perf_counter()
        R, E = pf.split(df, ds.target, test_size=0.2, strategy="stratified" if ds.task == "classification" else
                        "random", random_state=1000 + s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = _plain(R, E, ds.target, ds.task, metric, s, pos)
            pre = _prefit(R, E, ds.target, ds.task, answer, metric, s, pos)
        row = {"dataset": name, "seed": s, "metric": metric, "higher_is_better": higher, "plain": plain,
               "prefit": pre, "seconds": round(time.perf_counter() - t0, 1)}
        for flow in ("plain", "prefit"):
            f = row[flow]
            sign = 1 if higher else -1
            f["optimism_cv"] = sign * (f["cv"] - f["final"])
            f["optimism_own_test"] = sign * (f["own_test"] - f["final"])
        print(f"{name} seed {s}: plain cv {plain['cv']:.4f} test {plain['own_test']:.4f} final {plain['final']:.4f}"
              f" | prefit cv {pre['cv']:.4f} test {pre['own_test']:.4f} final {pre['final']:.4f}"
              f" ({row['seconds']} s)", flush=True)
        out.append(row)
    return out


def summarize(rows: list[dict]) -> str:
    md = ["# Real-data benchmark: plain flow vs prefit flow", "",
          f"prefit {pf.__version__}. Protocol: see the docstring of `benchmarks/real_data.py`. Optimism = how much "
          "better an estimate looks than the score on the final evaluation set E that neither flow saw (metric "
          "units; positive = optimistic). Mean ± standard deviation over seeds.", "",
          "| dataset | metric | seeds | flow | CV optimism | own-test optimism | score on E | test rows seen in "
          "training | distinct training rows |", "|---|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["dataset"] for r in rows):
        rs = [r for r in rows if r["dataset"] == name]
        for flow in ("plain", "prefit"):
            oc = np.array([r[flow]["optimism_cv"] for r in rs])
            ot = np.array([r[flow]["optimism_own_test"] for r in rs])
            fin = np.array([r[flow]["final"] for r in rs])
            seen = (f"{np.mean([r['plain']['test_rows_seen_in_training'] / r['plain']['n_test'] for r in rs]):.1%}"
                    if flow == "plain" else "0 (by construction)")
            md.append(f"| {name} | {rs[0]['metric']} | {len(rs)} | {flow} | {oc.mean():+.4f} ± {oc.std(ddof=1):.4f} | "
                      f"{ot.mean():+.4f} ± {ot.std(ddof=1):.4f} | {fin.mean():.4f} ± {fin.std(ddof=1):.4f} | "
                      f"{seen} | {np.mean([r[flow]['distinct_train_rows'] for r in rs]):,.0f} of "
                      f"{np.mean([r[flow]['n_train'] for r in rs]):,.0f} |")
    md += ["", "Reading it: an optimism near 0 means the reported estimate describes the model. A flow can have a "
           "better score on E and still report it dishonestly, or the reverse; both columns matter. prefit does not "
           "choose the model, so on data without hazards both flows should score the same on E. With exact "
           "duplicates, prefit keeps every copy on one side of the split: its training part then holds fewer "
           "distinct rows than a plain split of the same size (which puts copies on both sides and so scores some "
           "test rows on their own twins); the last column shows that cost.", ""]
    exported: dict[str, dict[str, int]] = {}
    for r in rows:
        for x in r["prefit"]["exported"]:
            exported.setdefault(r["dataset"], {}).setdefault(x, 0)
            exported[r["dataset"]][x] += 1
    md.append("Conditional or contextual steps prefit exported (seeds out of total):")
    for name in dict.fromkeys(r["dataset"] for r in rows):
        n = sum(r["dataset"] == name for r in rows)
        steps = {k: v for k, v in exported.get(name, {}).items() if k not in ("d.task", "d.metric")}
        md.append(f"- {name}: " + (", ".join(f"{k} {v}/{n}" for k, v in steps.items()) or "none"))
    return "\n".join(md) + "\n"


def main() -> int:
    import os

    names = sys.argv[1:] or list(SEEDS)
    if os.environ.get("PREFIT_BENCH_SEEDS"):  # a quick smoke run: PREFIT_BENCH_SEEDS=1
        SEEDS.update({k: int(os.environ["PREFIT_BENCH_SEEDS"]) for k in SEEDS})
    path = OUT / "real_data.json"
    rows = [r for r in (json.loads(path.read_text()) if path.exists() else []) if r["dataset"] not in names]
    for name in names:
        rows += run(name, SEEDS[name])
        path.write_text(json.dumps(rows, indent=1, default=str))
    (OUT / "real_data.md").write_text(summarize(rows))
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
