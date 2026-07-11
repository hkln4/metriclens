"""Second pass: review() judges whether the init hypothesis held against real
errors, for both regression (tail domination) and classification (imbalance)."""
import numpy as np
import pandas as pd

import metriclens as ml


def _skewed_regression_roadmap():
    rng = np.random.default_rng(0)
    y = rng.exponential(2.0, size=2000)
    df = pd.DataFrame({"f1": rng.normal(size=2000), "target": y})
    return ml.analyze(df, target="target", use_llm=False)


def test_regression_hypothesis_holds_when_tails_dominate():
    roadmap = _skewed_regression_roadmap()
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.exponential(2.0, size=n)
    # mostly tiny errors, a few huge ones -> tails dominate, hypothesis holds
    y_pred = y_true + rng.normal(0, 0.05, n)
    y_pred[:20] += 50.0
    rep = ml.review(y_true, y_pred, roadmap=roadmap)
    assert rep.task == "regression"
    check = next(c for c in rep.checks
                 if c.finding_id in {"target_skew", "heavy_tailed_target"})
    assert check.held is True
    assert rep.metrics["RMSE"] > rep.metrics["MAE"]


def test_regression_hypothesis_fails_when_errors_even():
    roadmap = _skewed_regression_roadmap()
    rng = np.random.default_rng(2)
    n = 2000
    y_true = rng.exponential(2.0, size=n)
    y_pred = y_true + rng.normal(0, 1.0, n)  # uniform-ish errors, no tail spike
    rep = ml.review(y_true, y_pred, roadmap=roadmap)
    check = next(c for c in rep.checks
                 if c.finding_id in {"target_skew", "heavy_tailed_target"})
    assert check.held is False


def test_classification_imbalance_hypothesis_holds_for_majority_predictor():
    rng = np.random.default_rng(3)
    n = 2000
    df = pd.DataFrame({"f1": rng.normal(size=n),
                       "label": (rng.random(n) < 0.05).astype(int)})
    roadmap = ml.analyze(df, target="label", use_llm=False)
    y_true = df["label"].to_numpy()
    y_pred = np.zeros_like(y_true)  # trivial majority-class predictor
    rep = ml.review(y_true, y_pred, roadmap=roadmap, task="classification")
    assert rep.task == "classification"
    assert rep.metrics["accuracy"] >= 0.8
    assert rep.metrics["minority_recall"] == 0.0
    check = next(c for c in rep.checks if c.finding_id == "class_imbalance")
    assert check.held is True


def test_review_without_roadmap_still_reports():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.0, 2.8, 4.2, 10.0])
    rep = ml.review(y_true, y_pred, task="regression")
    assert rep.metrics["MAE"] > 0
    assert rep.checks  # an observation check is always produced
