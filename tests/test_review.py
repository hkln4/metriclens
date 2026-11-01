import numpy as np
import pandas as pd

import metriclens as ml


def _skew_report():
    from fixtures import skewed_target

    return ml.analyze(skewed_target(), "y", task="regression")


def test_regression_tails_hold_against_gaussian_reference():
    rep = _skew_report()
    rng = np.random.default_rng(1)
    y = rng.exponential(2.0, 2000)
    p = y + rng.normal(0, 0.05, 2000)
    p[:20] += 50
    rv = ml.review(y, p, rep)
    c = next(c for c in rv.checks if c.finding_id == "f.target_skew")
    assert c.held is True and "Gaussian" in c.reference


def test_regression_tails_do_not_hold_for_gaussian_errors():
    rep = _skew_report()
    rng = np.random.default_rng(2)
    y = rng.exponential(2.0, 4000)
    rv = ml.review(y, y + rng.normal(0, 1, 4000), rep)
    assert next(c for c in rv.checks if c.finding_id == "f.target_skew").held is False


def test_classification_majority_predictor(imb_df):
    rep = ml.analyze(imb_df, "y")
    y = imb_df.y.to_numpy()
    rv = ml.review(y, np.zeros_like(y), rep, y_score=np.random.default_rng(0).random(len(y)))
    assert rv.metrics["minority_recall"] == 0.0
    assert next(c for c in rv.checks if c.finding_id == "f.class_imbalance").held is True
    beat = next(c for c in rv.checks if "majority" in c.hypothesis)
    assert beat.held is False
    assert rv.thresholds and "roc_auc" in rv.metrics
    assert {f.id for f in rv.figures} >= {"fig.review.confusion", "fig.review.pr"}


def test_unverified_assumptions_are_listed(imb_df):
    rep = ml.analyze(imb_df, "y")
    y = imb_df.y.to_numpy()
    rv = ml.review(y, y, rep)
    ids = {u["id"] for u in rv.unverified}
    assert "d.metric" in ids and "d.class_weight" in ids
    assert "never verified" in rv.to_text()


def test_segments_and_html(reg_df):
    y = reg_df.y.to_numpy()
    p = y + np.where(reg_df.cat == "a", 3.0, 0.1)
    rv = ml.review(y, p, segments=reg_df.cat, task="regression")
    assert rv.segments[0]["value"] == "a"
    html = rv.to_html()
    assert 'id="review"' in html and "fig.review.segments" in html


def test_review_without_report():
    rv = ml.review(pd.Series([1.0, 2, 3, 4, 5]), [1.1, 2, 2.8, 4.2, 10.0], task="regression")
    assert rv.metrics["MAE"] > 0 and rv.checks and not rv.unverified
