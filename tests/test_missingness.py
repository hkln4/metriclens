"""Missing values: evidence that changes a decision, not a generic 'ask how it went missing'."""
import numpy as np
import pandas as pd
import pytest

import prefit as pf
from prefit.enums import DecisionClass, DecisionStatus


def _structural(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    tenure[:30] = 0
    monthly = rng.uniform(20, 110, n).round(2)
    total = (tenure * monthly).astype(float)
    total[tenure == 0] = np.nan
    churn = (rng.random(n) < 0.35 - 0.004 * tenure).astype(int)
    return pd.DataFrame({"tenure": tenure, "monthly": monthly, "total": total, "churn": churn})


def _informative(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(size=n) > 0.5).astype(int)
    lab = rng.normal(size=n)
    lab[rng.random(n) < np.where(y == 1, 0.30, 0.05)] = np.nan   # a test ordered more often for positives
    return pd.DataFrame({"x": x, "lab": lab, "z": rng.normal(size=n), "y": y})


def test_structural_missingness_asks_whether_missing_means_zero():
    df = _structural()
    rep = pf.analyze(df, "churn", task="classification")
    sv = rep.details.profile.columns["total"].missingness["structural"]
    assert sv["column"] == "tenure" and sv["value"] == 0 and sv["coverage"] == 1.0 and sv["precision"] == 1.0
    d = rep.decision("d.fill_zero.total")
    assert d.cls is DecisionClass.CONTEXTUAL and d.status is DecisionStatus.AWAITING_USER
    assert "'tenure' = 0" in d.why_short and d.alternatives
    q = next(q for q in rep.questions if q.id == "q.missing_is_zero.total")
    assert q.keyword == "missing_is_zero" and q.column == "total"
    # no separate review item: the structural question carries the evidence
    assert not any(x.id.startswith("d.missing_signal.") for x in rep.decisions)
    yes = rep.answer(missing_is_zero={"total": True})
    assert yes.decision("d.fill_zero.total").status is DecisionStatus.CONFIRMED
    built = yes.export()
    assert any(x["id"] == "d.fill_zero.total" for x in built.included)
    X = built.preprocessor.fit_transform(df[built.feature_columns])
    assert np.isfinite(X).all()
    no = rep.answer(missing_is_zero={"total": False})
    assert no.decision("d.fill_zero.total").status is DecisionStatus.DECLINED


def test_informative_missingness_is_a_review_item_with_the_rates():
    rep = pf.analyze(_informative(), "y", task="classification")
    t = rep.details.profile.columns["lab"].missingness["target"]
    assert t["p_value"] < 0.01 and t["missing"] > t["observed"] + 0.2
    d = rep.decision("d.missing_signal.lab")
    assert d.cls is DecisionClass.REVIEW and "%" in d.why_short and "permutation p" in d.why_short
    assert any("leak" in a for a in d.alternatives)
    assert "fig.missing_target.lab" in d.figure_ids
    # MCAR columns raise nothing
    assert not any(x.id in ("d.missing_signal.z", "d.missing_signal.x") for x in rep.decisions)


def test_no_generic_missingness_review_without_evidence():
    from fixtures import CASES

    case = next(c for c in CASES if c.name == "high_missing")
    rep = pf.analyze(case.make(), case.target, task=case.task)
    assert not any(d.id == "d.missingness_review" for d in rep.decisions)
    assert any(d.id == "d.drop_mostly_missing" for d in rep.decisions)


def _mar(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a * 0.9 + rng.normal(size=n) * 0.3                         # predictable from a
    b[rng.random(n) < 1 / (1 + np.exp(-2 * a)) * 0.4] = np.nan       # missing more often when a is large (MAR)
    y = a + np.nan_to_num(b) * 0.5 + rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=n), "y": y})


def test_model_imputation_is_a_conditional_decision_with_alternatives():
    rep = pf.analyze(_mar(), "y", task="regression")
    mi = rep.details.profile.columns["b"].missingness
    assert mi["predictable"]["column"] == "a" and mi["features"]["column"] == "a"
    d = rep.decision("d.impute_model")
    assert d.cls is DecisionClass.CONDITIONAL and d.step.kind == "iterative_impute"
    assert len(d.alternatives) == 3 and any("NaN" in a for a in d.alternatives)


@pytest.mark.slow
def test_verify_tests_model_imputation_against_median():
    from sklearn.linear_model import Ridge

    df = _mar()
    rep = pf.analyze(df, "y", task="regression", estimator=Ridge(),
                     answers={"metric_priority": "typical_error"})
    rep = rep.verify(budget=pf.Budget(n_repeats=1))
    e = next(x for x in rep.experiments if x.decision_id == "d.impute_model")
    assert e.family == "user" and e.outcome
    assert rep.decision("d.impute_model").state == "tested"


def test_missingness_is_measured_on_small_counts_but_not_on_tiny_ones():
    df = _structural()
    df.loc[30:33, "monthly"] = np.nan                                  # 4 rows: below the minimum of 5
    rep = pf.analyze(df, "churn", task="classification")
    assert rep.details.profile.columns["monthly"].missingness == {}
    assert rep.details.profile.columns["total"].missingness["n_missing"] == 30
