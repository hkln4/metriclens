import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

import metriclens as ml
from metriclens.enums import DecisionStatus
from metriclens.verify import corrected_ttest, holm


def test_corrected_ttest_and_holm():
    base = np.array([0.80, 0.81, 0.79, 0.80, 0.82, 0.80, 0.81, 0.79, 0.80, 0.81])
    mean, ci, p = corrected_ttest(base, base + 0.05, 0.25)
    assert mean == pytest.approx(0.05) and p < 1e-6 and ci[0] <= mean <= ci[1]
    mean, ci, p = corrected_ttest(base, base[::-1], 0.25)
    assert p > 0.5
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _skewed_reg():
    from fixtures import skewed_target

    return skewed_target()


def test_proxy_results_are_sensitivity_only():
    df = _skewed_reg()
    rep = ml.analyze(df, "y", task="regression", verify=True, budget=ml.Budget(max_experiments=4, n_repeats=1))
    d = rep.decision("d.log_target")
    assert d.experiment_ids
    assert d.status in {DecisionStatus.SUPPORTED_BY_PROXY, DecisionStatus.INCONCLUSIVE, DecisionStatus.REFUTED}
    assert d.status is not DecisionStatus.VERIFIED          # proxies never verify
    assert not any(x["id"] == "d.log_target" for x in rep.export().included)
    assert rep.baseline["families"] and "dummy" in rep.baseline
    assert any("sensitivity" in x for x in rep.limitations)
    assert any(f.id == "fig.exp.log_target" for f in rep.figures)


def test_user_estimator_can_verify():
    df = _skewed_reg()
    rep = ml.analyze(df, "y", task="regression", estimator=Ridge(), verify=True,
                     budget=ml.Budget(max_experiments=4, n_repeats=1))
    d = rep.decision("d.log_target")
    assert all(e.family == "user" for e in rep.experiments)
    assert d.status in {DecisionStatus.VERIFIED, DecisionStatus.INCONCLUSIVE, DecisionStatus.REFUTED}
    included = {x["id"] for x in rep.export().included}
    assert ("d.log_target" in included) == (d.status is DecisionStatus.VERIFIED)


def test_budget_is_enforced(messy_df):
    rep = ml.analyze(messy_df, "y", verify=True, budget=ml.Budget(max_experiments=1, n_repeats=1))
    assert len(rep.experiments) == 1
    assert any(d.status is DecisionStatus.SKIPPED_BUDGET for d in rep.decisions)


def test_estimator_without_class_weight(imb_df):
    rep = ml.analyze(imb_df, "y", estimator=KNeighborsClassifier(), verify=True,
                     budget=ml.Budget(max_experiments=2, n_repeats=1))
    assert rep.decision("d.class_weight").status is DecisionStatus.NOT_APPLICABLE


def test_metric_change_flags_stale_experiments(imb_df):
    rep = ml.analyze(imb_df, "y", estimator=LogisticRegression(max_iter=500), verify=True,
                     budget=ml.Budget(max_experiments=2, n_repeats=1))
    rep2 = rep.answer({"metric_priority": "ranking"})
    assert "re-run report.verify" in rep2.decision("d.class_weight").note


def test_verify_never_touches_holdout(reg_df):
    rep = ml.analyze(reg_df, "y", verify=True, budget=ml.Budget(max_experiments=0))
    assert rep.baseline["n_rows"] == rep.profile.dataset.n_rows
