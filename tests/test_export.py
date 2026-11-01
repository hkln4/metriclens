import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_validate

import metriclens as ml


def test_export_runs_with_string_labels_and_positive_class(messy_df):
    rep = ml.analyze(messy_df, "y", holdout=0.2, answers={"metric_priority": "miss_costly", "leakage.leak": "no",
                                                          "id_like.customer_id": "identifier"})
    ex = rep.export(LogisticRegression(max_iter=1000))
    dev, _ = rep.holdout(messy_df)
    res = cross_validate(ex.model, dev[ex.feature_columns], dev.y, cv=ex.cv, scoring=ex.scoring,
                         error_score="raise")
    assert 0 <= res["test_recall"].mean() <= 1
    assert rep.metric_plan.positive_label == "yes"
    assert {"ts", "note", "customer_id", "leak"}.isdisjoint(ex.feature_columns)


def test_unverified_conditional_decisions_are_excluded(messy_df):
    rep = ml.analyze(messy_df, "y")
    ex = rep.export()
    ids = {d["id"] for d in ex.excluded}
    assert {"d.power_transform", "d.target_encode"} <= ids
    assert all(d["reason"] for d in ex.excluded)
    assert any("unconfirmed" in a or "unanswered" in a for a in ex.assumptions)
    assert "EXPORTED MODEL RECIPE" in ex.describe()


def test_export_with_groups_column(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 5)
    rep = ml.analyze(df, "y", groups="g", holdout=0.2)
    ex = rep.export(Ridge())
    dev, _ = rep.holdout(df)
    assert ex.groups == "g"
    cross_validate(ex.model, dev[ex.feature_columns], dev.y, cv=ex.cv, groups=dev.g, scoring=ex.scoring,
                   error_score="raise")


def test_preprocessor_only_without_estimator(reg_df):
    ex = ml.analyze(reg_df, "y").export()
    Xt = ex.preprocessor.fit_transform(reg_df[ex.feature_columns])
    assert Xt.shape[0] == len(reg_df) and not np.isnan(Xt).any()


def test_export_handles_missing_and_nullable_ints():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": pd.array(rng.integers(0, 50, 800), dtype="Int64"),
                       "c": rng.choice(["x", "y", None], 800), "y": rng.normal(size=800)})
    df.loc[:100, "a"] = pd.NA
    rep = ml.analyze(df, "y")
    ex = rep.export(Ridge())
    cross_validate(ex.model, df[ex.feature_columns], df.y, cv=ex.cv, error_score="raise")


def _verified(rep, did):
    from metriclens.enums import DecisionStatus
    from metriclens.frozen import evolve

    decs = [evolve(d, status=DecisionStatus.VERIFIED) if d.id == did else d for d in rep.decisions]
    new = evolve(rep, decisions=decs)
    new._estimator = rep._estimator
    return new


def test_estimator_steps_are_not_silently_dropped_without_estimator():
    from fixtures import skewed_target

    rep = _verified(ml.analyze(skewed_target(), "y", task="regression"), "d.log_target")
    ex = rep.export()
    assert ex.model is None
    assert "d.log_target" in {d["id"] for d in ex.needs_estimator}
    assert "d.log_target" not in {d["id"] for d in ex.included}
    ex2 = rep.export(Ridge())
    assert type(ex2.model).__name__ == "TransformedTargetRegressor"
    assert ex2.model_kind.startswith("TransformedTargetRegressor")
    assert "d.log_target" in {d["id"] for d in ex2.included}


def test_class_weight_on_estimator_without_it_is_excluded_not_included(imb_df):
    from sklearn.neighbors import KNeighborsClassifier

    rep = _verified(ml.analyze(imb_df, "y"), "d.class_weight")
    ex = rep.export(KNeighborsClassifier())
    assert "d.class_weight" not in {d["id"] for d in ex.included}
    assert any(d["id"] == "d.class_weight" and d["status"] == "not_applicable" for d in ex.excluded)
