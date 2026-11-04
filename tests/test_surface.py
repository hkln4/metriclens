"""Public surface: immutability, revisions, readable reprs, tables."""
import warnings

import numpy as np
import pandas as pd
import pydantic
import pytest

import prefit as pf
from prefit.profiling.stats import bootstrap_ci, skewness


def test_reports_are_deeply_immutable(imb_df):
    rep = pf.analyze(imb_df, "y")
    with pytest.raises(pydantic.ValidationError):
        rep.metric_plan = None
    with pytest.raises(pydantic.ValidationError):
        rep.decisions[0].status = "applied"
    assert isinstance(rep.decisions, tuple) and isinstance(rep.decisions[0].evidence_ids, tuple)
    with pytest.raises(TypeError, match="immutable"):
        rep.details.evidence["x"] = 1
    with pytest.raises(TypeError, match="immutable"):
        rep.details.profile.target.stats["class_counts"]["0"] = 5
    with pytest.raises(AttributeError):
        rep.decisions.append(None)


def test_every_change_is_a_revision(imb_df):
    rep = pf.analyze(imb_df, "y")
    r1 = rep.answer({"metric_priority": "symmetric"})
    r2 = r1.verify(imb_df, budget=pf.Budget(max_experiments=0))
    assert [h["action"] for h in r2.details.manifest.history] == ["analyze", "answer", "verify"]
    assert (rep.details.manifest.revision, r1.details.manifest.revision, r2.details.manifest.revision) == (0, 1, 2)


def test_reprs_are_short(imb_df):
    rep = pf.analyze(imb_df, "y")
    assert len(repr(rep)) < 500 and "Next:" in repr(rep)
    for obj in (rep.decisions[0], rep.details.findings[0], rep.questions[0], rep.metric_plan, rep.details):
        assert len(repr(obj)) < 300 and repr(obj).startswith("<")
    assert "<b>Run next</b>" in rep._repr_html_() and "report.answer(" in rep._repr_html_()


def test_decisions_table(imb_df):
    t = pf.analyze(imb_df, "y").decisions_table()
    assert isinstance(t, pd.DataFrame)
    assert list(t.columns) == ["stage", "status", "class", "decision", "why", "key evidence", "do", "experiment",
                               "standard", "note", "id"]
    assert t.set_index("id").loc["d.fit_in_fold", "standard"]
    row = t[t.id == "d.class_weight"].iloc[0]
    assert "Imbalance ratio" in row["key evidence"]


def test_no_v0_aliases(imb_df):
    rep = pf.analyze(imb_df, "y")
    for name in ("recommended_metrics", "avoid", "validation_strategy", "preprocessing_notes", "split", "findings"):
        assert not hasattr(rep, name)


def test_bootstrap_interval_refers_to_the_full_sample():
    rng = np.random.default_rng(0)
    n = 100_000
    v = rng.normal(size=n)
    lo, hi = bootstrap_ci(v, skewness, 200, 5_000, 0)
    analytic = 2 * 1.96 * np.sqrt(6 / n)          # SE of skewness for normal data
    assert 0.6 * analytic < hi - lo < 1.6 * analytic
    assert lo < skewness(v) < hi


def test_infinite_values_are_a_finding_not_warnings():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000), "y": rng.normal(size=1000)})
    df.loc[:20, "a"] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rep = pf.analyze(df, "y")
    f = rep.details.finding("f.infinite_values")
    assert "a" in f.columns
    assert rep.details.profile.columns["a"].stats["inf_count"] == 21
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_validate

    ex = rep.export(Ridge())
    cross_validate(ex.model, df[ex.feature_columns], df.y, cv=ex.cv, error_score="raise")


def test_scale_is_an_export_option_not_a_procedural_claim(reg_df):
    rep = pf.analyze(reg_df, "y")
    assert "scal" not in rep.decision("d.fit_in_fold").action.lower().replace("scaling is", "")
    assert "StandardScaler" in str(rep.export().preprocessor)
    assert "StandardScaler" not in str(rep.export(scale=False).preprocessor)


def test_verification_uses_the_export_splitter(imb_df):
    rep = pf.analyze(imb_df, "y", verify=True, budget=pf.Budget(max_experiments=1))
    assert "StratifiedKFold" in rep.decision("d.cv").step.params["name"]
    assert rep.baseline["cv"].startswith("RepeatedStratifiedKFold") and "export splitter" in rep.baseline["cv"]
    assert all(e.cv == rep.baseline["cv"] for e in rep.experiments)


def test_skewed_title_separates_near_threshold():
    from prefit.rules.catalog import skewed_features  # noqa: F401

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.lognormal(sigma=1.2, size=3000), "b": rng.normal(size=3000),
                       "y": rng.normal(size=3000)})
    title = pf.analyze(df, "y").decision("d.power_transform").title
    assert "skewed" in title and title.startswith("Power-transform 1 skewed")


def test_association_is_named_2fold_cv(reg_df):
    rep = pf.analyze(reg_df, "y")
    assert "2-fold cross-validated" in rep.details.profile.relations.association_method
    assert "held-out score" not in rep.to_html()
