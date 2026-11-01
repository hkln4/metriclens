import numpy as np
import pandas as pd
import pytest

import metriclens as ml
from metriclens.enums import Task
from metriclens.guard import reserve_holdout


def test_misspelled_target_raises_with_suggestion(reg_df):
    with pytest.raises(KeyError, match="Did you mean"):
        ml.analyze(reg_df, target="yy")


def test_unknown_keyword_is_rejected(reg_df):
    with pytest.raises(TypeError):
        ml.analyze(reg_df, target="y", taks="regression")


def test_non_dataframe_rejected():
    with pytest.raises(TypeError):
        ml.analyze([1, 2, 3], target="x")


def test_groups_wrong_length(reg_df):
    with pytest.raises(ValueError, match="groups"):
        ml.analyze(reg_df, target="y", groups=np.arange(10))


def test_empty_target_rejected():
    df = pd.DataFrame({"a": [1, 2], "y": [np.nan, np.nan]})
    with pytest.raises(ValueError):
        ml.analyze(df, target="y")


def test_holdout_is_stratified_and_disjoint(imb_df):
    dev, hold, info = reserve_holdout(imb_df, "y", Task.CLASSIFICATION, 0.2, None, None, 0)
    assert info["used"] and set(dev).isdisjoint(hold) and len(dev) + len(hold) == len(imb_df)
    p_all = imb_df.y.mean()
    assert abs(imb_df.y.iloc[hold].mean() - p_all) < 0.01


def test_holdout_by_time_takes_the_latest_rows(reg_df):
    df = reg_df.assign(t=pd.date_range("2021", periods=len(reg_df), freq="D")).sample(frac=1, random_state=0)
    dev, hold, info = reserve_holdout(df, "y", Task.REGRESSION, 0.2, None, "t", 0)
    assert df.t.iloc[hold].min() > df.t.iloc[dev].max()


def test_holdout_keeps_groups_whole(reg_df):
    g = pd.Series(np.arange(len(reg_df)) // 10)
    dev, hold, _ = reserve_holdout(reg_df, "y", Task.REGRESSION, 0.2, g, None, 0)
    assert set(g.iloc[dev]).isdisjoint(set(g.iloc[hold]))


def test_small_data_skips_holdout():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=150), "y": rng.normal(size=150)})
    rep = ml.analyze(df, target="y")
    assert rep.profile.holdout["used"] is False
    assert rep.decision("d.holdout").title.startswith("No holdout")


def test_profile_never_sees_holdout_rows(reg_df):
    rep = ml.analyze(reg_df, target="y", holdout=0.25)
    assert rep.profile.dataset.n_rows == len(reg_df) - rep.profile.holdout["n_holdout"]
    dev, hold = rep.split(reg_df)
    assert len(dev) == rep.profile.dataset.n_rows and len(hold) == rep.profile.holdout["n_holdout"]
