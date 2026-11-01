"""Data partitions: ml.split, the internal holdout, fingerprints, external holdouts."""
import numpy as np
import pandas as pd
import pytest

import metriclens as ml
from metriclens.fingerprint import FingerprintMismatch


def test_analyze_does_not_split_by_default(reg_df):
    rep = ml.analyze(reg_df, "y")
    assert rep.details.profile.holdout["used"] is False
    assert rep.details.profile.dataset.n_rows == len(reg_df)
    with pytest.raises(ValueError, match="no internal holdout"):
        rep.holdout(reg_df)


@pytest.mark.parametrize("strategy", ["stratified", "random"])
def test_split_strategies(imb_df, strategy):
    dev, test = ml.split(imb_df, "y", test_size=0.25, strategy=strategy, random_state=1)
    assert len(dev) + len(test) == len(imb_df) and set(dev.index).isdisjoint(test.index)
    if strategy == "stratified":
        assert abs(test.y.mean() - imb_df.y.mean()) < 0.01


def test_split_by_group_and_time(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 7, t=pd.date_range("2020", periods=len(reg_df), freq="h"))
    dev, test = ml.split(df, "y", strategy="group", groups="g")
    assert set(dev.g).isdisjoint(test.g)
    dev, test = ml.split(df.sample(frac=1, random_state=0), "y", strategy="time", time_col="t")
    assert test.t.min() > dev.t.max()


def test_split_rejects_bad_requests(reg_df):
    with pytest.raises(ValueError, match="discrete target"):
        ml.split(reg_df, "y", strategy="stratified")
    with pytest.raises(ValueError, match="groups="):
        ml.split(reg_df, "y", strategy="group")
    with pytest.raises(ValueError):
        ml.split(reg_df, "y", strategy="whatever")


def test_split_matches_internal_holdout(imb_df):
    _, test = ml.split(imb_df, "y", test_size=0.2, random_state=0)
    rep = ml.analyze(imb_df, "y", holdout=0.2, seed=0)
    _, hold = rep.holdout(imb_df)
    assert list(test.index) == list(hold.index)


@pytest.mark.parametrize("change,msg", [
    (lambda d: d.sample(frac=1, random_state=0), "different order"),
    (lambda d: d.assign(f1=d.f1 * 2), "have different values"),
    (lambda d: d.iloc[:-5], "shape"),
    (lambda d: d.drop(columns="f2"), "column names"),
    (lambda d: d.reset_index(drop=True).set_axis(range(1, len(d) + 1)), "index differs"),
])
def test_fingerprint_rejects_a_different_frame(reg_df, change, msg):
    rep = ml.analyze(reg_df, "y", holdout=0.2)
    other = change(reg_df)
    with pytest.raises(FingerprintMismatch, match=msg):
        rep.holdout(other)
    with pytest.raises(FingerprintMismatch, match="Nothing was re-split"):
        rep.verify(other, budget=ml.Budget(max_experiments=0))


def test_verify_rejects_the_test_frame(imb_df):
    dev, test = ml.split(imb_df, "y")
    rep = ml.analyze(dev, "y")
    with pytest.raises(FingerprintMismatch):
        rep.verify(test)


def test_holdout_is_rederived_after_json_roundtrip(tmp_path, imb_df):
    rep = ml.analyze(imb_df, "y", holdout=0.2)
    path = tmp_path / "r.json"
    rep.to_json(str(path))
    loaded = ml.load_report(str(path))
    assert loaded._holdout_positions is None
    assert list(loaded.holdout(imb_df)[1].index) == list(rep.holdout(imb_df)[1].index)
    with pytest.raises(FingerprintMismatch):
        loaded.holdout(imb_df.iloc[::-1])


def test_register_external_holdout_records_metadata_only(imb_df):
    dev, test = ml.split(imb_df, "y")
    rep = ml.analyze(dev, "y")
    rep2 = rep.register_external_holdout(test, dev_df=dev, name="test set")
    ext = rep2.details.manifest.external_holdout
    assert ext["n_rows"] == len(test) and ext["overlap_rows"] == 0 and ext["profiled"] is False
    assert rep2.details.profile == rep.details.profile                     # nothing was profiled
    assert rep2.details.manifest.revision == rep.details.manifest.revision + 1
    assert "test set" in rep2.decision("d.holdout").title
    assert rep.details.manifest.external_holdout is None                    # old revision untouched


def test_external_overlap_is_reported_without_claiming_leakage(imb_df):
    dev, test = ml.split(imb_df, "y")
    test = pd.concat([test, dev.iloc[:7]])
    rep = ml.analyze(dev, "y")
    with pytest.warns(UserWarning, match="Exact row overlap detected between development and external holdout"):
        rep2 = rep.register_external_holdout(test, dev_df=dev)
    f = rep2.details.finding("f.external_holdout_overlap")
    assert "This may inflate evaluation" in f.message and "leakage" not in f.message.lower()
    assert rep2.details.manifest.external_holdout["overlap_rows"] == 7


def test_external_holdout_needs_the_dev_frame_itself(imb_df):
    dev, test = ml.split(imb_df, "y")
    rep = ml.analyze(dev, "y")
    with pytest.raises(FingerprintMismatch):
        rep.register_external_holdout(test, dev_df=test)


def test_split_columns_stay_out_of_features(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 5)
    rep = ml.analyze(df, "y", groups="g")
    assert "g" not in rep.details.profile.columns
    assert rep.decision("d.split_columns").cls.value == "procedural"
    rep2 = ml.analyze(df, "y", groups="g", split_columns_as_features=True)
    assert "g" in rep2.details.profile.columns
    assert rep2.details.finding("f.split_columns_as_features").severity.value == "warn"


def test_single_class_target_is_rejected():
    df = pd.DataFrame({"a": np.arange(100.0), "y": ["yes"] * 100})
    with pytest.raises(ValueError, match="single value"):
        ml.analyze(df, "y")
