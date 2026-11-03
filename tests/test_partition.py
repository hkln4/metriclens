"""Data partitions: pf.split, the internal holdout, fingerprints, external holdouts."""
import numpy as np
import pandas as pd
import pytest

import prefit as pf
from prefit.fingerprint import FingerprintMismatch


def test_analyze_does_not_split_by_default(reg_df):
    rep = pf.analyze(reg_df, "y")
    assert rep.details.profile.holdout["used"] is False
    assert rep.details.profile.dataset.n_rows == len(reg_df)
    with pytest.raises(ValueError, match="no internal holdout"):
        rep.holdout(reg_df)


@pytest.mark.parametrize("strategy", ["stratified", "random"])
def test_split_strategies(imb_df, strategy):
    dev, test = pf.split(imb_df, "y", test_size=0.25, strategy=strategy, random_state=1)
    assert len(dev) + len(test) == len(imb_df) and set(dev.index).isdisjoint(test.index)
    if strategy == "stratified":
        assert abs(test.y.mean() - imb_df.y.mean()) < 0.01


def test_split_by_group_and_time(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 7, t=pd.date_range("2020", periods=len(reg_df), freq="h"))
    dev, test = pf.split(df, "y", strategy="group", groups="g")
    assert set(dev.g).isdisjoint(test.g)
    dev, test = pf.split(df.sample(frac=1, random_state=0), "y", strategy="time", time_col="t")
    assert test.t.min() > dev.t.max()


def test_split_rejects_bad_requests(reg_df):
    with pytest.raises(ValueError, match="discrete target"):
        pf.split(reg_df, "y", strategy="stratified")
    with pytest.raises(ValueError, match="groups="):
        pf.split(reg_df, "y", strategy="group")
    with pytest.raises(ValueError):
        pf.split(reg_df, "y", strategy="whatever")


def test_split_matches_internal_holdout(imb_df):
    _, test = pf.split(imb_df, "y", test_size=0.2, random_state=0)
    rep = pf.analyze(imb_df, "y", holdout=0.2, seed=0)
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
    rep = pf.analyze(reg_df, "y", holdout=0.2)
    other = change(reg_df)
    with pytest.raises(FingerprintMismatch, match=msg):
        rep.holdout(other)
    with pytest.raises(FingerprintMismatch, match="Nothing was re-split"):
        rep.verify(other, budget=pf.Budget(max_experiments=0))


def test_verify_rejects_the_test_frame(imb_df):
    dev, test = pf.split(imb_df, "y")
    rep = pf.analyze(dev, "y")
    with pytest.raises(FingerprintMismatch):
        rep.verify(test)


def test_holdout_is_rederived_after_json_roundtrip(tmp_path, imb_df):
    rep = pf.analyze(imb_df, "y", holdout=0.2)
    path = tmp_path / "r.json"
    rep.to_json(str(path))
    loaded = pf.load_report(str(path))
    assert loaded._holdout_positions is None
    assert list(loaded.holdout(imb_df)[1].index) == list(rep.holdout(imb_df)[1].index)
    with pytest.raises(FingerprintMismatch):
        loaded.holdout(imb_df.iloc[::-1])


def test_register_external_holdout_records_metadata_only(imb_df):
    dev, test = pf.split(imb_df, "y")
    rep = pf.analyze(dev, "y")
    rep2 = rep.register_external_holdout(test, dev_df=dev, name="test set")
    ext = rep2.details.manifest.external_holdout
    assert ext["n_rows"] == len(test) and ext["overlap_rows"] == 0 and ext["profiled"] is False
    assert rep2.details.profile == rep.details.profile                     # nothing was profiled
    assert rep2.details.manifest.revision == rep.details.manifest.revision + 1
    assert "test set" in rep2.decision("d.holdout").title
    assert rep.details.manifest.external_holdout is None                    # old revision untouched


def test_external_overlap_is_reported_without_claiming_leakage(imb_df):
    dev, test = pf.split(imb_df, "y")
    test = pd.concat([test, dev.iloc[:7]])
    rep = pf.analyze(dev, "y")
    with pytest.warns(UserWarning, match="Exact row overlap detected between development and external holdout"):
        rep2 = rep.register_external_holdout(test, dev_df=dev)
    f = rep2.details.finding("f.external_holdout_overlap")
    assert "This may inflate evaluation" in f.message and "leakage" not in f.message.lower()
    assert rep2.details.manifest.external_holdout["overlap_rows"] == 7


def test_external_holdout_needs_the_dev_frame_itself(imb_df):
    dev, test = pf.split(imb_df, "y")
    rep = pf.analyze(dev, "y")
    with pytest.raises(FingerprintMismatch):
        rep.register_external_holdout(test, dev_df=test)


def test_split_columns_stay_out_of_features(reg_df):
    df = reg_df.assign(g=np.arange(len(reg_df)) // 5)
    rep = pf.analyze(df, "y", groups="g")
    assert "g" not in rep.details.profile.columns
    assert rep.decision("d.split_columns").cls.value == "procedural"
    rep2 = pf.analyze(df, "y", groups="g", split_columns_as_features=True)
    assert "g" in rep2.details.profile.columns
    assert rep2.details.finding("f.split_columns_as_features").severity.value == "warn"


def test_single_class_target_is_rejected():
    df = pd.DataFrame({"a": np.arange(100.0), "y": ["yes"] * 100})
    with pytest.raises(ValueError, match="single value"):
        pf.analyze(df, "y")


# --------------------------------------------------------------------------- #
# Exact duplicates stay on one side
# --------------------------------------------------------------------------- #

def _dup_df():
    from fixtures import duplicates

    return duplicates()  # 3000 rows + 300 exact copies


def _shared_hashes(a, b):
    from prefit.fingerprint import row_hashes

    return set(row_hashes(a)) & set(row_hashes(b))


@pytest.mark.parametrize("strategy", ["stratified", "random"])
def test_split_keeps_exact_duplicates_on_one_side(strategy):
    df = _dup_df()
    dev, test = pf.split(df, "y", test_size=0.2, strategy=strategy, random_state=3)
    assert not _shared_hashes(dev, test)
    assert len(dev) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.2) < 0.01
    if strategy == "stratified":
        assert abs(test.y.mean() - df.y.mean()) < 0.01


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_internal_holdout_keeps_exact_duplicates_on_one_side(task):
    df = _dup_df()
    rep = pf.analyze(df, "y", task=task, holdout=0.2, seed=5)
    dev, hold = rep.holdout(df)
    assert not _shared_hashes(dev, hold)
    h = rep.details.profile.holdout
    assert h["duplicate_groups_kept_together"] == 300
    assert "exact duplicates kept on one side" in h["method"]
    assert "300 set(s) of exact duplicate rows were kept whole" in rep.to_text()


def test_duplicate_aware_split_is_seeded():
    df = _dup_df()
    a = pf.split(df, "y", random_state=11)[1].index
    b = pf.split(df, "y", random_state=11)[1].index
    c = pf.split(df, "y", random_state=12)[1].index
    assert list(a) == list(b) and list(a) != list(c)


@pytest.mark.parametrize("fixture,task,digests", [
    ("imbalanced", "classification", {0: "5435928856ab7ef3", 7: "9e8333213392abdb"}),
    ("clean_classification", "classification", {0: "659e0a3ef141fda4", 7: "b4b71ae79ac7be0c"}),
    ("clean_regression", "regression", {0: "f19d06da0d13bdb8", 7: "a7d606975ae85d5c"}),
])
def test_holdout_without_duplicates_matches_0_1_0(fixture, task, digests):
    """No duplicates -> the row-level path, bit-for-bit what prefit 0.1.0 returned."""
    import hashlib

    import fixtures

    from prefit.enums import Task
    from prefit.guard import reserve_holdout

    df = getattr(fixtures, fixture)()
    for seed, digest in digests.items():
        _, hold, info = reserve_holdout(df, "y", Task(task), 0.2, None, None, seed)
        assert hashlib.sha256(hold.astype(np.int64).tobytes()).hexdigest()[:16] == digest
        assert "duplicate_groups_kept_together" not in info


def test_rederived_holdout_matches_analyze_with_duplicates(tmp_path):
    df = _dup_df()
    rep = pf.analyze(df, "y", holdout=0.2, seed=2)
    path = tmp_path / "r.json"
    rep.to_json(str(path))
    loaded = pf.load_report(str(path))
    assert loaded._holdout_positions is None
    assert list(loaded.holdout(df)[1].index) == list(rep.holdout(df)[1].index)
    assert list(pf.split(df, "y", random_state=2)[1].index) == list(rep.holdout(df)[1].index)


def test_groups_win_over_duplicates_with_a_warning():
    df = _dup_df()
    g = np.arange(len(df)) // 10  # each copy (rows 3000+) falls in a different group than its original
    with pytest.warns(UserWarning, match="exact duplicate rows span more than one of your groups"):
        dev, test = pf.split(df.assign(g=g), "y", strategy="group", groups="g")
    assert set(dev.g).isdisjoint(test.g)


def test_time_split_ignores_duplicates():
    df = _dup_df().assign(t=pd.date_range("2020", periods=3300, freq="h"))
    dev, test = pf.split(df, "y", strategy="time", time_col="t")
    assert test.t.min() > dev.t.max()
