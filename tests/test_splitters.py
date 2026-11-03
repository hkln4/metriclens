"""DuplicateAwareKFold: exact duplicates never straddle a fold boundary."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import prefit as pf
from prefit.splitters import DuplicateAwareKFold, duplicate_units


@pytest.fixture
def dup_xy():
    from fixtures import duplicates

    df = duplicates()
    return df.drop(columns="y"), df.y


@pytest.mark.parametrize("stratify", [True, False])
def test_no_unit_on_both_sides_of_a_fold(dup_xy, stratify):
    X, y = dup_xy
    units = duplicate_units(X, y)
    cv = DuplicateAwareKFold(n_splits=5, n_repeats=3, stratify=stratify, random_state=0)
    splits = list(cv.split(X, y))
    assert len(splits) == cv.get_n_splits() == 15
    for train, test in splits:
        assert not set(units[train]) & set(units[test])
        assert len(train) + len(test) == len(X)
    for r in range(3):  # each repeat is a partition of the rows
        tests = np.concatenate([te for _, te in splits[5 * r:5 * r + 5]])
        assert sorted(tests) == list(range(len(X)))


def test_repeats_differ_and_seed_is_stable(dup_xy):
    X, y = dup_xy
    a = [te for _, te in DuplicateAwareKFold(5, 3, True, 0).split(X, y)]
    b = [te for _, te in DuplicateAwareKFold(5, 3, True, 0).split(X, y)]
    c = [te for _, te in DuplicateAwareKFold(5, 3, True, 1).split(X, y)]
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not np.array_equal(a[0], a[5]) and not np.array_equal(a[5], a[10])
    assert not np.array_equal(a[0], c[0])


def test_stratification_is_approximately_kept(dup_xy):
    X, y = dup_xy
    for _, test in DuplicateAwareKFold(5, 2, stratify=True, random_state=0).split(X, y):
        assert abs(y.iloc[test].mean() - y.mean()) < 0.01
        assert abs(len(test) - len(X) / 5) <= 2


def test_imbalanced_classes_reach_every_fold():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=1000)})
    y = pd.Series((np.arange(1000) < 40).astype(int))
    X, y = pd.concat([X, X.iloc[:20]]), pd.concat([y, y.iloc[:20]])  # half the minority duplicated
    for _, test in DuplicateAwareKFold(5, 2, stratify=True, random_state=0).split(X, y):
        assert 6 <= y.iloc[test].sum() <= 14


def test_works_with_numpy_and_sklearn(dup_xy):
    X, y = dup_xy
    Xn = X.drop(columns="cat").to_numpy()
    res = cross_validate(LogisticRegression(), Xn, y.to_numpy(),
                         cv=DuplicateAwareKFold(4, 2, stratify=True, random_state=0))
    assert len(res["test_score"]) == 8


def test_rejects_impossible_requests():
    with pytest.raises(ValueError, match="at least 2"):
        DuplicateAwareKFold(n_splits=1)
    X = pd.DataFrame({"a": [1, 1, 2, 2]})
    with pytest.raises(ValueError, match="distinct rows"):
        list(DuplicateAwareKFold(n_splits=3).split(X))
    with pytest.raises(ValueError, match="needs y"):
        list(DuplicateAwareKFold(n_splits=2, stratify=True).split(X))


def test_export_and_verify_use_it_only_with_duplicates(dup_xy, imb_df):
    X, y = dup_xy
    df = X.assign(y=y)
    rep = pf.analyze(df, "y", verify=True, budget=pf.Budget(max_experiments=0))
    ex = rep.export(LogisticRegression())
    assert isinstance(ex.cv, DuplicateAwareKFold) and ex.cv.stratify and ex.cv.n_repeats == 1
    assert "exact duplicate rows (features + target) in one fold" in ex.describe()
    assert rep.baseline["cv"].startswith("DuplicateAwareKFold(n_splits=5, n_repeats=2")
    assert rep.baseline["protocol"] == "repeated_kfold"
    clean = pf.analyze(imb_df, "y").export()
    assert type(clean.cv).__name__ == "StratifiedKFold"
    assert "duplicate" not in clean.describe()


def test_small_data_keeps_repeats_and_regression_is_not_stratified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    df["y"] = df.a + rng.normal(size=500)
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    cv = pf.analyze(df, "y", task="regression").export().cv
    assert isinstance(cv, DuplicateAwareKFold) and cv.n_repeats == 3 and not cv.stratify
