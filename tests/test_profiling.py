import numpy as np
import pandas as pd
import pytest

from metriclens.enums import ColumnKind, Task
from metriclens.profiling.kinds import infer_kind
from metriclens.profiling.stats import excess_kurtosis, single_feature_score, skewness, wilson_ci
from metriclens.profiling.task_inference import infer_task

rng = np.random.default_rng(0)
N = 500


@pytest.mark.parametrize("series,kind", [
    (pd.Series(rng.normal(size=N), name="x"), ColumnKind.NUMERIC),
    (pd.Series(np.arange(N), name="user_id"), ColumnKind.ID_LIKE),
    (pd.Series(rng.permutation(N) + 1000, name="row"), ColumnKind.ID_LIKE),           # sequential ints
    (pd.Series(rng.integers(0, 10**7, N), name="income"), ColumnKind.NUMERIC),          # near-unique, not an id
    (pd.Series(rng.choice(list("abc"), N), name="c"), ColumnKind.CATEGORICAL),
    (pd.Series(rng.integers(0, 2, N), name="flag"), ColumnKind.BOOLEAN),
    (pd.Series([True, False] * (N // 2), name="b"), ColumnKind.BOOLEAN),
    (pd.Series(pd.date_range("2020", periods=N), name="d"), ColumnKind.DATETIME),
    (pd.Series([f"2021-01-{i % 28 + 1:02d}" for i in range(N)], name="ds"), ColumnKind.DATETIME),
    (pd.Series([str(i) for i in range(N)], name="numstr"), ColumnKind.ID_LIKE),
    (pd.Series([f"a long piece of free text number {i} here" for i in range(N)], name="t"), ColumnKind.TEXT),
    (pd.Series(["k"] * N, name="k"), ColumnKind.CONSTANT),
    (pd.Series([np.nan] * N, name="e"), ColumnKind.CONSTANT),
])
def test_kind_inference(series, kind):
    assert infer_kind(series)[0] is kind


def test_skew_and_kurtosis_match_pandas():
    v = rng.exponential(size=1000)
    assert skewness(v) == pytest.approx(pd.Series(v).skew())
    assert excess_kurtosis(v) == pytest.approx(pd.Series(v).kurt())


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_ci(20, 100)
    assert lo < 0.2 < hi and 0 <= lo and hi <= 1


@pytest.mark.parametrize("y,task,ambiguous", [
    (pd.Series([0, 1] * 50), Task.CLASSIFICATION, False),
    (pd.Series(["a", "b", "c"] * 30), Task.CLASSIFICATION, False),
    (pd.Series(rng.normal(size=200)), Task.REGRESSION, False),
    (pd.Series(rng.integers(0, 5, 200)), Task.CLASSIFICATION, True),
    (pd.Series(rng.poisson(8, 2000)), Task.REGRESSION, False),
])
def test_task_inference(y, task, ambiguous):
    t, ev = infer_task(y)
    assert t is task and ev["ambiguous"] is ambiguous


def test_single_feature_score_separates_signal_from_noise():
    y = pd.Series(rng.integers(0, 2, 2000))
    leak = y.astype(float) + rng.normal(0, 0.01, 2000)
    noise = pd.Series(rng.normal(size=2000))
    assert single_feature_score(leak, y, True, True, 0) > 0.95
    assert single_feature_score(noise, y, True, True, 0) < 0.15
    ids = pd.Series([f"id{i}" for i in range(2000)])
    assert single_feature_score(ids, y, False, True, 0) < 0.15  # memorization does not score
