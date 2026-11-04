"""Column kinds on data as it arrives from CSV exports and databases.

The 0.3 profiler read numbers stored as text (Telco's TotalCharges: blanks for
new customers) as a categorical column with thousands of categories and no
missing values. These cases pin the behaviour down."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hs

import prefit as pf
from prefit.enums import ColumnKind
from prefit.profiling.kinds import infer_kind, missing_mask, to_numeric

rng = np.random.default_rng(0)
N = 400


def _num_text(blank: str, n_blank: int = 11) -> pd.Series:
    v = np.round(rng.uniform(18, 8700, N), 2).astype(str).astype(object)
    v[:n_blank] = blank
    return pd.Series(v, name="TotalCharges")


@pytest.mark.parametrize("blank", [" ", "", "NA", "n/a", "?", "null", "-", "NaN"])
def test_numbers_stored_as_text_are_numeric_with_placeholders_missing(blank):
    s = _num_text(blank)
    kind, reason = infer_kind(s)
    assert kind is ColumnKind.NUMERIC and "stored as text" in reason
    assert int(missing_mask(s, kind).sum()) == 11
    assert to_numeric(s).isna().sum() == 11


@pytest.mark.parametrize("series,kind", [
    (pd.Series([f"{i:05d}" for i in rng.integers(0, 60, N)], name="zip"), ColumnKind.CATEGORICAL),  # codes
    (pd.Series([str(i) for i in range(N)], name="numstr"), ColumnKind.ID_LIKE),       # sequential ids as text
    (pd.Series(rng.choice(["1", "2", "3"], N), name="grade"), ColumnKind.NUMERIC),     # small integers as text
    (pd.Series(["1.5", "x"] * (N // 2), name="mixed"), ColumnKind.CATEGORICAL),        # a real word: not numbers
    (pd.Series(["?"] * N, name="all_placeholder"), ColumnKind.CONSTANT),
    (pd.Series(pd.array(rng.integers(0, 50, N), dtype="Int64"), name="nullable"), ColumnKind.NUMERIC),
    (pd.Series(pd.array(rng.choice(["a", "b"], N), dtype="string"), name="str_dtype"), ColumnKind.CATEGORICAL),
    (pd.Series(pd.Categorical(rng.choice(["lo", "hi"], N)), name="cat_dtype"), ColumnKind.CATEGORICAL),
    (pd.Series(pd.array(rng.uniform(size=N).round(3).astype(str), dtype="string"), name="num_string_dtype"),
     ColumnKind.NUMERIC),
    (pd.Series([1, "a", 2.5, None] * (N // 4), name="object_mixed"), ColumnKind.CATEGORICAL),
])
def test_dirty_kinds(series, kind):
    assert infer_kind(series)[0] is kind


def test_telco_like_frame_end_to_end():
    n = 2000
    tenure = rng.integers(0, 72, n)
    tenure[:15] = 0
    monthly = rng.uniform(20, 110, n).round(2)
    total = (tenure * monthly).round(2).astype(str).astype(object)
    total[tenure == 0] = " "
    churn = np.where(rng.random(n) < 0.3 - 0.003 * tenure, "Yes", "No")
    df = pd.DataFrame({"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn})
    rep = pf.analyze(df, "Churn", task="classification")
    cp = rep.details.profile.columns["TotalCharges"]
    assert cp.kind is ColumnKind.NUMERIC
    assert cp.missing_fraction == pytest.approx((tenure == 0).sum() / n, abs=1e-6)
    assert not any(d.id == "d.target_encode" for d in rep.decisions)        # 0.3 proposed this
    assert "TotalCharges" in rep.details.profile.dataset.missing_columns
    built = rep.export()
    assert "TotalCharges" in built.feature_columns
    X = built.preprocessor.fit_transform(df[built.feature_columns], df["Churn"])
    assert np.isfinite(X).all()


@settings(max_examples=40, deadline=None)
@given(values=hs.lists(hs.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=30, max_size=80),
       blanks=hs.lists(hs.sampled_from([" ", "", "NA", "?"]), min_size=0, max_size=10))
def test_property_numbers_as_text_round_trip(values, blanks):
    """Any column of formatted floats plus placeholder values is numeric, with exactly
    the placeholders missing and the parsed values equal to the originals."""
    text = [repr(v) for v in values] + blanks
    s = pd.Series(text, name="x", dtype=object)
    kind, _ = infer_kind(s)
    if len(set(values)) <= 1:
        assert kind in (ColumnKind.CONSTANT, ColumnKind.NUMERIC)
        return
    assert kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN, ColumnKind.ID_LIKE)
    assert int(missing_mask(s, ColumnKind.NUMERIC).sum()) == len(blanks)
    assert np.allclose(to_numeric(s).dropna().to_numpy(), values)


@settings(max_examples=25, deadline=None)
@given(data=hs.data())
def test_property_analyze_never_crashes_on_mixed_frames(data):
    """Random small frames of mixed, dirty columns: analyze profiles them and the HTML renders."""
    n = data.draw(hs.integers(60, 150))
    gen = np.random.default_rng(data.draw(hs.integers(0, 10_000)))
    cols = {}
    for i in range(data.draw(hs.integers(1, 5))):
        kind = data.draw(hs.sampled_from(["num", "numtext", "cat", "bool", "const", "allnan", "mixed"]))
        if kind == "num":
            v = gen.normal(size=n)
            v[gen.random(n) < 0.2] = np.nan
        elif kind == "numtext":
            v = np.round(gen.normal(size=n), 3).astype(str).astype(object)
            v[gen.random(n) < 0.1] = " "
        elif kind == "cat":
            v = gen.choice(["a", "b", "c", None], n)
        elif kind == "bool":
            v = gen.random(n) < 0.5
        elif kind == "const":
            v = np.full(n, 7)
        elif kind == "allnan":
            v = np.full(n, np.nan)
        else:
            v = gen.choice(np.array([1, "x", 2.5, None], dtype=object), n)
        cols[f"c{i}_{kind}"] = v
    cols["y"] = gen.integers(0, 2, n)
    rep = pf.analyze(pd.DataFrame(cols), "y", task="classification")
    assert rep.to_html().startswith("<!doctype html>")
