"""Versioned benchmark cases.

Each case is a deterministic dataset generator plus expectations:
  must_fire     rule ids that must produce a finding (trigger fired)
  must_not_fire rule ids that must produce no finding at all
  uncertain     rule ids that must produce a finding marked near-threshold
  no_split_overlap  no row hash may sit on both sides of the internal holdout
                    or of any exported CV fold (expected overlap: 0)
Synthetic cases inject one known hazard into otherwise clean data, so a rule's
false positives show up on the clean cases and its misses on the injected ones.
Real-data cases use datasets bundled with scikit-learn (no download).

Changing a case or an expectation means bumping MANIFEST_VERSION.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MANIFEST_VERSION = "2026.11-4"


@dataclass
class Case:
    name: str
    make: Callable[[], pd.DataFrame]
    target: str
    task: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    must_fire: list[str] = field(default_factory=list)
    must_not_fire: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    no_split_overlap: bool = False
    source: str = "synthetic"


def _base(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.uniform(size=n),
                         "cat": rng.choice(["a", "b", "c", "d"], n)})


def _reg(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    df = _base(n, seed)
    rng = np.random.default_rng(seed + 100)
    df["y"] = 2 * df.f1 - df.f2 + rng.normal(0, 1, n) + 10
    return df


def _cls(n: int = 3000, seed: int = 0, pos: float = 0.5) -> pd.DataFrame:
    df = _base(n, seed)
    rng = np.random.default_rng(seed + 100)
    logit = 1.2 * df.f1 - 0.8 * df.f2 + np.log(pos / (1 - pos))
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def clean_regression() -> pd.DataFrame:
    return _reg()


def clean_classification() -> pd.DataFrame:
    return _cls()


def imbalanced() -> pd.DataFrame:
    return _cls(pos=0.04)


def mild_imbalance_near_threshold() -> pd.DataFrame:
    # ~3:1 on 400 rows: the bootstrap interval of the ratio straddles 3
    rng = np.random.default_rng(7)
    df = _base(400, 7)
    df["y"] = (rng.random(400) < 0.25).astype(int)
    return df


def skewed_target() -> pd.DataFrame:
    df = _reg()
    rng = np.random.default_rng(3)
    df["y"] = np.exp(0.6 * df.f1 + rng.normal(0, 0.6, len(df)))
    return df


def leaky_numeric_regression() -> pd.DataFrame:
    df = _reg()
    rng = np.random.default_rng(4)
    df["leak"] = df.y + rng.normal(0, 0.01, len(df))
    return df


def leaky_categorical_classification() -> pd.DataFrame:
    df = _cls()
    df["status"] = np.where(df.y == 1, "closed_won", "closed_lost")
    return df


def duplicates() -> pd.DataFrame:
    df = _cls()
    return pd.concat([df, df.iloc[:300]], ignore_index=True)


def id_column() -> pd.DataFrame:
    df = _reg()
    df.insert(0, "customer_id", np.arange(100_000, 100_000 + len(df)))
    return df


def integer_feature_not_id() -> pd.DataFrame:
    # near-unique integers that are a measurement, not an identifier
    df = _reg()
    rng = np.random.default_rng(5)
    df["income"] = rng.integers(10_000, 2_000_000, len(df))
    return df


def datetime_column() -> pd.DataFrame:
    df = _reg()
    df["signup"] = pd.date_range("2020-01-01", periods=len(df), freq="h")
    return df


def high_missing() -> pd.DataFrame:
    df = _reg()
    rng = np.random.default_rng(6)
    df.loc[rng.random(len(df)) < 0.6, "f3"] = np.nan
    return df


def high_cardinality() -> pd.DataFrame:
    df = _cls()
    rng = np.random.default_rng(8)
    df["zip"] = [f"z{x}" for x in rng.integers(0, 400, len(df))]
    return df


def count_target() -> pd.DataFrame:
    df = _reg()
    rng = np.random.default_rng(9)
    df["y"] = rng.poisson(np.exp(0.8 + 0.4 * df.f1))
    return df


def ordinal_score() -> pd.DataFrame:
    # a 0-6 rating (Wine Quality style): integers with few values, ordered
    df = _reg()
    df["y"] = np.clip(np.round(df.y - df.y.mean() + 3), 0, 6).astype(int)
    return df


def small_data() -> pd.DataFrame:
    return _reg(n=300)


def zero_inflated() -> pd.DataFrame:
    df = _reg()
    rng = np.random.default_rng(10)
    df["y"] = np.where(rng.random(len(df)) < 0.5, 0.0, np.abs(df.y))
    return df


def constant_column() -> pd.DataFrame:
    df = _reg()
    df["version"] = "v1"
    return df


def _sk(loader: str) -> Callable[[], pd.DataFrame]:
    def make() -> pd.DataFrame:
        from sklearn import datasets

        return getattr(datasets, loader)(as_frame=True).frame
    return make


def numbers_as_text() -> pd.DataFrame:
    # Telco-style: a numeric column stored as text, blank where it does not exist yet (tenure == 0)
    rng = np.random.default_rng(1021)  # not 21: _base(seed=21) draws f1 from that stream
    df = _cls(seed=21)
    tenure = rng.integers(0, 72, len(df))
    monthly = rng.uniform(20, 110, len(df)).round(2)
    total = (tenure * monthly).round(2).astype(str).astype(object)
    total[tenure == 0] = " "
    return df.assign(tenure=tenure, monthly=monthly, total=total)


def informative_missing() -> pd.DataFrame:
    # a lab value recorded far more often for positives (missingness caused by the outcome)
    rng = np.random.default_rng(1022)  # not 22: _base(seed=22) draws f1 from that stream
    df = _cls(seed=22)
    lab = rng.normal(size=len(df))
    lab[rng.random(len(df)) < np.where(df.y == 1, 0.30, 0.05)] = np.nan
    return df.assign(lab=lab)


def random_missing() -> pd.DataFrame:
    # 15% missing completely at random: no missingness rule may fire
    rng = np.random.default_rng(1023)  # not 23: _base(seed=23) draws f1 from that stream
    df = _cls(seed=23)
    v = rng.normal(size=len(df))
    v[rng.random(len(df)) < 0.15] = np.nan
    return df.assign(m=v)


def mar_missing() -> pd.DataFrame:
    # a column predictable from f1 and missing more often when f1 is large
    rng = np.random.default_rng(1024)  # not 24: _base(seed=24) draws f1 from that stream
    df = _reg(seed=24)
    b = df.f1 * 0.9 + rng.normal(size=len(df)) * 0.3
    b[rng.random(len(df)) < 1 / (1 + np.exp(-2 * df.f1)) * 0.4] = np.nan
    return df.assign(b=b)


CASES: list[Case] = [
    Case("clean_regression", clean_regression, "y", "regression",
         must_not_fire=["target_skew", "leakage_hint", "duplicate_rows", "id_like_columns", "high_missingness",
                        "small_data", "constant_columns", "zero_inflation", "heavy_tailed_target"]),
    Case("clean_classification", clean_classification, "y", "classification",
         must_not_fire=["class_imbalance", "leakage_hint", "duplicate_rows", "id_like_columns", "small_data"]),
    Case("imbalanced", imbalanced, "y", "classification", must_fire=["class_imbalance"]),
    Case("mild_imbalance_near_threshold", mild_imbalance_near_threshold, "y", "classification",
         uncertain=["class_imbalance"]),
    Case("skewed_target", skewed_target, "y", "regression", must_fire=["target_skew"]),
    Case("leaky_numeric_regression", leaky_numeric_regression, "y", "regression", must_fire=["leakage_hint"]),
    Case("leaky_categorical_classification", leaky_categorical_classification, "y", "classification",
         must_fire=["leakage_hint"]),
    Case("duplicates", duplicates, "y", "classification", must_fire=["duplicate_rows"]),
    Case("duplicates_split_overlap", duplicates, "y", "classification", kwargs={"holdout": 0.2},
         must_fire=["duplicate_rows"], no_split_overlap=True),
    Case("id_column", id_column, "y", "regression", must_fire=["id_like_columns"], must_not_fire=["leakage_hint"]),
    Case("integer_feature_not_id", integer_feature_not_id, "y", "regression", must_not_fire=["id_like_columns"]),
    Case("datetime_column", datetime_column, "y", "regression", must_fire=["temporal_order"],
         must_not_fire=["id_like_columns"]),
    Case("high_missing", high_missing, "y", "regression", must_fire=["high_missingness"]),
    Case("numbers_as_text_structural", numbers_as_text, "y", "classification",
         must_fire=["structural_missingness"], must_not_fire=["high_cardinality"]),
    Case("informative_missing", informative_missing, "y", "classification",
         must_fire=["informative_missingness"], must_not_fire=["structural_missingness"]),
    Case("random_missing", random_missing, "y", "classification",
         must_not_fire=["informative_missingness", "structural_missingness", "model_imputation"]),
    Case("mar_missing", mar_missing, "y", "regression", must_fire=["model_imputation"],
         must_not_fire=["structural_missingness"]),
    Case("high_cardinality", high_cardinality, "y", "classification", must_fire=["high_cardinality"],
         must_not_fire=["id_like_columns"]),
    # many distinct counts leave no doubt (regression): no question; a 0-6 score could be labels or a score: asked
    Case("count_target_inferred_regression", count_target, "y", None,
         must_not_fire=["class_imbalance", "task_confirmation"]),
    Case("ordinal_score_inferred_ambiguous", ordinal_score, "y", None, must_fire=["task_confirmation"]),
    Case("small_data", small_data, "y", "regression", must_fire=["small_data"]),
    Case("zero_inflated", zero_inflated, "y", "regression", must_fire=["zero_inflation"]),
    Case("constant_column", constant_column, "y", "regression", must_fire=["constant_columns"]),
    Case("sk_breast_cancer", _sk("load_breast_cancer"), "target", "classification",
         must_not_fire=["leakage_hint", "duplicate_rows", "id_like_columns"], source="sklearn:load_breast_cancer"),
    Case("sk_diabetes", _sk("load_diabetes"), "target", "regression",
         must_not_fire=["leakage_hint", "duplicate_rows", "id_like_columns"], source="sklearn:load_diabetes"),
    Case("sk_wine", _sk("load_wine"), "target", "classification",
         must_not_fire=["class_imbalance", "leakage_hint", "duplicate_rows"], source="sklearn:load_wine"),
]
