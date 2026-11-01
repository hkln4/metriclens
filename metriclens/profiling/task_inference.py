"""Infer the learning task from the target column.

Heuristic, and always reported as inferred so the user confirms it. The hard case
is an integer target: 0/1 labels and small counts look identical. Up to
`MAX_DISCRETE_CLASSES` distinct integers we guess classification but mark the
inference ambiguous; above it we guess regression (counts).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..enums import Task

MAX_DISCRETE_CLASSES = 10


def infer_task(y: pd.Series) -> tuple[Task, dict[str, Any]]:
    """Return (task, evidence). Evidence explains *why*, for confirmation."""
    non_null = y.dropna()
    n_unique = int(non_null.nunique())
    evidence: dict[str, Any] = {"n_unique": n_unique, "dtype": str(y.dtype), "ambiguous": False}

    if len(non_null) == 0:
        evidence["reason"] = "target has no non-missing values"
        return Task.UNKNOWN, evidence
    if pd.api.types.is_bool_dtype(non_null) or not pd.api.types.is_numeric_dtype(non_null):
        evidence["reason"] = "non-numeric or boolean target"
        return Task.CLASSIFICATION, evidence
    if n_unique == 2:
        evidence["reason"] = "binary target"
        return Task.CLASSIFICATION, evidence

    integer_like = bool((non_null == non_null.round()).all())
    evidence["integer_like"] = integer_like
    if integer_like and n_unique <= MAX_DISCRETE_CLASSES:
        evidence["ambiguous"] = True
        evidence["reason"] = (f"integer target with {n_unique} distinct values (<= {MAX_DISCRETE_CLASSES}): "
                              "could be class labels or small counts")
        return Task.CLASSIFICATION, evidence
    evidence["reason"] = ("integer target with many distinct values (counts)" if integer_like
                          else "continuous numeric target")
    return Task.REGRESSION, evidence
