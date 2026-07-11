"""Infer the learning task from the target column.

Heuristic; result is always flagged (task_inferred=True) for confirmation. The
tricky case is a numeric target: numeric does NOT imply regression (0/1 labels
are numeric too). We decide by cardinality + whether values are whole numbers.
"""
from __future__ import annotations
from typing import Any
import pandas as pd
from ..enums import Task

_MAX_DISCRETE_CLASSES = 20  # numeric integer target with <= this many distinct values -> classification

def infer_task(y: pd.Series) -> tuple[Task, dict[str, Any]]:
    """Return (task, evidence). Evidence explains *why*, for confirmation."""
    non_null = y.dropna()
    n_unique = int(non_null.nunique())
    evidence: dict[str, Any] = {"n_unique": n_unique, "dtype": str(y.dtype)}

    if len(non_null) == 0:
        return Task.UNKNOWN, evidence

    if pd.api.types.is_bool_dtype(non_null) or not pd.api.types.is_numeric_dtype(non_null):
        evidence["reason"] = "non-numeric or boolean target"
        return Task.CLASSIFICATION, evidence

    integer_like = bool((non_null == non_null.round()).all())
    evidence["integer_like"] = integer_like

    if n_unique == 2:
        evidence["reason"] = "binary target"
        return Task.CLASSIFICATION, evidence

    if integer_like and n_unique <= _MAX_DISCRETE_CLASSES:
        evidence["reason"] = f"integer target with {n_unique} distinct values (<= {_MAX_DISCRETE_CLASSES})"
        return Task.CLASSIFICATION, evidence
    evidence["reason"] = "continuous or high-cardinality numeric target"
    return Task.REGRESSION, evidence