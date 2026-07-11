"""Feature-column statistics: missingness, cardinality, constant/near-unique
flags, plus a numeric leakage *hint* (high |corr| is a flag to inspect, not a
verdict). Classification-target leakage is deferred to a later step."""
from __future__ import annotations
from typing import Any
import pandas as pd

_LEAKAGE_CORR_THRESHOLD = 0.98   # |Pearson r| at/above this -> possible leak
_NEAR_UNIQUE_RATIO = 0.95        # cardinality/n at/above this -> id-like

def profile_features(df: pd.DataFrame, target: str | None, task_is_regression: bool) -> dict[str, Any]:
    n = len(df)
    feature_cols = [c for c in df.columns if c != target]
    columns: dict[str, Any] = {}
    constant_columns: list[str] = []
    near_unique_columns: list[str] = []

    for col in feature_cols:
        s = df[col]
        cardinality = int(s.nunique(dropna=True))
        columns[col] = {
            "dtype": str(s.dtype),
            "cardinality": cardinality,
            "missing_fraction": float(s.isna().mean()) if n else 0.0,
        }
        if cardinality <= 1:
            constant_columns.append(col)
        # Near-unique only matters for identifier-like columns. A continuous
        # float feature naturally has near-unique cardinality and is NOT a
        # leak/uselessness signal, so floats are excluded.
        elif n and not pd.api.types.is_float_dtype(s) and cardinality / n >= _NEAR_UNIQUE_RATIO:
            near_unique_columns.append(col)

    leakage_hints: list[dict[str, Any]] = []
    if target is not None and task_is_regression and target in df.columns:
        y = df[target]
        if pd.api.types.is_numeric_dtype(y):
            for col in feature_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                corr = df[[col, target]].corr().iloc[0, 1]
                if pd.notna(corr) and abs(corr) >= _LEAKAGE_CORR_THRESHOLD:
                    leakage_hints.append({"feature": col, "correlation": float(corr)})

    dup_mask = df.duplicated()  # exact full-row duplicates

    return {
        "columns": columns,
        "constant_columns": constant_columns,
        "near_unique_columns": near_unique_columns,
        "leakage_hints": leakage_hints,
        "max_missing_fraction": max((c["missing_fraction"] for c in columns.values()), default=0.0),
        "duplicate_rows": int(dup_mask.sum()),
        "duplicate_fraction": float(dup_mask.mean()) if n else 0.0,
    }