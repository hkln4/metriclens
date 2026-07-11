"""Target-column statistics, branching on task. Pure measurement, no conclusions."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def _outlier_fraction_iqr(s: pd.Series) -> float:
    """Fraction of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((s < low) | (s > high)).mean())

def profile_regression_target(y: pd.Series) -> dict[str, Any]:
    s = y.dropna().astype(float)
    if len(s) == 0:
        return {}
    return {
        "mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()),
        "min": float(s.min()), "max": float(s.max()),
        "skew": float(s.skew()),        # Fisher (0 == normal); RMSE-vs-MAE signal
        "kurtosis": float(s.kurt()),
        "outlier_fraction": _outlier_fraction_iqr(s),
        "zero_fraction": float((s == 0).mean()),
        "negative_fraction": float((s < 0).mean()),
        "n_missing": int(y.isna().sum()),
    }

def profile_classification_target(y: pd.Series) -> dict[str, Any]:
    s = y.dropna()
    counts = s.value_counts()
    if len(counts) == 0:
        return {}
    return {
        "n_classes": len(counts),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "minority_fraction": float(counts.min() / counts.sum()),
        "n_missing": int(y.isna().sum()),
    }