"""Column-kind inference: what a column *is* for modelling, with the reason.

Every inference carries a human-readable reason so the report can show why a
column was treated as, say, an identifier. Inference is conservative: a column
is only called ID-like or text when the evidence is strong, because those kinds
lead to review warnings.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from ..enums import ColumnKind

NEAR_UNIQUE_RATIO = 0.95
TEXT_MEAN_LENGTH = 30
_ID_NAME = re.compile(r"(^|[_\W])(id|uuid|guid|key|idx|index|code)$|^(id|uuid)[_\W]|[a-z]Id$", re.IGNORECASE)
_DATE_CHARS = re.compile(r"[-/:.T ]")


def infer_kind(s: pd.Series) -> tuple[ColumnKind, str]:
    """Return (kind, reason) for one column."""
    non_null = s.dropna()
    n = len(non_null)
    if n == 0:
        return ColumnKind.CONSTANT, "all values are missing"
    nunique = int(non_null.nunique())
    if nunique <= 1:
        return ColumnKind.CONSTANT, "a single distinct value"
    unique_ratio = nunique / n
    name = str(s.name)

    if pd.api.types.is_bool_dtype(non_null):
        return ColumnKind.BOOLEAN, "boolean dtype"
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return ColumnKind.DATETIME, "datetime dtype"

    if pd.api.types.is_numeric_dtype(non_null):
        if nunique == 2 and set(np.unique(non_null.to_numpy())) <= {0, 1}:
            return ColumnKind.BOOLEAN, "numeric with values {0, 1}"
        if pd.api.types.is_integer_dtype(non_null) and unique_ratio >= NEAR_UNIQUE_RATIO:
            if _ID_NAME.search(name):
                return ColumnKind.ID_LIKE, f"near-unique integers ({unique_ratio:.0%} unique) and an id-like name"
            if _is_sequential(non_null):
                return ColumnKind.ID_LIKE, f"near-unique integers ({unique_ratio:.0%} unique) forming a sequence"
        return ColumnKind.NUMERIC, "numeric dtype"

    # object / string / category
    as_str = non_null.astype(str)
    if _looks_like_datetime(as_str):
        return ColumnKind.DATETIME, "strings that parse as dates"
    mean_len = float(as_str.str.len().mean())
    spaced = float(as_str.str.contains(" ", regex=False).mean())
    if mean_len >= TEXT_MEAN_LENGTH and spaced >= 0.5:
        return ColumnKind.TEXT, f"free text (mean length {mean_len:.0f} characters)"
    if unique_ratio >= NEAR_UNIQUE_RATIO and n >= 20:
        return ColumnKind.ID_LIKE, f"near-unique strings ({unique_ratio:.0%} unique)"
    return ColumnKind.CATEGORICAL, f"{nunique} distinct non-numeric values"


def _is_sequential(s: pd.Series) -> bool:
    v = np.sort(s.to_numpy())
    if len(v) < 20:
        return False
    steps = np.diff(v)
    return bool(np.mean(steps == 1) >= 0.9)


def _looks_like_datetime(s: pd.Series) -> bool:
    sample = s.head(50)
    if not sample.str.contains(_DATE_CHARS).mean() >= 0.9:
        return False
    if sample.str.fullmatch(r"[-+]?\d+(\.\d+)?").any():
        return False  # plain numbers stored as strings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)
