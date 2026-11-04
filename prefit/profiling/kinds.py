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
# values that stand for "missing" in exported text files; compared after strip + lower
PLACEHOLDERS = frozenset({"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing"})
_LEADING_ZERO = re.compile(r"^[-+]?0\d")


def missing_mask(s: pd.Series, kind: ColumnKind | None = None) -> pd.Series:
    """True where a value is missing: NaN/None, and for numbers stored as text
    also blanks and placeholders such as "NA" or "?" (they are read as NaN)."""
    if kind is ColumnKind.NUMERIC and not pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(_blank_to_nan(s), errors="coerce").isna()
    return s.isna()


def to_numeric(s: pd.Series) -> pd.Series:
    """A NUMERIC column as floats: numbers stored as text are parsed, blanks and
    placeholders become NaN, +/-inf is kept (callers decide what to do with it)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(_blank_to_nan(s), errors="coerce").astype(float)


def _blank_to_nan(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    st = s.astype(object).where(s.notna(), None)
    codes, uniq = pd.factorize(st)
    low = np.array([v.strip().lower() in PLACEHOLDERS if isinstance(v, str) else False for v in uniq] + [False])
    return st.where(~low[codes], None)


def numbers_as_text(non_null: pd.Series) -> tuple[pd.Series, int] | None:
    """(parsed numbers, number of placeholder values) when every non-missing value
    of a text column is a number or a missing-value placeholder; else None.

    Values with a leading zero ("02134") are codes, not numbers, and so are
    columns where fewer than half of the values are numbers. Works on the
    distinct values, after rejecting most text columns from a small sample."""
    if len(non_null) == 0 or not (pd.api.types.is_object_dtype(non_null) or pd.api.types.is_string_dtype(non_null)):
        return None
    if not all(isinstance(v, str) for v in non_null.head(1000)):
        return None
    if not _parse_unique(pd.Series(non_null.head(200).unique())):
        return None  # cheap rejection: most text columns fail on their first values
    codes, uniq = pd.factorize(non_null)
    parsed_u = _parse_unique(pd.Series(uniq))
    if parsed_u is None:
        return None
    values, placeholder_u = parsed_u
    placeholder = placeholder_u[codes]
    if (~placeholder).sum() < max(1, len(non_null) // 2):
        return None
    parsed = pd.Series(values[codes], index=non_null.index)[~placeholder]
    return parsed, int(placeholder.sum())


def _parse_unique(u: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """(numbers with NaN at placeholders, placeholder mask) for distinct text values, or None."""
    if not all(isinstance(v, str) for v in u):
        return None
    st = u.str.strip()
    placeholder = st.str.lower().isin(PLACEHOLDERS).to_numpy()
    vals = st[~placeholder]
    if vals.str.match(_LEADING_ZERO).any():
        return None
    parsed = pd.to_numeric(vals, errors="coerce")
    if parsed.isna().any():
        return None
    out = np.full(len(u), np.nan)
    out[~placeholder] = parsed.astype(float).to_numpy()
    return out, placeholder


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
    text_num = numbers_as_text(non_null)
    if text_num is not None:
        parsed, n_placeholder = text_num
        blanks = f"; {n_placeholder} blank/placeholder value(s) read as missing" if n_placeholder else ""
        ints = bool((parsed == parsed.round()).all())
        p_unique = parsed.nunique() / len(parsed)
        if ints and p_unique >= NEAR_UNIQUE_RATIO and (_ID_NAME.search(name) or _is_sequential(parsed)):
            return ColumnKind.ID_LIKE, f"near-unique integers stored as text ({p_unique:.0%} unique)"
        if parsed.nunique() <= 1:
            return ColumnKind.CONSTANT, "a single distinct number (stored as text)"
        return ColumnKind.NUMERIC, f"numbers stored as text{blanks}"
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
