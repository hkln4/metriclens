"""Input guard: validate what the user passed, then reserve the holdout.

Two jobs, both before any statistic is computed:

1. Fail loudly on inputs that would otherwise produce a silently empty report
   (a misspelled target, an unknown keyword, a groups array of the wrong length).
2. Split off a holdout that *no* metriclens decision ever sees. The profile, the
   rules and the experiments all run on the development part only, so the
   holdout stays an honest final check — the EDA itself cannot leak into it.
"""

from __future__ import annotations

import difflib
from typing import Any

import numpy as np
import pandas as pd

from .enums import Task

MIN_HOLDOUT_ROWS = 50
MIN_DEV_ROWS = 200


def validate_inputs(
    df: Any,
    target: str,
    groups: Any = None,
    time_col: str | None = None,
) -> pd.Series | None:
    """Raise on bad input; return the groups as a Series aligned to `df` (or None)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"metriclens expects a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if df.columns.duplicated().any():
        dupes = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise ValueError(f"Duplicate column names are not supported: {dupes}.")
    if not isinstance(target, str) or not target:
        raise ValueError("`target` must be the name of the target column.")
    _require_column(df, target, "target")
    if df[target].notna().sum() == 0:
        raise ValueError(f"Target column '{target}' has no non-missing values.")
    if time_col is not None:
        _require_column(df, time_col, "time_col")
        if time_col == target:
            raise ValueError("`time_col` cannot be the target column.")
        parsed = _as_datetime(df[time_col])
        if parsed.notna().mean() < 0.95:
            raise ValueError(f"`time_col` '{time_col}' could not be parsed as datetimes for >5% of rows.")

    if groups is not None and time_col is not None:
        raise ValueError(
            "groups= and time_col= together are not supported in v1: no splitter can keep groups whole and respect "
            "time order at once without assumptions about your data. Pass one of them (time_col if predictions run "
            "forward in time), or build a custom splitter and evaluate with it yourself.")
    if groups is None:
        return None
    if isinstance(groups, str):
        _require_column(df, groups, "groups")
        if groups == target:
            raise ValueError("`groups` cannot be the target column.")
        g = df[groups]
    else:
        arr = np.asarray(groups).ravel()
        if len(arr) != len(df):
            raise ValueError(f"`groups` has {len(arr)} values but the DataFrame has {len(df)} rows.")
        g = pd.Series(arr, index=df.index)
    if g.isna().any():
        raise ValueError("`groups` contains missing values; every row needs a group.")
    return g


def _require_column(df: pd.DataFrame, name: str, role: str) -> None:
    if name in df.columns:
        return
    cols = [str(c) for c in df.columns]
    close = difflib.get_close_matches(str(name), cols, n=3)
    hint = f" Did you mean {close}?" if close else ""
    raise KeyError(f"{role} column '{name}' not found in the DataFrame.{hint}")


def _as_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


# --------------------------------------------------------------------------- #
# Holdout
# --------------------------------------------------------------------------- #

def reserve_holdout(
    df: pd.DataFrame,
    target: str,
    task: Task,
    fraction: float | None,
    groups: pd.Series | None,
    time_col: str | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (dev_positions, holdout_positions, info). Positions are 0-based row
    positions into `df`. The split respects time order, groups and class ratios."""
    n = len(df)
    all_pos = np.arange(n)
    if not fraction:
        return all_pos, np.array([], dtype=int), {"used": False, "reason": "disabled by the user (holdout=None)."}
    if not 0 < fraction <= 0.5:
        raise ValueError("`holdout` must be a fraction in (0, 0.5], or None to disable it.")
    n_hold = int(round(n * fraction))
    if n_hold < MIN_HOLDOUT_ROWS or n - n_hold < MIN_DEV_ROWS:
        return all_pos, np.array([], dtype=int), {
            "used": False,
            "reason": (f"too few rows ({n}) for a {fraction:.0%} holdout "
                       f"(needs >= {MIN_HOLDOUT_ROWS} holdout and >= {MIN_DEV_ROWS} dev rows). "
                       "A final performance estimate must come from nested cross-validation."),
        }

    rng = np.random.default_rng(seed)
    if time_col is not None:
        order = np.argsort(_as_datetime(df[time_col]).to_numpy(), kind="stable")
        hold = np.sort(order[n - n_hold:])
        method = f"last {fraction:.0%} of rows by '{time_col}'"
    elif groups is not None:
        uniq = pd.unique(groups.to_numpy())
        rng.shuffle(uniq)
        sizes = groups.value_counts()
        chosen, total = [], 0
        for g in uniq:
            if total >= n_hold:
                break
            chosen.append(g)
            total += int(sizes[g])
        hold = np.flatnonzero(groups.isin(chosen).to_numpy())
        method = f"{len(chosen)} whole groups (~{fraction:.0%} of rows; not stratified by class)"
    elif task is Task.CLASSIFICATION:
        y = df[target].to_numpy()
        hold_parts = []
        for cls in pd.unique(y[pd.notna(y)]):
            members = np.flatnonzero(y == cls)
            rng.shuffle(members)
            k = int(round(len(members) * fraction))
            if len(members) >= 2:
                k = max(1, k)
            hold_parts.append(members[:k])
        hold = np.sort(np.concatenate(hold_parts)) if hold_parts else np.array([], dtype=int)
        method = f"stratified random {fraction:.0%}"
    else:
        perm = rng.permutation(n)
        hold = np.sort(perm[:n_hold])
        method = f"random {fraction:.0%}"

    mask = np.zeros(n, dtype=bool)
    mask[hold] = True
    return all_pos[~mask], all_pos[mask], {
        "used": True, "fraction": fraction, "n_holdout": int(mask.sum()),
        "n_dev": int((~mask).sum()), "method": method, "seed": seed,
    }


def sample_rows(
    df: pd.DataFrame, max_rows: int, seed: int, stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic (stratified when possible) subsample above `max_rows`."""
    n = len(df)
    if n <= max_rows:
        return df, {"sampled": False, "n_rows": n}
    rng = np.random.default_rng(seed)
    if stratify is not None:
        parts = []
        frac = max_rows / n
        for _, idx in stratify.groupby(stratify, dropna=False).indices.items():
            idx = np.asarray(idx)
            k = max(1, int(round(len(idx) * frac)))
            parts.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        pos = np.sort(np.concatenate(parts))
    else:
        pos = np.sort(rng.choice(n, size=max_rows, replace=False))
    return df.iloc[pos], {"sampled": True, "n_rows": int(len(pos)), "of_rows": n, "seed": seed}
