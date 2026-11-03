"""Input guard: validate what the user passed, then reserve the holdout.

Two jobs, both before any statistic is computed:

1. Fail loudly on inputs that would otherwise produce a silently empty report
   (a misspelled target, an unknown keyword, a groups array of the wrong length).
2. Split off a holdout that *no* prefit decision ever sees. The profile, the
   rules and the experiments all run on the development part only, so the
   holdout stays an honest final check — the EDA itself cannot leak into it.
"""

from __future__ import annotations

import difflib
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .enums import Task
from .fingerprint import row_hashes

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
        raise TypeError(f"prefit expects a pandas DataFrame, got {type(df).__name__}.")
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
    n_target = int(df[target].nunique(dropna=True))
    if n_target < 2:
        raise ValueError(f"Target column '{target}' has a single value ({df[target].dropna().iloc[0]!r}); there is "
                         "nothing to learn. Check the target column, or the filter that produced this frame.")
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
    positions into `df`. The split respects time order, groups and class ratios.

    Without groups or time_col, exact duplicate rows (all columns, target included)
    are one unit and land on the same side, so a copy never scores its twin. A
    frame without duplicates takes the plain row-level path, so a seed gives the
    same split as before. With groups, the user's groups win (a warning names
    duplicates that span groups); time_col ignores duplicates."""
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
    units = None
    if time_col is None:
        # a groups column is not part of the row's identity (copies in different groups are still copies)
        own = df.drop(columns=[groups.name]) if groups is not None and groups.name in df.columns else df
        units, distinct = pd.factorize(row_hashes(own))
        if len(distinct) == n:
            units = None  # no duplicates: the row-level path below, unchanged
    extra: dict[str, Any] = {}
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
        if units is not None:
            spread = int((pd.Series(groups.to_numpy()).groupby(units).nunique() > 1).sum())
            if spread:
                warnings.warn(
                    f"{spread} set(s) of exact duplicate rows span more than one of your groups. The holdout keeps "
                    "your groups whole, so these copies can land on both sides of the split.", stacklevel=3)
    elif units is not None:
        hold, extra = _holdout_by_units(df[target].to_numpy(), units, task, fraction, n_hold, rng)
        kind = "stratified random" if task is Task.CLASSIFICATION else "random"
        method = f"{kind} {fraction:.0%}; exact duplicates kept on one side"
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
        "n_dev": int((~mask).sum()), "method": method, "seed": seed, **extra,
    }


def _holdout_by_units(y: np.ndarray, units: np.ndarray, task: Task, fraction: float, n_hold: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Holdout positions drawn as whole units (sets of identical rows).

    Units are shuffled and added until the row count reaches its target: per
    class when stratified (identical rows share the target, so a unit has one
    class), over all rows otherwise. A class never moves to the holdout whole."""
    sizes = np.bincount(units)
    first = np.unique(units, return_index=True)[1]  # one row per unit, in unit order
    pools: list[tuple[np.ndarray, int, bool]] = []
    if task is Task.CLASSIFICATION:
        unit_y = y[first]
        for cls in pd.unique(y[pd.notna(y)]):
            members = np.flatnonzero(unit_y == cls)
            n_rows = int(sizes[members].sum())
            k = int(round(n_rows * fraction))
            if n_rows >= 2:
                k = max(1, k)
            pools.append((members, k, True))
    else:
        pools.append((np.arange(len(sizes)), n_hold, False))
    chosen = []
    for members, k, keep_one in pools:
        members = members.copy()
        rng.shuffle(members)
        if k <= 0 or len(members) == 0:
            continue
        reach = int(np.searchsorted(np.cumsum(sizes[members]), k)) + 1
        chosen.append(members[:min(reach, len(members) - 1 if keep_one else len(members))])
    picked = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    hold = np.flatnonzero(np.isin(units, picked))
    return hold, {"duplicate_groups_kept_together": int((sizes > 1).sum())}


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
