"""Tabular profiler: measures, never concludes.

Runs on the development rows only. Every statistic that a rule compares against
a threshold gets an uncertainty estimate — a bootstrap interval (rescaled to the
full sample size) or a Wilson interval — so the rule engine can tell "clearly
past the threshold" from "near it". Infinite values are counted and excluded
from distribution statistics.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..config import ProfileOptions
from ..enums import ColumnKind, Task
from . import stats as st
from .kinds import infer_kind
from .profile import ColumnProfile, DataProfile, DatasetProfile, Relations, TargetProfile


def profile_tabular(
    df: pd.DataFrame,
    target: str,
    task: Task,
    task_inferred: bool,
    task_evidence: dict[str, Any],
    roles: dict[str, str],
    options: ProfileOptions,
    seed: int,
    n_rows_input: int,
    sampling: dict[str, Any],
    holdout: dict[str, Any],
) -> DataProfile:
    feature_cols = [c for c in df.columns if c != target and c not in roles]
    y = df[target]
    classification = task is Task.CLASSIFICATION
    columns = {str(c): _profile_column(df[c], y, classification, options, seed) for c in feature_cols}
    return DataProfile(
        task=task, task_inferred=task_inferred,
        target=_profile_target(y, task, task_inferred, task_evidence, options, seed),
        dataset=_profile_dataset(df, target, feature_cols, columns, n_rows_input),
        columns=columns,
        relations=_profile_relations(df, target, task, columns, options.max_corr_features),
        roles=roles, sampling=sampling, holdout=holdout,
    )


def _datetime_as_number(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = dt.notna()
    out[ok] = dt[ok].astype("int64").astype(float)
    return out


def _finite(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    return v.where(np.isfinite(v))


def _profile_column(s: pd.Series, y: pd.Series, classification: bool, options: ProfileOptions,
                    seed: int) -> ColumnProfile:
    n = len(s)
    n_missing = int(s.isna().sum())
    kind, reason = infer_kind(s)
    non_null = s.dropna()
    cardinality = int(non_null.nunique())
    stats: dict[str, Any] = {}
    ci: dict[str, tuple[float, float]] = {}
    if kind is ColumnKind.NUMERIC:
        raw = pd.to_numeric(non_null, errors="coerce").astype(float)
        n_inf = int(np.isinf(raw).sum())
        finite = _finite(s)
        stats = st.numeric_stats(finite)
        stats["inf_count"] = n_inf
        v = finite.dropna().to_numpy()
        boot = st.bootstrap_ci(v, st.skewness, options.n_bootstrap, options.bootstrap_rows, seed)
        if boot:
            ci["skew"] = boot
    elif kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        show = (not options.redact_values) and cardinality <= options.max_labelled_cardinality
        stats = st.categorical_stats(s, show_labels=show)
    elif kind is ColumnKind.DATETIME:
        dt = _datetime_as_number(s).dropna()
        if len(dt):
            stats = {"min": str(pd.Timestamp(int(dt.min()))), "max": str(pd.Timestamp(int(dt.max()))),
                     "monotonic_in_row_order": bool(dt.is_monotonic_increasing)}
    elif kind in (ColumnKind.TEXT, ColumnKind.ID_LIKE):
        stats = {"mean_length": st.r(float(non_null.astype(str).str.len().mean()), 2)}

    association = None
    if kind not in (ColumnKind.CONSTANT, ColumnKind.TEXT):
        if kind is ColumnKind.DATETIME:
            x, numeric = _datetime_as_number(s), True
        elif kind is ColumnKind.NUMERIC:
            x, numeric = _finite(s), True
        else:
            numeric = kind is ColumnKind.BOOLEAN or (kind is ColumnKind.ID_LIKE and pd.api.types.is_numeric_dtype(s))
            x = s
        association = st.single_feature_score(x, y, numeric, classification, seed)

    return ColumnProfile(
        name=str(s.name), kind=kind, kind_reason=reason, dtype=str(s.dtype),
        missing_fraction=st.r(n_missing / n if n else 0.0), missing_ci=st.wilson_ci(n_missing, n),
        cardinality=cardinality, unique_ratio=st.r(cardinality / len(non_null) if len(non_null) else 0.0),
        stats=stats, ci=ci, association=association,
    )


def _profile_target(
    y: pd.Series, task: Task, inferred: bool, evidence: dict[str, Any],
    options: ProfileOptions, seed: int,
) -> TargetProfile:
    n = len(y)
    n_missing = int(y.isna().sum())
    nb, br = options.n_bootstrap, options.bootstrap_rows
    stats: dict[str, Any] = {}
    ci: dict[str, tuple[float, float]] = {}
    if task is Task.REGRESSION:
        finite = _finite(y)
        v = finite.dropna().to_numpy()
        stats = st.numeric_stats(finite)
        stats["inf_count"] = int(np.isinf(pd.to_numeric(y.dropna(), errors="coerce").astype(float)).sum())
        for key, fn in (("skew", st.skewness), ("kurtosis", st.excess_kurtosis),
                        ("outlier_fraction", st.iqr_outlier_fraction)):
            b = st.bootstrap_ci(v, fn, nb, br, seed)
            if b:
                ci[key] = b
        z = st.wilson_ci(int((v == 0).sum()), len(v))
        if z:
            ci["zero_fraction"] = z
    elif task is Task.CLASSIFICATION:
        vc = y.dropna().astype(str).value_counts()
        k = len(vc)
        stats = {
            "n_classes": k,
            "class_counts": {str(c): int(v) for c, v in vc.items()},
            "imbalance_ratio": st.r(float(vc.max() / vc.min())),
            "minority_fraction": st.r(float(vc.min() / vc.sum())),
            "majority_fraction": st.r(float(vc.max() / vc.sum())),
            "minority_class": str(vc.idxmin()),
        }
        codes = pd.Categorical(y.dropna().astype(str), categories=list(vc.index)).codes

        def ratio(sample: np.ndarray) -> float:
            c = np.bincount(sample, minlength=k)
            lo, hi = int(np.min(c)), int(np.max(c))
            return hi / lo if lo > 0 else float("inf")

        b = st.bootstrap_ci(codes.astype(int), ratio, nb, br, seed)
        if b:
            ci["imbalance_ratio"] = b
    return TargetProfile(name=str(y.name), task=task, task_inferred=inferred, task_evidence=evidence,
                         n_missing=n_missing, missing_fraction=st.r(n_missing / n if n else 0.0),
                         stats=stats, ci=ci)


def _profile_dataset(
    df: pd.DataFrame, target: str, feature_cols: list, columns: dict[str, ColumnProfile],
    n_rows_input: int,
) -> DatasetProfile:
    n = len(df)
    cols = [*feature_cols, target]
    # hash rows directly (no string conversion): fast on wide/long frames
    row_hash = pd.util.hash_pandas_object(df[cols], index=False)
    dup_mask = row_hash.duplicated()
    conflicting = 0
    if feature_cols:
        feat_hash = pd.util.hash_pandas_object(df[feature_cols], index=False).to_numpy()
        tgt_hash = pd.util.hash_pandas_object(df[target], index=False).to_numpy()
        per_key = pd.DataFrame({"k": feat_hash, "t": tgt_hash}).groupby("k")["t"].nunique()
        conflicting = int((per_key > 1).sum())
    kind_counts: dict[str, int] = {}
    for cp in columns.values():
        kind_counts[cp.kind.value] = kind_counts.get(cp.kind.value, 0) + 1

    missing_cols = sorted((c for c, p in columns.items() if p.missing_fraction > 0),
                          key=lambda c: -columns[c].missing_fraction)
    cooc: dict[str, Any] = {}
    if len(missing_cols) >= 3:
        top = missing_cols[:15]
        m = df[top].isna().to_numpy()
        inter = m.T.astype(int) @ m.astype(int)
        union = m.sum(0)[:, None] + m.sum(0)[None, :] - inter
        jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        cooc = {"columns": top, "jaccard": [[st.r(x, 3) for x in row] for row in jac]}

    return DatasetProfile(
        n_rows=n, n_rows_input=n_rows_input, n_columns=df.shape[1], n_features=len(feature_cols),
        duplicate_rows=int(dup_mask.sum()), duplicate_fraction=st.r(float(dup_mask.mean()) if n else 0.0),
        conflicting_duplicate_groups=conflicting, kind_counts=kind_counts,
        missing_columns=missing_cols, missing_cooccurrence=cooc,
    )


def _profile_relations(
    df: pd.DataFrame, target: str, task: Task, columns: dict[str, ColumnProfile], max_corr: int,
) -> Relations:
    numeric = [c for c, p in columns.items() if p.kind is ColumnKind.NUMERIC]
    pearson: dict[str, float] = {}
    if task is Task.REGRESSION and pd.api.types.is_numeric_dtype(df[target]):
        yv = _finite(df[target])
        for c in numeric:
            corr = _finite(df[c]).corr(yv)
            if pd.notna(corr):
                pearson[c] = st.r(abs(float(corr)), 4)
    corr_cols: list[str] = []
    corr_matrix: list[list[float | None]] = []
    pairs: list[dict[str, Any]] = []
    if len(numeric) >= 2:
        fin = df[numeric].apply(_finite)
        corr_cols = sorted(numeric, key=lambda c: -(columns[c].association or 0.0))[:max_corr]
        corr_matrix = [[st.r(x, 3) for x in row] for row in fin[corr_cols].corr(method="spearman").to_numpy()]
        if len(numeric) <= 200:
            full = fin.corr(method="spearman").to_numpy()
            for i, j in zip(*np.triu_indices(len(numeric), k=1)):
                v = full[i, j]
                if np.isfinite(v) and abs(v) >= 0.95:
                    pairs.append({"a": numeric[i], "b": numeric[j], "spearman": st.r(float(v), 4)})
    return Relations(corr_columns=corr_cols, corr_matrix=corr_matrix,
                     high_corr_pairs=sorted(pairs, key=lambda p: -abs(p["spearman"]))[:50], pearson_target=pearson)
