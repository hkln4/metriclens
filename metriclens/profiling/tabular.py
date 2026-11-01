"""Tabular profiler: measures, never concludes.

Runs on the development rows only (the holdout has already been removed by the
input guard). Every statistic that a rule compares against a threshold gets an
uncertainty estimate — a bootstrap interval or a Wilson interval — so the rule
engine can tell "clearly past the threshold" from "near it".
"""

from __future__ import annotations

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
    columns = {str(c): _profile_column(df[c], options, seed) for c in feature_cols}
    target_profile = _profile_target(y, task, task_inferred, task_evidence, options, seed)

    classification = task is Task.CLASSIFICATION
    for name, cp in columns.items():
        if cp.kind in (ColumnKind.CONSTANT, ColumnKind.TEXT):
            continue
        numeric = cp.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN) or (
            cp.kind is ColumnKind.ID_LIKE and pd.api.types.is_numeric_dtype(df[name]))
        if cp.kind is ColumnKind.DATETIME:
            x = _datetime_as_number(df[name])
            numeric = True
        else:
            x = df[name]
        if task is not Task.UNKNOWN:
            cp.association = st.single_feature_score(x, y, numeric, classification, seed)

    return DataProfile(
        task=task, task_inferred=task_inferred, target=target_profile,
        dataset=_profile_dataset(df, target, feature_cols, columns, n_rows_input),
        columns=columns,
        relations=_profile_relations(df, target, task, columns, options.max_corr_features),
        roles=roles, sampling=sampling, holdout=holdout,
    )


def _datetime_as_number(s: pd.Series) -> pd.Series:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    ok = dt.notna()
    out[ok] = dt[ok].astype("int64").astype(float)
    return out


def _profile_column(s: pd.Series, options: ProfileOptions, seed: int) -> ColumnProfile:
    n = len(s)
    n_missing = int(s.isna().sum())
    kind, reason = infer_kind(s)
    non_null = s.dropna()
    cardinality = int(non_null.nunique())
    cp = ColumnProfile(
        name=str(s.name), kind=kind, kind_reason=reason, dtype=str(s.dtype),
        missing_fraction=st.r(n_missing / n if n else 0.0),
        missing_ci=st.wilson_ci(n_missing, n),
        cardinality=cardinality,
        unique_ratio=st.r(cardinality / len(non_null) if len(non_null) else 0.0),
    )
    if kind is ColumnKind.NUMERIC:
        cp.stats = st.numeric_stats(s)
        v = non_null.astype(float).to_numpy()
        ci = st.bootstrap_ci(v, st.skewness, options.n_bootstrap, options.bootstrap_rows, seed)
        if ci:
            cp.ci["skew"] = ci
    elif kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        show = (not options.redact_values) and cardinality <= options.max_labelled_cardinality
        cp.stats = st.categorical_stats(s, show_labels=show)
    elif kind is ColumnKind.DATETIME:
        dt = _datetime_as_number(s).dropna()
        if len(dt):
            cp.stats = {"min": str(pd.Timestamp(int(dt.min()))), "max": str(pd.Timestamp(int(dt.max()))),
                        "monotonic_in_row_order": bool(dt.is_monotonic_increasing)}
    elif kind in (ColumnKind.TEXT, ColumnKind.ID_LIKE):
        cp.stats = {"mean_length": st.r(float(non_null.astype(str).str.len().mean()), 2)}
    return cp


def _profile_target(
    y: pd.Series, task: Task, inferred: bool, evidence: dict[str, Any],
    options: ProfileOptions, seed: int,
) -> TargetProfile:
    n = len(y)
    n_missing = int(y.isna().sum())
    tp = TargetProfile(name=str(y.name), task=task, task_inferred=inferred, task_evidence=evidence,
                       n_missing=n_missing, missing_fraction=st.r(n_missing / n if n else 0.0))
    nb, br = options.n_bootstrap, options.bootstrap_rows
    if task is Task.REGRESSION:
        v = y.dropna().astype(float).to_numpy()
        tp.stats = st.numeric_stats(y)
        for key, fn in (("skew", st.skewness), ("kurtosis", st.excess_kurtosis),
                        ("outlier_fraction", st.iqr_outlier_fraction)):
            ci = st.bootstrap_ci(v, fn, nb, br, seed)
            if ci:
                tp.ci[key] = ci
        zeros = int((v == 0).sum())
        z_ci = st.wilson_ci(zeros, len(v))
        if z_ci:
            tp.ci["zero_fraction"] = z_ci
    elif task is Task.CLASSIFICATION:
        vc = y.dropna().astype(str).value_counts()
        k = len(vc)
        tp.stats = {
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

        ci = st.bootstrap_ci(codes.astype(int), ratio, nb, br, seed)
        if ci:
            tp.ci["imbalance_ratio"] = ci
    return tp


def _profile_dataset(
    df: pd.DataFrame, target: str, feature_cols: list, columns: dict[str, ColumnProfile],
    n_rows_input: int,
) -> DatasetProfile:
    n = len(df)
    hashable = df[[*feature_cols, target]].astype(str) if feature_cols else df[[target]].astype(str)
    dup_mask = hashable.duplicated()
    conflicting = 0
    if feature_cols:
        feats = hashable.drop(columns=[target])
        grouped = hashable.assign(_k=pd.util.hash_pandas_object(feats, index=False).to_numpy())
        per_key = grouped.groupby("_k")[target].nunique()
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
    rel = Relations()
    numeric = [c for c, p in columns.items() if p.kind is ColumnKind.NUMERIC]
    if task is Task.REGRESSION and pd.api.types.is_numeric_dtype(df[target]):
        yv = df[target].astype(float)
        for c in numeric:
            corr = df[c].astype(float).corr(yv)
            if pd.notna(corr):
                rel.pearson_target[c] = st.r(abs(float(corr)), 4)
    if len(numeric) >= 2:
        ranked = sorted(numeric, key=lambda c: -(columns[c].association or 0.0))[:max_corr]
        corr = df[ranked].astype(float).corr(method="spearman")
        rel.corr_columns = ranked
        rel.corr_matrix = [[st.r(x, 3) for x in row] for row in corr.to_numpy()]
        pairs = []
        full = df[numeric].astype(float).corr(method="spearman").to_numpy() if len(numeric) <= 200 else None
        if full is not None:
            iu = np.triu_indices(len(numeric), k=1)
            for i, j in zip(*iu):
                v = full[i, j]
                if np.isfinite(v) and abs(v) >= 0.95:
                    pairs.append({"a": numeric[i], "b": numeric[j], "spearman": st.r(float(v), 4)})
        rel.high_corr_pairs = sorted(pairs, key=lambda p: -abs(p["spearman"]))[:50]
    return rel
