"""How values go missing: measurements for every column with missing values.

Measures, never concludes (the rules decide what matters):

target       Does missingness relate to the target? The target in missing vs.
             observed rows (the share of the minority class, or the mean), a
             seeded permutation test, and the single-feature score of the
             missing-value indicator.
structural   Is the column missing exactly when another column takes one value?
             (Telco: TotalCharges is blank exactly where tenure == 0.)
features     Which other feature best separates missing from observed rows: a
             rank AUC for numeric features, the spread of the missing rate across
             categories for categorical ones. Missingness that depends on other
             features (missing at random, given them) biases median imputation.
predictable  For numeric columns, the strongest rank correlation with another
             numeric feature: what a model-based imputer could use.

Runs on the development rows only, on a seeded sample of at most
`ProfileOptions.association_rows` rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..enums import ColumnKind
from . import stats as st
from .kinds import missing_mask
from .profile import ColumnProfile

MIN_MISSING = 5          # fewer missing rows: nothing is measured
MAX_COLUMNS = 30         # the most-missing columns are measured
N_PERMUTATIONS = 499
STRUCTURAL_SHARE = 0.95  # coverage and precision of "missing exactly when <col> == <value>"
MIN_CATEGORY_ROWS = 20

_USABLE = (ColumnKind.NUMERIC, ColumnKind.BOOLEAN, ColumnKind.CATEGORICAL)


def profile_missingness(
    df: pd.DataFrame, target: str, columns: dict[str, ColumnProfile], classification: bool,
    max_rows: int, seed: int, corr: pd.DataFrame | None, redact: bool,
) -> dict[str, dict[str, Any]]:
    cols = [c for c, cp in columns.items()
            if 0 < cp.missing_fraction < 1 and cp.kind not in (ColumnKind.CONSTANT, ColumnKind.TEXT)]
    cols = sorted(cols, key=lambda c: -columns[c].missing_fraction)[:MAX_COLUMNS]
    if not cols:
        return {}
    rows = df[df[target].notna()]
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=seed)
    y = rows[target]
    masks = {c: missing_mask(rows[c], columns[c].kind).to_numpy() for c in cols}
    keep = [c for c in cols if MIN_MISSING <= masks[c].sum() < len(rows) - MIN_MISSING]
    if not keep:
        return {}
    rng = np.random.default_rng(seed)
    yinfo = _target_matrix(y, classification)
    numeric_rel = _numeric_auc(rows, keep, masks, columns)
    out: dict[str, dict[str, Any]] = {}
    for c in keep:
        m = masks[c]
        info: dict[str, Any] = {"n_missing": int(m.sum()), "n_rows": int(len(m))}
        info["target"] = _target_relation(m, yinfo, rng)
        info["indicator_score"] = st.single_feature_score(
            pd.Series(m.astype(float), index=y.index), y, True, classification, seed)
        s = _structural(rows, c, m, columns, redact)
        if s:
            info["structural"] = s
        f = _strongest_feature(rows, c, m, columns, numeric_rel.get(c, {}))
        if f:
            info["features"] = f
        if corr is not None and columns[c].kind is ColumnKind.NUMERIC and c in corr.columns:
            v = corr[c].drop(c, errors="ignore").abs().dropna()
            if len(v):
                info["predictable"] = {"column": str(v.idxmax()), "abs_spearman": st.r(float(v.max()), 4)}
        out[c] = info
    return out


def _target_matrix(y: pd.Series, classification: bool) -> dict[str, Any]:
    if classification:
        lab = y.astype(str)
        vc = lab.value_counts()
        classes = list(vc.index)
        codes = pd.Categorical(lab, categories=classes).codes
        return {"kind": "class_share", "Y": np.eye(len(classes))[codes], "classes": classes,
                "focus": classes.index(str(vc.idxmin()))}
    v = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
    return {"kind": "mean", "Y": v[:, None], "std": float(np.nanstd(v))}


def _target_relation(m: np.ndarray, yi: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Target in missing vs observed rows, with a permutation p-value.

    Statistic: total-variation distance between the class distributions
    (classification) or the absolute difference of means (regression)."""
    Y = yi["Y"]
    n, k = len(m), int(m.sum())
    total = Y.sum(0)

    def stat(sum_missing: np.ndarray, k_: int) -> float:
        a = sum_missing / k_
        b = (total - sum_missing) / (n - k_)
        return float(0.5 * np.abs(a - b).sum()) if yi["kind"] == "class_share" else float(abs(a[0] - b[0]))

    sm = Y[m].sum(0)
    observed = stat(sm, k)
    # permute the smaller group: the same null, fewer draws
    small = min(k, n - k)
    null = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        idx = rng.choice(n, small, replace=False)
        s_small = Y[idx].sum(0)
        null[i] = stat(s_small if small == k else total - s_small, k)
    p = (1 + int((null >= observed - 1e-12).sum())) / (1 + N_PERMUTATIONS)
    a, b = sm / k, (total - sm) / (n - k)
    out: dict[str, Any] = {"kind": yi["kind"], "statistic": st.r(observed, 4), "p_value": st.r(p, 4),
                           "n_permutations": N_PERMUTATIONS}
    if yi["kind"] == "class_share":
        j = yi["focus"]
        out.update({"class": yi["classes"][j], "missing": st.r(float(a[j]), 4), "observed": st.r(float(b[j]), 4)})
    else:
        out.update({"missing": st.r(float(a[0]), 6), "observed": st.r(float(b[0]), 6),
                    "std": st.r(yi["std"], 6)})
    return out


def _structural(rows: pd.DataFrame, c: str, m: np.ndarray, columns: dict[str, ColumnProfile],
                redact: bool) -> dict[str, Any] | None:
    """The other column whose single value marks exactly the missing rows, if any."""
    k = int(m.sum())
    best: tuple[float, dict[str, Any]] | None = None
    for o, cp in columns.items():
        if o == c or cp.kind not in _USABLE:
            continue
        v = rows[o]
        vm = v[m]
        vm = vm[vm.notna()]
        if len(vm) < STRUCTURAL_SHARE * k:
            continue
        top = vm.value_counts()
        value, hits = top.index[0], int(top.iloc[0])
        coverage = hits / k
        if coverage < STRUCTURAL_SHARE:
            continue
        same = (v == value).to_numpy()
        precision = float(m[same].mean()) if same.any() else 0.0
        if precision < STRUCTURAL_SHARE:
            continue
        score = coverage * precision
        if best is None or score > best[0]:
            shown: Any = value
            if isinstance(value, (np.integer, np.floating, float, int)) and not isinstance(value, bool):
                f = float(value)
                shown = int(f) if f.is_integer() else st.r(f)
            elif redact and cp.kind is ColumnKind.CATEGORICAL:
                shown = "<redacted>"
            else:
                shown = str(value)
            best = (score, {"column": o, "value": shown, "coverage": st.r(coverage, 4),
                            "precision": st.r(precision, 4), "n_value_rows": int(same.sum()),
                            "numeric_value": cp.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN)})
    return best[1] if best else None


def _numeric_auc(rows: pd.DataFrame, cols: list[str], masks: dict[str, np.ndarray],
                 columns: dict[str, ColumnProfile]) -> dict[str, dict[str, float]]:
    """Rank AUC of every numeric feature between missing and observed rows, all columns at once."""
    feats = [o for o, cp in columns.items() if cp.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN)]
    if not feats:
        return {}
    X = rows[feats].apply(lambda s: pd.to_numeric(s, errors="coerce")).astype(float)
    X = X.where(np.isfinite(X))
    ok = X.notna().to_numpy()
    R = np.nan_to_num(X.rank().to_numpy())
    M = np.column_stack([masks[c] for c in cols]).astype(float)
    rank_sum = R.T @ M                       # (features, missing columns)
    k1 = ok.T.astype(float) @ M
    k0 = ok.sum(0)[:, None] - k1
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - k1 * (k1 + 1) / 2) / (k1 * k0)
    out: dict[str, dict[str, float]] = {}
    for j, c in enumerate(cols):
        out[c] = {f: float(auc[i, j]) for i, f in enumerate(feats)
                  if f != c and k1[i, j] >= MIN_MISSING and k0[i, j] >= MIN_MISSING and np.isfinite(auc[i, j])}
    return out


def _strongest_feature(rows: pd.DataFrame, c: str, m: np.ndarray, columns: dict[str, ColumnProfile],
                       aucs: dict[str, float]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for f, a in aucs.items():
        strength = abs(2 * a - 1)
        if best is None or strength > best["strength"]:
            best = {"column": f, "measure": "rank AUC (missing vs observed rows)", "value": st.r(a, 4),
                    "strength": st.r(strength, 4)}
    for o, cp in columns.items():
        if o == c or cp.kind is not ColumnKind.CATEGORICAL:
            continue
        codes, _ = pd.factorize(rows[o].astype(str).where(rows[o].notna(), "__missing__"))
        size = np.bincount(codes)
        rate = np.bincount(codes, weights=m.astype(float), minlength=len(size)) / np.maximum(size, 1)
        big = size >= MIN_CATEGORY_ROWS
        if big.sum() < 2:
            continue
        lo, hi = float(np.min(rate[big])), float(np.max(rate[big]))
        spread = hi - lo
        if best is None or spread > best["strength"]:
            best = {"column": o, "measure": "missing-rate range across categories (>= 20 rows)",
                    "value": st.r(spread, 4), "strength": st.r(spread, 4),
                    "rate_range": [st.r(lo, 4), st.r(hi, 4)]}
    return best
