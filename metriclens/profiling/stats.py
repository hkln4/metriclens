"""Pure statistics: per-kind column summaries, bootstrap intervals and the
single-feature association screen. No interpretation lives here."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def r(x: Any, nd: int = 6) -> Any:
    """Round floats for stable, JSON-friendly output; NaN/inf become None."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return round(float(x), nd)
    return x


def iqr_outlier_fraction(v: np.ndarray) -> float:
    q1, q3 = np.quantile(v, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    return float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean())


def skewness(v: np.ndarray) -> float:
    """Adjusted Fisher-Pearson skewness (same estimator as pandas .skew())."""
    n = len(v)
    if n < 3:
        return float("nan")
    d = v - v.mean()
    m2 = float((d ** 2).mean())
    if m2 == 0:
        return 0.0
    g1 = float((d ** 3).mean()) / m2 ** 1.5
    return g1 * np.sqrt(n * (n - 1)) / (n - 2)


def excess_kurtosis(v: np.ndarray) -> float:
    """Bias-corrected excess kurtosis (same estimator as pandas .kurt()); 0 == normal."""
    n = len(v)
    if n < 4:
        return float("nan")
    d = v - v.mean()
    m2 = float((d ** 2).mean())
    if m2 == 0:
        return 0.0
    g2 = float((d ** 4).mean()) / m2 ** 2 - 3
    return ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (r(max(0.0, centre - half)), r(min(1.0, centre + half)))


def numeric_stats(s: pd.Series) -> dict[str, Any]:
    v = s.dropna().astype(float).to_numpy()
    if len(v) == 0:
        return {}
    qs = np.quantile(v, QUANTILES)
    values, counts = np.unique(v, return_counts=True)
    return {
        "mean": r(v.mean()), "std": r(v.std(ddof=1) if len(v) > 1 else 0.0),
        "quantiles": {f"p{int(q * 100):02d}": r(x) for q, x in zip(QUANTILES, qs)},
        "skew": r(skewness(v)), "kurtosis": r(excess_kurtosis(v)),
        "outlier_fraction": r(iqr_outlier_fraction(v)),
        "zero_fraction": r(float((v == 0).mean())),
        "negative_fraction": r(float((v < 0).mean())),
        "dominant_fraction": r(float(np.max(counts)) / len(v)),
    }


def categorical_stats(s: pd.Series, show_labels: bool, top_k: int = 15) -> dict[str, Any]:
    vc = s.dropna().astype(str).value_counts()
    n = int(vc.sum())
    if n == 0:
        return {}
    freq = vc / n
    out: dict[str, Any] = {
        "dominant_fraction": r(float(freq.iloc[0])),
        "rare_category_fraction": r(float(freq[freq < 0.01].sum())),
        "n_rare_categories": int((freq < 0.01).sum()),
        "top_counts": [int(c) for c in vc.head(top_k)],
    }
    if show_labels:
        out["top_labels"] = [str(k) for k in vc.head(top_k).index]
    return out


def bootstrap_ci(
    v: np.ndarray, fn: Callable[[np.ndarray], float], n_boot: int, max_rows: int,
    seed: int, alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of `fn` over rows of `v` (seeded)."""
    if n_boot <= 0 or len(v) < 10:
        return None
    rng = np.random.default_rng(seed)
    m = min(len(v), max_rows)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = fn(v[rng.integers(0, len(v), size=m)])
    stats = stats[np.isfinite(stats)]
    if len(stats) < n_boot // 2:
        return None
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (r(lo), r(hi))


# --------------------------------------------------------------------------- #
# Single-feature association screen (task-agnostic leakage hint)
# --------------------------------------------------------------------------- #

def _numeric_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_ev: np.ndarray) -> np.ndarray:
    """1-D nearest-neighbour by rank: average the targets of the k fit rows
    closest in value (all rows sharing the value when a tie block is >= k).
    Missing values are predicted from fit rows that are also missing."""
    k = int(np.clip(np.sqrt(len(x_fit)), 10, 50))
    default = y_fit.mean(0)
    out = np.tile(default, (len(x_ev), 1)).astype(float)
    ok_f, ok_e = np.isfinite(x_fit), np.isfinite(x_ev)
    if ok_f.sum() >= k:
        order = np.argsort(x_fit[ok_f], kind="stable")
        xs, ys = x_fit[ok_f][order], y_fit[ok_f][order]
        cs = np.vstack([np.zeros((1, ys.shape[1])), np.cumsum(ys, 0)])
        v = x_ev[ok_e]
        left = np.searchsorted(xs, v, "left")
        right = np.searchsorted(xs, v, "right")
        lo = np.clip((left + right) // 2 - k // 2, 0, len(xs) - k)
        hi = lo + k
        tie = (right - left) >= k
        lo = np.where(tie, left, lo)
        hi = np.where(tie, right, hi)
        out[ok_e] = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    miss_f = ~ok_f
    if miss_f.sum() >= 5:
        out[~ok_e] = y_fit[miss_f].mean(0)
    return out


def _lookup_predict(c_fit: np.ndarray, y_fit: np.ndarray, c_ev: np.ndarray) -> np.ndarray:
    """Category -> mean target (one-hot mean for classification); rare/unseen -> overall mean."""
    default = y_fit.mean(0)
    df = pd.DataFrame(y_fit).assign(_c=c_fit)
    g = df.groupby("_c")
    means, sizes = g.mean(), g.size()
    table = {c: means.loc[c].to_numpy() for c in sizes.index[sizes >= 5]}
    return np.array([table.get(c, default) for c in c_ev], dtype=float).reshape(len(c_ev), -1)


def single_feature_score(
    x: pd.Series, y: pd.Series, numeric: bool, classification: bool, seed: int,
) -> float | None:
    """How well this one feature alone predicts the target, on held-out rows.

    A one-feature predictor is fit on a random half of the rows and scored on
    the other half: rank-based nearest neighbours for numeric features, a
    category lookup table for the rest. Score:
      regression      R^2 on the held-out half
      classification  balanced accuracy rescaled so chance = 0 and perfect = 1
    Held-out scoring stops identifiers and high-cardinality columns from scoring
    well by memorization alone.
    """
    mask = y.notna().to_numpy()
    yt = y.to_numpy()[mask]
    n = len(yt)
    if n < 40:
        return None
    if classification:
        labels, yi = np.unique(yt.astype(str), return_inverse=True)
        k = len(labels)
        if k < 2:
            return None
        ymat = np.eye(k)[yi]
    else:
        yf = yt.astype(float)
        ymat = yf[:, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fit, ev = perm[: n // 2], perm[n // 2:]
    if numeric:
        xv = pd.to_numeric(x, errors="coerce").astype(float).to_numpy()[mask]
        pred = _numeric_predict(xv[fit], ymat[fit], xv[ev])
    else:
        codes, _ = pd.factorize(x.astype(str).where(x.notna(), "__nan__"))
        codes = codes[mask]
        pred = _lookup_predict(codes[fit], ymat[fit], codes[ev])
    if classification:
        p = pred.argmax(1)
        true = yi[ev]
        recalls = [float((p[true == j] == j).mean()) for j in range(k) if (true == j).any()]
        bacc = float(np.mean(recalls))
        return r(max(0.0, (bacc - 1 / k) / (1 - 1 / k)), 4)
    true_r = yf[ev]
    ss_tot = float(((true_r - true_r.mean()) ** 2).sum())
    if ss_tot == 0:
        return None
    return r(max(0.0, 1 - float(((true_r - pred[:, 0]) ** 2).sum()) / ss_tot), 4)
