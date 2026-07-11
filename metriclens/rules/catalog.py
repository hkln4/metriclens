"""The rule catalog: deterministic findings derived from a DataProfile.
Every rule is evidence-gated; thresholds are tunable module constants."""
from __future__ import annotations
from ..profiling.profile import DataProfile
from ..rules.finding import Finding, MetricSuggestion, Pitfall
from ..enums import Modality, Severity, Task
from .registry import rule

TAB = Modality.TABULAR
IMBALANCE_WARN, IMBALANCE_CRIT = 3.0, 10.0
SKEW_INFO, SKEW_WARN, OUTLIER_WARN = 1.0, 2.0, 0.05
MISSING_WARN, MISSING_CRIT = 0.20, 0.50
SMALL_DATA_WARN, SMALL_DATA_CRIT = 1000, 200
ZERO_INFLATION_WARN = 0.30
KURTOSIS_WARN = 3.0          # Fisher excess kurtosis; 0 == normal
DUPLICATE_WARN = 0.05        # >= this fraction of duplicate rows -> WARN, else INFO


@rule("class_imbalance", applies_to=[(TAB, Task.CLASSIFICATION)])
def class_imbalance(p: DataProfile) -> Finding | None:
    ratio = p.core.imbalance_ratio
    if ratio is None or ratio < IMBALANCE_WARN:
        return None
    minority = (p.modality_stats.get("target") or {}).get("minority_fraction")
    severity = Severity.CRITICAL if ratio >= IMBALANCE_CRIT else Severity.WARN
    return Finding(
        id="class_imbalance", severity=severity,
        evidence={"imbalance_ratio": round(ratio, 2), "minority_fraction": minority},
        message=(f"Class imbalance is {ratio:.1f}:1"
                 + (f" (minority class is {minority:.1%} of data)" if minority else "")
                 + ". A trivial majority-class predictor would score high on accuracy."),
        candidate_metrics=MetricSuggestion(
            primary=["PR-AUC / average precision", "balanced accuracy", "macro F1"],
            secondary=["recall on the minority class", "MCC"],
            rationale="Metrics that account for the rare class, not overall hit-rate."),
        pitfalls=[
            Pitfall(metric="accuracy", reason="Dominated by the majority class; high even for a useless model."),
            Pitfall(metric="ROC-AUC", reason="Can look optimistic when positives are rare; prefer PR-AUC.")],
        validation_notes=["Use stratified splits / stratified k-fold to preserve class ratios."])


@rule("target_skew", applies_to=[(TAB, Task.REGRESSION)])
def target_skew(p: DataProfile) -> Finding | None:
    tg = p.modality_stats.get("target") or {}
    skew = tg.get("skew"); outliers = tg.get("outlier_fraction", 0.0)
    if skew is None: return None
    heavy = outliers >= OUTLIER_WARN
    if abs(skew) < SKEW_INFO and not heavy: return None
    severity = Severity.WARN if (abs(skew) >= SKEW_WARN or heavy) else Severity.INFO
    return Finding(
        id="target_skew", severity=severity,
        evidence={"skew": round(skew, 3), "outlier_fraction": round(outliers, 4)},
        message=(f"Target is skewed (skew={skew:.2f}) with {outliers:.1%} IQR-outliers. "
                 "Squared-error metrics will be dominated by a few large errors."),
        candidate_metrics=MetricSuggestion(
            primary=["MAE", "median absolute error"],
            secondary=["RMSE (report alongside, not alone)"],
            rationale="Absolute-error metrics resist tail domination; the MAE-vs-RMSE gap itself is informative."),
        pitfalls=[
            Pitfall(metric="RMSE", reason="Heavily weights a few large/outlier errors under a skewed target."),
            Pitfall(metric="R²", reason="Can look healthy while large-error behavior is poor.")],
        preprocessing_notes=["Consider a log/Box-Cox transform of the target if a multiplicative error scale is acceptable."])


@rule("high_missingness", applies_to=[(TAB, None)])
def high_missingness(p: DataProfile) -> Finding | None:
    feats = p.modality_stats.get("features") or {}
    max_missing = feats.get("max_missing_fraction", 0.0)
    if max_missing < MISSING_WARN:
        return None
    cols = feats.get("columns") or {}
    affected = {c: round(v.get("missing_fraction", 0.0), 3) for c, v in cols.items()
                if v.get("missing_fraction", 0.0) >= MISSING_WARN}
    severity = Severity.CRITICAL if max_missing >= MISSING_CRIT else Severity.WARN
    if max_missing >= MISSING_CRIT:
        prep = ["Columns above 50% missing carry little signal — consider dropping them rather than imputing.",
                "For the rest, impute and add a 'was-missing' indicator column so the model can use missingness itself as signal."]
    else:
        prep = ["Impute numeric columns by median (robust to skew), categoricals by a dedicated 'missing' category; add a missingness-indicator where it may be informative.",
                "Avoid mean-imputation on skewed columns — it shifts the distribution."]
    return Finding(
        id="high_missingness", severity=severity,
        evidence={"max_missing_fraction": round(max_missing, 3), "columns": affected},
        message=(f"{len(affected)} column(s) exceed {MISSING_WARN:.0%} missingness "
                 f"(worst {max_missing:.0%}). Metrics on complete cases only can misrepresent real performance."),
        pitfalls=[Pitfall(metric="any metric on complete-case data only",
                          reason="Dropping rows with missing values changes the evaluated population and usually flatters the score.")],
        preprocessing_notes=prep,
        validation_notes=["Fit imputation inside each CV fold only — never on the full dataset — to avoid leakage."])


@rule("target_leakage", applies_to=[(TAB, Task.REGRESSION)])
def target_leakage(p: DataProfile) -> Finding | None:
    feats = p.modality_stats.get("features") or {}
    hints = feats.get("leakage_hints") or []
    if not hints:
        return None
    worst = max(hints, key=lambda h: abs(h["correlation"]))
    return Finding(
        id="target_leakage", severity=Severity.CRITICAL,
        evidence={"leakage_hints": hints},
        message=(f"Feature '{worst['feature']}' correlates with the target at "
                 f"|r|={abs(worst['correlation']):.3f} — a likely leak. Any metric computed "
                 "with it present looks near-perfect and means nothing."),
        pitfalls=[Pitfall(metric="every metric (MAE, RMSE, R², ...)",
                          reason="A leaked feature inflates all of them; the score reflects the leak, not real skill.")],
        preprocessing_notes=["Inspect the flagged feature(s); if they encode the target or are unavailable at prediction time, remove them before evaluating."])


@rule("small_data", applies_to=[(TAB, None)])
def small_data(p: DataProfile) -> Finding | None:
    n = p.core.n_samples
    if n is None or n >= SMALL_DATA_WARN:
        return None
    severity = Severity.CRITICAL if n < SMALL_DATA_CRIT else Severity.WARN
    return Finding(
        id="small_data", severity=severity,
        evidence={"n_samples": n},
        message=(f"Only {n} samples. A single train/test split gives a noisy, high-variance "
                 "estimate; small metric differences are not reliable."),
        pitfalls=[Pitfall(metric="point estimates from one split",
                          reason="With few samples a single split's score swings widely with the random seed.")],
        validation_notes=["Use k-fold (or repeated/nested) cross-validation instead of one split.",
                          "Report confidence intervals / std across folds; don't over-read small score gaps."])


@rule("constant_features", applies_to=[(TAB, None)])
def constant_features(p: DataProfile) -> Finding | None:
    cols = (p.modality_stats.get("features") or {}).get("constant_columns") or []
    if not cols:
        return None
    return Finding(
        id="constant_features", severity=Severity.INFO,
        evidence={"constant_columns": cols},
        message=f"{len(cols)} constant column(s) carry no signal: {cols}.",
        preprocessing_notes=["Drop constant columns; they cannot help any model."])


@rule("id_like_features", applies_to=[(TAB, None)])
def id_like_features(p: DataProfile) -> Finding | None:
    cols = (p.modality_stats.get("features") or {}).get("near_unique_columns") or []
    if not cols:
        return None
    return Finding(
        id="id_like_features", severity=Severity.WARN,
        evidence={"near_unique_columns": cols},
        message=(f"{len(cols)} near-unique, identifier-like column(s): {cols}. "
                 "If fed to a model they invite memorization and leakage."),
        pitfalls=[Pitfall(metric="any metric with id-like features included",
                          reason="The model can memorize per-row identifiers, inflating scores without real generalization.")],
        preprocessing_notes=["Drop or hash identifier-like columns before modeling unless they are genuinely predictive."])


@rule("duplicate_rows", applies_to=[(TAB, None)])
def duplicate_rows(p: DataProfile) -> Finding | None:
    feats = p.modality_stats.get("features") or {}
    frac = feats.get("duplicate_fraction", 0.0)
    if not frac:
        return None
    severity = Severity.WARN if frac >= DUPLICATE_WARN else Severity.INFO
    return Finding(
        id="duplicate_rows", severity=severity,
        evidence={"duplicate_fraction": round(frac, 4), "duplicate_rows": feats.get("duplicate_rows")},
        message=(f"{frac:.1%} of rows are exact duplicates. If duplicates land on both sides "
                 "of a split, held-out metrics are optimistically inflated."),
        pitfalls=[Pitfall(metric="any held-out metric with duplicates across splits",
                          reason="The same row in train and test leaks the answer, overstating generalization.")],
        preprocessing_notes=["De-duplicate before splitting; if repeats are legitimate, split by group/id so copies stay on one side."])


@rule("zero_inflation", applies_to=[(TAB, Task.REGRESSION)])
def zero_inflation(p: DataProfile) -> Finding | None:
    zf = (p.modality_stats.get("target") or {}).get("zero_fraction")
    if zf is None or zf < ZERO_INFLATION_WARN:
        return None
    return Finding(
        id="zero_inflation", severity=Severity.WARN,
        evidence={"zero_fraction": round(zf, 3)},
        message=(f"{zf:.0%} of target values are exactly zero (zero-inflated). "
                 "Relative-error metrics break down around zero."),
        candidate_metrics=MetricSuggestion(
            primary=["MAE", "Tweedie deviance"],
            secondary=["two-part evaluation: P(zero) + error on positives"],
            rationale="Zero-inflated targets often need a hurdle / two-part view."),
        pitfalls=[Pitfall(metric="MAPE / MSLE", reason="Undefined or explosive when true values are zero.")],
        preprocessing_notes=["Consider a two-part (hurdle) model: classify zero-vs-positive, then regress on the positives."])


@rule("negative_target_values", applies_to=[(TAB, Task.REGRESSION)])
def negative_target_values(p: DataProfile) -> Finding | None:
    nf = (p.modality_stats.get("target") or {}).get("negative_fraction")
    if not nf:
        return None
    return Finding(
        id="negative_target_values", severity=Severity.INFO,
        evidence={"negative_fraction": round(nf, 3)},
        message=(f"{nf:.0%} of target values are negative. Log-scale transforms and "
                 "log-based metrics are not applicable as-is."),
        pitfalls=[Pitfall(metric="MSLE / log-RMSE", reason="Require non-negative values; undefined for negatives."),
                  Pitfall(metric="MAPE", reason="Sign changes near zero make percentage error unstable.")],
        preprocessing_notes=["For a multiplicative scale, shift the target or use a signed transform (e.g. arcsinh) instead of log."])


@rule("heavy_tailed_target", applies_to=[(TAB, Task.REGRESSION)])
def heavy_tailed_target(p: DataProfile) -> Finding | None:
    kurt = (p.modality_stats.get("target") or {}).get("kurtosis")
    if kurt is None or kurt < KURTOSIS_WARN:
        return None
    return Finding(
        id="heavy_tailed_target", severity=Severity.WARN,
        evidence={"kurtosis": round(kurt, 2)},
        message=(f"Target is heavy-tailed (excess kurtosis={kurt:.1f}). A few extreme values "
                 "will dominate squared-error metrics."),
        candidate_metrics=MetricSuggestion(
            primary=["MAE", "quantile / Huber loss"],
            secondary=["report the count and size of large errors separately"],
            rationale="Robust losses resist domination by rare extremes."),
        pitfalls=[Pitfall(metric="RMSE / MSE", reason="A handful of extreme points can swing the value arbitrarily.")])