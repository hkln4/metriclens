"""User-tunable limits. Defaults are deliberately conservative: prefit must
stay a fast pre-training step, never turn into an AutoML search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Budget(BaseModel):
    """Hard limits for the verification experiments.

    Each conditional decision is tested at most once per model family, as
    "current setup vs. current setup + this change" — no combinations, no search.
    """

    max_experiments: int = Field(6, ge=0, description="Total experiment runs (decision x model family).")
    max_seconds: float = Field(180.0, gt=0, description="Wall-clock budget for all experiments.")
    max_rows: int = Field(20_000, ge=100, description="Dev rows used for experiments (stratified subsample above this).")
    n_splits: int = Field(5, ge=2)
    n_repeats: int = Field(2, ge=1)
    alpha: float = Field(0.05, gt=0, lt=1, description="Family-wise error rate (Holm-corrected).")
    min_relative_effect: float = Field(
        0.01, ge=0, description="Smallest improvement that counts, as a fraction of |baseline score| "
                                "(e.g. 0.01 = 1%). Smaller significant differences are reported as negligible.")
    n_jobs: int | None = Field(
        None, description="Parallel fold fits (joblib). None = automatic: serial, switching to all cores once one "
                          "cross-validation takes longer than 2 s; 1 = always serial; -1 = always all cores. Folds "
                          "and seeds are fixed, so results do not depend on it.")
    min_effect: float | None = Field(
        None, ge=0, description="Absolute practical threshold in metric units; overrides min_relative_effect.")


class ProfileOptions(BaseModel):
    """Limits for the profiling stage."""

    max_rows: int = Field(200_000, ge=1_000, description="Rows profiled; above this a seeded sample is used.")
    n_bootstrap: int = Field(200, ge=0, description="Bootstrap resamples for threshold-relevant statistics.")
    bootstrap_rows: int = Field(5_000, ge=100, description="Rows per bootstrap resample (subsampled above this).")
    max_corr_features: int = Field(20, ge=2, description="Numeric features shown in the correlation matrix.")
    association_rows: int = Field(
        50_000, ge=1_000, description="Rows used (seeded sample above this) for the single-feature scores, the "
                                      "missingness screens and the rank correlations.")
    redact_values: bool = Field(
        False, description="Hide category labels everywhere (figures, report). Counts are kept.")
    max_labelled_cardinality: int = Field(
        50, description="Category labels are only shown for columns at or below this cardinality.")
