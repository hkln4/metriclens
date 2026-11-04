"""The DataProfile contract: deterministic output of the profiling layer.

The boundary artifact between "looking at the data" and "deciding". It holds
only aggregated statistics — never raw rows — and is JSON-serializable, so it can
be stored in the report, asserted against in tests, and (in part) sent to an LLM.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..enums import ColumnKind, Modality, Task
from ..frozen import Frozen


class ColumnProfile(Frozen):
    name: str
    kind: ColumnKind
    kind_reason: str
    dtype: str
    missing_fraction: float
    missing_ci: tuple[float, float] | None = None
    cardinality: int
    unique_ratio: float
    stats: dict[str, Any] = Field(default_factory=dict)
    ci: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # 2-fold cross-validated single-feature score (see stats.single_feature_score)
    association: float | None = None
    # the same predictor scored threshold-free (Gini for classification): where the signal is, not a leak screen
    signal: float | None = None
    # how values go missing: target relation, structural pattern, related features (see profiling.missingness)
    missingness: dict[str, Any] = Field(default_factory=dict)


class TargetProfile(Frozen):
    name: str
    task: Task
    task_inferred: bool
    task_evidence: dict[str, Any] = Field(default_factory=dict)
    n_missing: int
    missing_fraction: float
    stats: dict[str, Any] = Field(default_factory=dict)
    ci: dict[str, tuple[float, float]] = Field(default_factory=dict)


class DatasetProfile(Frozen):
    n_rows: int                 # rows profiled (after holdout and sampling)
    n_rows_input: int           # rows the user passed in
    n_columns: int
    n_features: int
    duplicate_rows: int
    duplicate_fraction: float
    # groups of rows with identical features but different targets
    conflicting_duplicate_groups: int = 0
    # distinct duplicated rows, and every row that has an identical twin (duplicate_rows counts extra copies)
    duplicate_groups: int = 0
    duplicate_rows_in_groups: int = 0
    # are copies close together in row order? {median_gap, random_median_gap, p_value, n_permutations}
    duplicate_adjacency: dict[str, Any] = Field(default_factory=dict)
    kind_counts: dict[str, int] = Field(default_factory=dict)
    missing_columns: tuple[str, ...] = ()
    # Jaccard co-missingness between the most-missing columns
    missing_cooccurrence: dict[str, Any] = Field(default_factory=dict)


class Relations(Frozen):
    association_method: str = (
        "2-fold cross-validated single-feature score on the development rows: rank nearest-neighbours "
        "(numeric) or a category lookup table, fit on one half and scored on the other, then swapped "
        "(R^2 for regression, chance-rescaled balanced accuracy for classification)")
    signal_method: str = (
        "the same 2-fold single-feature predictor, scored threshold-free: R^2 for regression, Gini = 2 AUC - 1 "
        "(one-vs-rest, averaged over classes) for classification")
    corr_method: str = "Spearman rank correlation"
    corr_columns: tuple[str, ...] = ()
    corr_matrix: tuple[tuple[float | None, ...], ...] = ()
    high_corr_pairs: tuple[dict[str, Any], ...] = ()
    # |Pearson r| of numeric features with a numeric target (regression only)
    pearson_target: dict[str, float] = Field(default_factory=dict)


class DataProfile(Frozen):
    modality: Modality = Modality.TABULAR
    task: Task
    task_inferred: bool
    target: TargetProfile
    dataset: DatasetProfile
    columns: dict[str, ColumnProfile]
    relations: Relations
    # columns that are not features: {"user_id": "groups", "ts": "time_col"}
    roles: dict[str, str] = Field(default_factory=dict)
    sampling: dict[str, Any] = Field(default_factory=dict)
    holdout: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"<DataProfile {self.task.value}: {self.dataset.n_rows} dev rows, "
                f"{self.dataset.n_features} features>")

    @property
    def task_needs_answer(self) -> bool:
        """The task was inferred and the inference could go either way (an integer target
        with few distinct values: labels or a score?). A text, boolean, binary or
        continuous target leaves no doubt, so nobody is asked."""
        return self.task_inferred and bool(self.target.task_evidence.get("ambiguous"))

    @property
    def feature_columns(self) -> list[str]:
        return list(self.columns)

    def kind_of(self, kind: ColumnKind) -> list[str]:
        return [c for c, p in self.columns.items() if p.kind is kind]
