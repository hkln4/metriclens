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
    kind_counts: dict[str, int] = Field(default_factory=dict)
    missing_columns: tuple[str, ...] = ()
    # Jaccard co-missingness between the most-missing columns
    missing_cooccurrence: dict[str, Any] = Field(default_factory=dict)


class Relations(Frozen):
    association_method: str = (
        "2-fold cross-validated single-feature score on the development rows: rank nearest-neighbours "
        "(numeric) or a category lookup table, fit on one half and scored on the other, then swapped "
        "(R^2 for regression, chance-rescaled balanced accuracy for classification)")
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
    def feature_columns(self) -> list[str]:
        return list(self.columns)

    def kind_of(self, kind: ColumnKind) -> list[str]:
        return [c for c, p in self.columns.items() if p.kind is kind]
