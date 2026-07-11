"""The DataProfile contract: deterministic output of the profiling layer.

This is the boundary artifact between "looking at the data" and "interpreting
it". The profiler produces ONLY this — pure statistics, no recommendations.
It is JSON-serializable so it can be (a) fed to the LLM and (b) asserted against
in tests.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..enums import Modality, Task


class CoreStats(BaseModel):
    """Modality-independent statistics.

    These hold for any modality — class imbalance matters just as much in image
    classification as in tabular. Anything modality-specific lives in
    DataProfile.modality_stats instead.
    """

    n_samples: int
    n_features: int | None = None
    n_classes: int | None = None
    # class label -> count, when the task is classification
    class_counts: dict[str, int] | None = None
    # max_count / min_count; 1.0 means perfectly balanced
    imbalance_ratio: float | None = None
    # rough estimate of mislabeled fraction, when computable
    label_noise_estimate: float | None = None
    # free-form notes about train/val/test split issues
    split_info: dict[str, Any] = Field(default_factory=dict)


class DataProfile(BaseModel):
    """Complete deterministic profile of a dataset.

    Composed of a fixed `core` plus a modality-specific `modality_stats` payload
    that only contains keys relevant to this modality. Rules read whichever keys
    exist; a rule that depends on a key the profiler never produced simply
    cannot fire (see Finding.evidence — this is what makes irrelevant metrics
    structurally impossible to recommend).
    """

    modality: Modality
    task: Task
    # True when `task` was inferred rather than supplied by the user, so the
    # orchestration layer knows to confirm it.
    task_inferred: bool = False
    core: CoreStats
    # modality-specific statistics; only applicable keys are present.
    # tabular example: {"target_skew": 2.3, "outlier_pct": 0.04, ...}
    # image example:   {"resolution_dist": ..., "channel_stats": ..., ...}
    modality_stats: dict[str, Any] = Field(default_factory=dict)
