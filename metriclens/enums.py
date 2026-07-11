"""Shared enums used across all layers.

These are the vocabulary the whole library speaks. Keeping them in one place
avoids string-typing modality/task/severity all over the codebase.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """The kind of data. Determines which profiler runs and which rules apply.

    This is the primary axis of the library — NOT an "ML vs DL" distinction.
    Only TABULAR is implemented in v0; the rest are reserved so contracts and
    dispatch logic are multi-modality from day one.
    """

    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    TIME_SERIES = "time_series"


class Task(str, Enum):
    """The learning task. Combined with Modality, determines metric relevance."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """How strongly a finding should be surfaced.

    INFO     - worth knowing, no action implied.
    WARN     - a metric will likely mislead; act on it.
    CRITICAL - reporting the naive metric here is actively wrong.
    """

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"
