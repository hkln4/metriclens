"""Shared vocabulary used across all layers.

Keeping these in one place avoids string-typing kinds, classes and statuses all
over the codebase. Every enum is a `str` subclass so it serializes to JSON as its
plain value.
"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """The kind of data. v1 implements TABULAR only."""

    TABULAR = "tabular"


class Task(str, Enum):
    """The supervised learning task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """How strongly a finding should be surfaced.

    INFO     - worth knowing, no action implied.
    WARN     - likely to distort training or evaluation; look at it.
    CRITICAL - evaluation is probably wrong until this is resolved.
    """

    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class ColumnKind(str, Enum):
    """What a column *is*, as far as modelling is concerned (not its dtype)."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID_LIKE = "id_like"
    CONSTANT = "constant"


class Trigger(str, Enum):
    """How a rule's measured statistic relates to its threshold.

    FIRED     - the statistic (and its whole confidence interval) is past the threshold.
    UNCERTAIN - the confidence interval straddles the threshold; the trigger
                could go either way on a different sample.
    """

    FIRED = "fired"
    UNCERTAIN = "uncertain"


class DecisionClass(str, Enum):
    """How much a decision can be known from the data alone.

    PROCEDURAL  - about *how to evaluate*, correct regardless of the data
                  (fit preprocessing inside folds, stratify, keep the holdout
                  untouched). Never modifies the data's content.
    CONDITIONAL - may help or hurt depending on the data and the model; only a
                  bounded experiment can support it.
    CONTEXTUAL  - cannot be decided from the data (cost of errors, what is
                  available at prediction time); needs the user's answer.
    REVIEW      - a warning that needs human inspection; metriclens proposes no
                  automatic action.
    """

    PROCEDURAL = "procedural"
    CONDITIONAL = "conditional"
    CONTEXTUAL = "contextual"
    REVIEW = "review"


class DecisionStatus(str, Enum):
    """Lifecycle of a decision. Only some statuses are pipeline-eligible."""

    APPLIED = "applied"                    # procedural, part of the pipeline
    PROPOSED = "proposed"                  # conditional, not tested yet
    UNCERTAIN = "uncertain"                # evidence sits near the threshold
    VERIFIED = "verified"                  # experiment with the user's estimator supports it
    SUPPORTED_BY_PROXY = "supported_by_proxy"  # only a proxy model supports it (sensitivity)
    INCONCLUSIVE = "inconclusive"          # experiment ran, difference within noise
    REFUTED = "refuted"                    # experiment shows it hurts
    NOT_APPLICABLE = "not_applicable"      # the estimator cannot take this change
    SKIPPED_BUDGET = "skipped_budget"      # experiment budget exhausted
    AWAITING_USER = "awaiting_user"        # contextual, unanswered
    CONFIRMED = "confirmed"                # contextual, user said yes
    DECLINED = "declined"                  # contextual, user said no
    NEEDS_REVIEW = "needs_review"          # review item
