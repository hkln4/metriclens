"""The Finding contract: output of a single rule.

The whole anti-platitude guarantee of the library lives in this file: a Finding
CANNOT be constructed without `evidence`. Because evidence is the set of real
numbers that triggered the rule, it is structurally impossible to emit a
recommendation like "choose an appropriate metric" that isn't bound to a
concrete statistic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..enums import Modality, Severity, Task


class MetricSuggestion(BaseModel):
    """Which metrics this finding suggests, split by priority."""

    primary: list[str] = Field(default_factory=list)
    secondary: list[str] = Field(default_factory=list)
    rationale: str = ""


class Pitfall(BaseModel):
    """A metric that will mislead under this finding's condition, and why."""

    metric: str
    reason: str


class Finding(BaseModel):
    """A single deterministic conclusion drawn from the DataProfile.

    `evidence` is mandatory and has no default — a Finding without the numbers
    that justify it cannot exist.
    """

    id: str
    severity: Severity
    # the triggering numbers, e.g. {"imbalance_ratio": 47.0}. REQUIRED.
    evidence: dict[str, Any]
    message: str
    candidate_metrics: MetricSuggestion = Field(default_factory=MetricSuggestion)
    pitfalls: list[Pitfall] = Field(default_factory=list)
    # how to validate given this finding, e.g. "use stratified k-fold".
    validation_notes: list[str] = Field(default_factory=list)
    # data-prep suggestions implied by this finding, e.g. "log-transform target".
    preprocessing_notes: list[str] = Field(default_factory=list)
    # (modality, task) pairs this rule is valid for; used for catalog clarity
    # and to skip expensive checks. The real safety is evidence-gating above.
    applies_to: list[tuple[Modality, Task]] = Field(default_factory=list)
