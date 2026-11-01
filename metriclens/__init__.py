"""metriclens — decision-driven EDA for tabular supervised learning.

Profiles a dataset before training, turns the measurements into classified,
evidence-backed decisions (procedural / conditional / contextual / review),
optionally tests the conditional ones with bounded experiments, and exports a
scikit-learn pipeline that contains only safe, confirmed or verified steps.
"""

from __future__ import annotations

from ._version import __version__
from .api import analyze, review
from .config import Budget, ProfileOptions
from .enums import ColumnKind, DecisionClass, DecisionStatus, Severity, Task
from .llm.config import LLMConfig
from .llm.provider import Provider
from .posttrain import ReviewReport
from .report.model import Report

__all__ = [
    "analyze", "review", "Report", "ReviewReport", "Budget", "ProfileOptions", "LLMConfig", "Provider",
    "Task", "Severity", "ColumnKind", "DecisionClass", "DecisionStatus", "__version__",
]
