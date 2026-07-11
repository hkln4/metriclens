"""metriclens — data-aware evaluation roadmaps.

Takes a dataset at the start of a project, profiles it deterministically, and
produces a reasoned roadmap of which metrics to report, which will mislead you
and why, and how to validate. LLM enrichment is optional; the deterministic core
works on its own.
"""

from __future__ import annotations

from .api import analyze, review
from .roadmap import Roadmap
from .error_analysis import ReviewReport
from .enums import Modality, Severity, Task

__version__ = "0.0.0"

__all__ = [
    "analyze",
    "review",
    "Roadmap",
    "ReviewReport",
    "Modality",
    "Task",
    "Severity",
    "__version__",
]
