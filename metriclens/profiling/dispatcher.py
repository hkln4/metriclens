"""Routes profiling to the right modality-specific profiler.

The pipeline downstream (rules -> llm -> roadmap) is shared across modalities;
only the input adapter and profiler differ. This dispatcher is where that fork
happens.

Step 0 stub: only the tabular profiler exists, and it returns a hollow profile.
"""

from __future__ import annotations

from typing import Any

from ..enums import Modality, Task
from .profile import DataProfile
from .tabular import profile_tabular


def profile_data(
    data: Any,
    target: str | None,
    modality: Modality,
    task: Task | None,
    **kwargs: Any,
) -> DataProfile:
    """Dispatch to the profiler for the given modality."""
    if modality is Modality.TABULAR:
        return profile_tabular(data, target=target, task=task, **kwargs)
    raise NotImplementedError(
        f"Profiling for modality '{modality.value}' is not implemented yet. "
        "v0 supports tabular only."
    )
