"""Tabular profiler — real deterministic statistics (Step 1). Only measures."""
from __future__ import annotations
from typing import Any
import pandas as pd
from ..enums import Modality, Task
from .features import profile_features
from .profile import CoreStats, DataProfile
from .target import profile_classification_target, profile_regression_target
from .task_inference import infer_task

def profile_tabular(data: Any, target: str | None, task: Task | None = None, **kwargs: Any) -> DataProfile:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Tabular profiling expects a pandas DataFrame, got {type(data)!r}.")
    df = data
    n_samples = int(len(df))
    n_features = int(df.shape[1] - (1 if target in df.columns else 0))

    task_inferred = task is None
    task_evidence: dict[str, Any] = {}
    if task is None:
        if target is not None and target in df.columns:
            task, task_evidence = infer_task(df[target])
        else:
            task = Task.UNKNOWN

    core = CoreStats(n_samples=n_samples, n_features=n_features)
    modality_stats: dict[str, Any] = {"task_inference": task_evidence}

    if target is not None and target in df.columns:
        y = df[target]
        if task is Task.REGRESSION:
            modality_stats["target"] = profile_regression_target(y)
        elif task is Task.CLASSIFICATION:
            tgt = profile_classification_target(y)
            modality_stats["target"] = tgt
            core.n_classes = tgt.get("n_classes")
            core.class_counts = tgt.get("class_counts")
            core.imbalance_ratio = tgt.get("imbalance_ratio")

    modality_stats["features"] = profile_features(df, target=target, task_is_regression=(task is Task.REGRESSION))

    return DataProfile(
        modality=Modality.TABULAR, task=task, task_inferred=task_inferred,
        core=core, modality_stats=modality_stats,
    )