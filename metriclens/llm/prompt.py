"""Builds the (system, user) prompt for narrative interpretation.

What the model receives is ONLY the precomputed profile summary, the findings,
and the user's context. There is no raw data here — by construction the
DataProfile never carried any — so the trust boundary is enforced structurally."""
from __future__ import annotations
import json
from ..profiling.profile import DataProfile
from ..rules.finding import Finding

SYSTEM = (
    "You interpret a PRECOMPUTED data profile for a machine-learning project. "
    "Every number you are given was already computed by deterministic code and "
    "is correct. You must NEVER invent, recompute, or estimate any statistic, "
    "and you must base every statement on the findings provided.\n\n"
    "Write a concise narrative (a few short paragraphs, prose, no headers) that:\n"
    "1. Prioritizes the most severe findings first and explains WHY each matters.\n"
    "2. Ties every metric recommendation to the specific evidence number that "
    "drives it (e.g. 'because the imbalance ratio is 47:1, ...').\n"
    "3. States plainly which metrics will mislead here and why.\n"
    "4. If the task was inferred rather than given, tells the user to confirm it.\n"
    "Do not introduce metrics that none of the findings mention. If there are no "
    "findings, say the data looks unremarkable for this task and suggest the "
    "conventional metrics, briefly."
)

def _profile_summary(profile: DataProfile) -> dict:
    """Compact, LLM-facing view — numbers only, never raw data."""
    return {
        "modality": profile.modality.value,
        "task": profile.task.value,
        "task_inferred": profile.task_inferred,
        "task_inference": profile.modality_stats.get("task_inference", {}),
        "core": profile.core.model_dump(exclude_none=True),
        "target": profile.modality_stats.get("target", {}),
    }

def build_messages(profile, findings, context):
    payload = {
        "profile": _profile_summary(profile),
        "findings": [f.model_dump() for f in findings],
        "user_context": context or "(none provided)",
    }
    user = ("Here is the precomputed profile and the deterministic findings.\n\n"
            + json.dumps(payload, indent=2, default=str)
            + "\n\nWrite the interpretation narrative.")
    return SYSTEM, user