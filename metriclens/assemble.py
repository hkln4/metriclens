"""Assemble the ordered roadmap, the limitations and the rule track record."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .enums import DecisionStatus
from .report.model import Decision, RoadmapStage

STAGES = [
    ("split", "1. Set up validation"),
    ("data_review", "2. Review the data"),
    ("columns", "3. Choose the columns"),
    ("preprocessing", "4. Preprocess inside the folds"),
    ("target", "5. Target and class handling"),
    ("metrics", "6. Pick the metric"),
    ("final", "7. Baseline and final evaluation"),
]

_STATUS_WORDS = {
    DecisionStatus.APPLIED: "applied",
    DecisionStatus.PROPOSED: "untested",
    DecisionStatus.UNCERTAIN: "near threshold",
    DecisionStatus.VERIFIED: "verified",
    DecisionStatus.SUPPORTED_BY_PROXY: "proxy-supported",
    DecisionStatus.INCONCLUSIVE: "inconclusive",
    DecisionStatus.REFUTED: "refuted",
    DecisionStatus.NOT_APPLICABLE: "not applicable",
    DecisionStatus.SKIPPED_BUDGET: "skipped (budget)",
    DecisionStatus.AWAITING_USER: "awaiting your answer",
    DecisionStatus.CONFIRMED: "confirmed",
    DecisionStatus.DECLINED: "declined",
    DecisionStatus.NEEDS_REVIEW: "needs review",
}


def status_word(s: DecisionStatus) -> str:
    return _STATUS_WORDS[s]


def build_roadmap(decisions: list[Decision]) -> list[RoadmapStage]:
    out = []
    for stage, title in STAGES:
        ds = [d for d in decisions if d.stage == stage]
        if not ds:
            continue
        counts: dict[str, int] = {}
        for d in ds:
            w = status_word(d.status)
            counts[w] = counts.get(w, 0) + 1
        out.append(RoadmapStage(stage=stage, title=title, decision_ids=[d.id for d in ds],
                                summary=" · ".join(f"{n} {w}" for w, n in counts.items())))
    return out


def build_limitations(report_like: dict[str, Any]) -> list[str]:
    lim = [
        "Leakage screening covers single-feature association and numeric correlation only. Leakage through "
        "combinations of features, through time, through preprocessing, or through repeated entities is not "
        "detected; a clean screen is not proof of no leakage.",
        "Thresholds are rule triggers, not verdicts. Each finding shows its rule version and thresholds; values "
        "whose interval straddles a threshold are marked 'near threshold'.",
        "Column kinds and the task are inferred heuristically; check the schema table.",
    ]
    sampling = report_like.get("sampling") or {}
    if sampling.get("sampled"):
        lim.append(f"Profiling used a seeded sample of {sampling['n_rows']} of {sampling['of_rows']} development rows.")
    exps = report_like.get("experiments") or []
    if exps:
        lim.append("Experiments test one change at a time against the same base setup; interactions between "
                   "changes are not measured.")
        lim.append("Experiment results are development evidence from cross-validation on the dev rows. A claim "
                   "about generalization needs the untouched holdout (or nested CV when there is none).")
        if any(e.family != "user" for e in exps):
            lim.append("Proxy-model results are a sensitivity analysis; they do not stand in for your final "
                       "model. Pass estimator= to verify decisions for your model.")
    elif report_like.get("has_conditional"):
        lim.append("Conditional decisions were not tested (verify was off); none of them is in the exported "
                   "pipeline.")
    return lim


def load_track_record() -> dict[str, Any]:
    try:
        text = resources.files("metriclens").joinpath("data/track_record.json").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return {}
    return json.loads(text)
