"""Tabular views of a report, for notebooks (pandas DataFrames)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import experiment_summary, key_evidence, task_dependence_note


def decisions_table(report: Any) -> pd.DataFrame:
    rows = []
    for st in report.roadmap:
        for did in st.decision_ids:
            d = report.decision(did)
            rows.append({
                "stage": st.title, "status": d.status.value, "class": d.cls.value, "decision": d.title,
                "why": d.why_short, "key evidence": key_evidence(report, d), "do": d.do_short,
                "experiment": experiment_summary(report, d), "standard": d.standard,
                "note": " ".join(x for x in (d.review_note, d.note, task_dependence_note(report, d)) if x),
                "id": d.id,
            })
    return pd.DataFrame(rows, columns=["stage", "status", "class", "decision", "why", "key evidence", "do",
                                       "experiment", "standard", "note", "id"])
