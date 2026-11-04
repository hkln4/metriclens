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


def first_look(report: Any) -> pd.DataFrame:
    """One row per feature column: what prefit measured and which decisions mention it."""
    p = report.details.profile
    by_col: dict[str, list[str]] = {}
    fcols = {f.id: f.columns for f in report.details.findings}
    for d in report.decisions:
        if d.standard:
            continue
        cols = {c for fid in d.finding_ids for c in fcols.get(fid, ())}
        cols |= {c for eid in d.evidence_ids for c in report.details.evidence[eid].columns}
        for c in cols:
            by_col.setdefault(c, []).append(d.id)
    rows = []
    for c, cp in p.columns.items():
        rows.append({
            "column": c, "kind": cp.kind.value, "missing %": round(100 * cp.missing_fraction, 2),
            "distinct": cp.cardinality, "skew": cp.stats.get("skew"),
            "signal": cp.signal, "leak score": cp.association,
            "decisions": ", ".join(sorted(set(by_col.get(c, [])))),
        })
    return pd.DataFrame(rows, columns=["column", "kind", "missing %", "distinct", "skew", "signal", "leak score",
                                       "decisions"])
