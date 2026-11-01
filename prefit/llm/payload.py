"""Build the exact payload sent to an LLM, from an allowlist of report fields.

With `share_column_names=False`, column names never leave the machine: columns
become `col_1..n`, every id that embeds a column name is replaced by an opaque id
(E1, F1, D1, Q1), and free-text messages (which mention column names) are not
sent — only rule ids, severities, classes, statuses and numbers.
"""

from __future__ import annotations

from typing import Any

from .config import LLMConfig

NEVER_SENT = ["raw rows", "category labels", "per-column quantiles", "holdout data"]


def build_payload(report: Any, cfg: LLMConfig) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (payload, reverse_map). reverse_map maps sent ids/names -> real ones."""
    p = report.details.profile
    share = cfg.share_column_names
    col_alias = {c: (c if share else f"col_{i + 1}") for i, c in enumerate(p.columns)}
    col_alias[report.target] = report.target if share else "target"
    ids: dict[str, str] = {}

    def oid(real: str, prefix: str) -> str:
        if share:
            return real
        if real not in ids:
            ids[real] = f"{prefix}{sum(1 for v in ids.values() if v.startswith(prefix)) + 1}"
        return ids[real]

    def val(v: Any) -> Any:
        if share:
            return v
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, list):
            return [col_alias.get(x, "…") if isinstance(x, str) else val(x) for x in v]
        if isinstance(v, dict):
            return {col_alias.get(k, "…"): val(x) for k, x in v.items()}
        return "(text withheld)"

    payload: dict[str, Any] = {
        "task": report.task.value,
        "task_inferred": p.task_inferred,
        "n_rows": p.dataset.n_rows,
        "n_features": p.dataset.n_features,
        "target": col_alias[report.target],
        "columns": [{"name": col_alias[c], "kind": cp.kind.value, "missing_fraction": cp.missing_fraction,
                     "distinct": cp.cardinality} for c, cp in p.columns.items()],
        "evidence": {oid(eid, "E"): ({"label": e.label} if share else {}) | {"value": val(e.value), "ci": e.ci,
                                                                               "columns": [col_alias.get(c, "…") for c in e.columns]}
                     for eid, e in report.details.evidence.items()},
        "findings": [{"id": oid(f.id, "F"), "rule": f.rule_id, "severity": f.severity.value,
                      "near_threshold": f.trigger.value == "uncertain",
                      "evidence_ids": [oid(x, "E") for x in f.evidence_ids]}
                     | ({"message": f.message} if share else {}) for f in report.details.findings],
        "decisions": [{"id": oid(d.id, "D"), "rule": d.rule_id, "class": d.cls.value, "status": d.status.value}
                      | ({"title": d.title, "action": d.action} if share else {}) for d in report.decisions],
        "questions": [{"id": oid(q.id, "Q"), "options": [o.value for o in q.options]}
                      | ({"text": q.text} if share else {}) for q in report.questions if q.answer is None],
    }
    if share:
        payload["task_reason"] = p.target.task_evidence.get("reason")
    if cfg.include_context and report.details.context.text:
        payload["user_context"] = report.details.context.text
    reverse = {v: k for k, v in ids.items()} | {v: k for k, v in col_alias.items()}
    return payload, reverse


def disclosure(cfg: LLMConfig, provider: str, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": provider, "model": model,
        "column names sent": cfg.share_column_names,
        "free-text context sent": "user_context" in payload,
        "fields sent": ", ".join(payload),
        "never sent": ", ".join(NEVER_SENT),
        "payload size (chars)": len(str(payload)),
    }
