"""Plain-text / markdown render. `print(report)` shows the plain-text form.

Short by design: what needs you, then one line of why / do per decision.
Evidence tables, rule versions and the profile live in `report.details` and in
the HTML report's Details section.
"""

from __future__ import annotations

from typing import Any

from ..rules.metrics import label, natural
from .common import (
    CLASS_LABEL,
    attention,
    experiment_summary,
    fmt,
    headline,
    key_evidence,
    next_lines,
    partition_note,
    state_line,
    status_word,
    task_dependence_note,
)


def render_text(report: Any, md: bool) -> str:
    H = (lambda t: f"## {t}") if md else (lambda t: t.upper())
    B = (lambda t: f"**{t}**") if md else (lambda t: t)
    lines = ["# prefit report" if md else "PREFIT REPORT", "", headline(report), "",
             B("Status: ") + state_line(report) + " (done / needs you / tested; details: report.decisions_table())",
             "", "Data partition: " + partition_note(report), "", B("Run next:")]
    code = next_lines(report)
    lines += (["```python", *code, "```"] if md else [f"    {c}" for c in code]) + [""]

    att = attention(report)
    if att:
        lines.append(H("Needs your attention"))
        lines += [f"- [{status_word(d.status)}] {d.title} ({d.id})" for d in att]
        lines.append("")

    pq = report.pending_questions()
    if pq:
        lines.append(H("Questions for you"))
        for q in pq:
            opts = " | ".join(f"{o.value}: {o.label}" for o in q.options)
            lines.append(f"- {q.id[2:]}: {q.text}  [{opts}]")
            if q.suggestion:
                lines.append(f"    suggestion awaiting your confirmation (LLM): {q.suggestion} — {q.suggestion_reason}")
        lines.append("")

    lines.append(H("Roadmap"))
    for st in report.roadmap:
        lines.append(B(st.title) + f" — {st.summary}")
        standard = [report.decision(i) for i in st.decision_ids if report.decision(i).standard]
        for did in st.decision_ids:
            d = report.decision(did)
            if d.standard:
                continue
            lines.append(f"- [{CLASS_LABEL[d.cls]} · {status_word(d.status)}] {d.title}")
            ke = key_evidence(report, d)
            lines.append(f"    why: {d.why_short}" + (f" ({ke})" if ke else ""))
            lines.append(f"    do:  {d.do_short}")
            ex = experiment_summary(report, d)
            if ex:
                lines.append(f"    experiment: {ex}")
            for alt in d.alternatives:
                lines.append(f"    or:  {alt}")
            if d.note:
                lines.append(f"    note: {d.note}")
            if d.review_note:
                lines.append(f"    your note: {d.review_note}")
            tn = task_dependence_note(report, d)
            if tn:
                lines.append(f"    task: {tn}")
        if standard:
            lines.append("- standard practice (applied on every dataset): "
                         + "; ".join(f"{d.title} ({d.id})" for d in standard))
        lines.append("")

    mp = report.metric_plan
    lines.append(H("Metrics"))
    lines.append(f"- optimize/report: {', '.join(label(m) for m in mp.primary)}"
                 + (" (provisional until you answer)" if mp.dev_metric_provisional else ""))
    if mp.report_alongside:
        lines.append(f"- alongside: {', '.join(label(m) for m in mp.report_alongside)}")
    for m, why in mp.avoid.items():
        lines.append(f"- avoid {m}: {why}")
    b = report.baseline
    if b.get("families") and b.get("dummy", {}).get("mean") is not None:
        mt = b["metric"]
        fams = ", ".join(f"{k} {fmt(natural(mt, v['mean']))}" for k, v in b["families"].items())
        lines.append(f"- baseline {label(mt)} on {b['cv']}: dummy {fmt(natural(mt, b['dummy']['mean']))}; {fams}")
    lines.append("")

    llm = report.details.llm
    if llm:
        lines.append(H(f"Optional interpretation · LLM ({llm.provider})"))
        lines.append(llm.summary)
        for c in llm.priorities:
            lines.append(f"- {c.text} [{', '.join(c.evidence_ids + c.decision_ids)}]")
        for s in llm.semantic_flags:
            lines.append(f"- manual review required · {s.column}: {s.concern} — {s.reason}")
        lines.append("")

    lines.append(H("Limitations"))
    lines += [f"- {x}" for x in report.details.limitations]
    lines += ["", "Details (findings, evidence, profile, manifest): report.details · decisions as a table: "
                  "report.decisions_table()"]
    return "\n".join(lines).rstrip() + "\n"
