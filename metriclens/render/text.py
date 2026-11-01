"""Plain-text / markdown render. `print(report)` shows the plain-text form."""

from __future__ import annotations

from typing import Any

from ..rules.metrics import label, natural
from .common import CLASS_LABEL, fmt, fmt_ci, headline, priorities, status_word


def render_text(report: Any, md: bool) -> str:
    H = (lambda t: f"## {t}") if md else (lambda t: t.upper())
    B = (lambda t: f"**{t}**") if md else (lambda t: t)
    dmap = {d.id: d for d in report.decisions}
    lines = ["# metriclens report" if md else "METRICLENS REPORT", "", headline(report), ""]

    pr = priorities(report)
    if pr:
        lines += [H("Needs your attention")]
        lines += [f"- [{status_word(d.status)}] {d.title} ({d.id})" for d in pr]
        lines.append("")

    lines.append(H("Roadmap"))
    for st in report.roadmap:
        lines.append(B(f"{st.title}") + f" — {st.summary}")
        for did in st.decision_ids:
            d = dmap[did]
            lines.append(f"- [{CLASS_LABEL[d.cls]} · {status_word(d.status)}] {d.title}")
            lines.append(f"    do:  {d.action}")
            lines.append(f"    why: {d.rationale}")
            if d.note:
                lines.append(f"    note: {d.note}")
        lines.append("")

    if report.pending_questions:
        lines.append(H("Questions for you"))
        for q in report.pending_questions:
            opts = " | ".join(f"{o.value}: {o.label}" for o in q.options)
            lines.append(f"- {q.id[2:]}: {q.text}  [{opts}]")
            if q.suggestion:
                lines.append(f"    LLM suggestion (not applied): {q.suggestion} — {q.suggestion_reason}")
        lines.append('  answer with: report = report.answer({"<question>": "<option>"})')
        lines.append("")

    mp = report.metric_plan
    lines.append(H("Metrics"))
    lines.append(f"- optimize/report: {', '.join(label(m) for m in mp.primary)}"
                 + (" (provisional)" if mp.dev_metric_provisional else ""))
    if mp.report_alongside:
        lines.append(f"- alongside: {', '.join(label(m) for m in mp.report_alongside)}")
    for m, why in mp.avoid.items():
        lines.append(f"- avoid {m}: {why}")
    lines += [f"- note: {n}" for n in mp.notes]
    lines.append("")

    if report.findings:
        lines.append(H("Findings"))
        for f in report.findings:
            lines.append(f"- [{f.severity.value}{'' if f.trigger.value == 'fired' else ', near threshold'}] "
                         f"{f.message}  (rule {f.rule_id} v{f.rule_version})")
            for eid in f.evidence_ids:
                e = report.evidence[eid]
                lines.append(f"    {e.label}: {fmt(e.value)} {fmt_ci(e.ci)}  ({e.method})")
        lines.append("")

    if report.experiments:
        lines.append(H("Experiments (development evidence)"))
        for e in report.experiments:
            lines.append(f"- {e.decision_id} · {e.family} ({e.estimator}) · {label(e.metric)}: "
                         f"diff {fmt(e.mean_diff)} {fmt_ci(e.ci)}, Holm p={fmt(e.p_adjusted)} -> {e.outcome}")
        if report.baseline.get("dummy", {}).get("mean") is not None:
            b = report.baseline
            mt = b["metric"]
            fams = ", ".join(f"{k} {fmt(natural(mt, v['mean']))}" for k, v in b["families"].items())
            lines.append(f"- baseline {label(mt)}: dummy {fmt(natural(mt, b['dummy']['mean']))}; {fams}")
        lines.append("")

    if report.llm:
        lines.append(H(f"Optional interpretation · LLM ({report.llm.provider})"))
        lines.append(report.llm.summary)
        for c in report.llm.priorities:
            lines.append(f"- {c.text} [{', '.join(c.evidence_ids + c.decision_ids)}]")
        for s in report.llm.semantic_flags:
            lines.append(f"- manual review required · {s.column}: {s.concern} — {s.reason}")
        lines.append("")

    lines.append(H("Limitations"))
    lines += [f"- {x}" for x in report.limitations]
    return "\n".join(lines).rstrip() + "\n"
