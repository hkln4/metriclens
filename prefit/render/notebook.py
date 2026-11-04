"""In-notebook view: `report.show()` and `report.show("d.xxx")`.

Shows only what bears on a decision that needs you or that is specific to this
data (standard practice stays in the HTML report). Figures are plotly JSON with
color tokens ("var:series-1"); they are resolved here against the light theme
of the HTML report. Without plotly installed, each figure's data table is shown
instead, so nothing is lost. Outside IPython the same content is printed.
"""

from __future__ import annotations

import re
from typing import Any

from .common import fmt, fmt_ci, key_evidence, status_word

_THEME: dict[str, str] | None = None


def theme() -> dict[str, str]:
    """CSS custom properties of the light theme, parsed from the HTML report's stylesheet."""
    global _THEME
    if _THEME is None:
        from .html import CSS

        block = CSS[CSS.index(":root{") + 6: CSS.index("}", CSS.index(":root{"))]
        _THEME = {k: v.strip() for k, v in re.findall(r"--([\w-]+):([^;]+);", block)}
    return _THEME


def resolve(x: Any) -> Any:
    """Replace color tokens ("var:name", "scale:diverging" / "scale:sequential") with colors."""
    t = theme()
    if isinstance(x, str):
        if x.startswith("var:"):
            return t.get(x[4:], "#888888")
        if x == "scale:diverging":
            return [[0, t.get("div-neg", "#2a78d6")], [0.5, t.get("div-mid", "#f0f0f0")],
                    [1, t.get("div-pos", "#eb6834")]]
        if x == "scale:sequential":
            return [[i / 4, t.get(f"seq-{i}", "#2a78d6")] for i in range(5)]
        return x
    if isinstance(x, (list, tuple)):
        return [resolve(v) for v in x]
    if isinstance(x, dict):
        return {k: resolve(v) for k, v in x.items()}
    return x


def figure(spec: Any) -> Any:
    """A plotly Figure for a FigureSpec, or None when plotly is not installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    plot = resolve(spec.plot)
    lay = dict(plot.get("layout") or {})
    lay.setdefault("title", {"text": spec.title})
    lay.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font={"family": "system-ui, sans-serif", "size": 12, "color": theme().get("text-secondary", "#444")})
    return go.Figure(data=plot.get("data") or [], layout=lay)


def figure_table(spec: Any) -> Any:
    import pandas as pd

    t = spec.table or {}
    return pd.DataFrame(t.get("rows", []), columns=t.get("columns") or None)


def shown_figures(report: Any) -> list[Any]:
    """The figures worth showing in a notebook: those behind decisions that need you, or
    that are specific to this data (not standard practice), at most one per decision."""
    figs = {f.id: f for f in report.details.figures}
    out, seen = [], set()
    keep = [d for d in report.decisions if not d.standard and (d.state != "done" or d.cls.value == "review")]
    keep.sort(key=lambda d: (d.state != "needs_you",))
    for d in keep:
        fid = next((f for f in d.figure_ids if f in figs and f not in seen), None)
        if fid:
            seen.add(fid)
            out.append((d, figs[fid]))
    if "fig.target" in figs and "fig.target" not in seen:
        out.insert(0, (None, figs["fig.target"]))
    return out


def _ipython() -> Any:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, Markdown, display
    except ImportError:
        return None
    if get_ipython() is None:
        return None
    return display, HTML, Markdown


def _show_figure(spec: Any, ip: Any) -> None:
    fig = figure(spec)
    if ip is None:
        print(f"\n[{spec.title}] {spec.what}")
        print(figure_table(spec).to_string(index=False))
        return
    display, _HTML, Markdown = ip
    display(Markdown(f"**{spec.title}** — {spec.what} *Why this figure:* {spec.why}"))
    display(fig if fig is not None else figure_table(spec))


def show(report: Any, decision_id: str | None = None) -> None:
    ip = _ipython()
    if decision_id is not None:
        _show_decision(report, decision_id if decision_id.startswith("d.") else f"d.{decision_id}", ip)
        return
    if ip is None:
        print(report.to_text())
        print("FIRST LOOK")
        print(report.first_look().to_string(index=False))
    else:
        display, HTML, Markdown = ip
        display(HTML(report._repr_html_()))
        display(Markdown("**First look** — one row per column; *decisions* lists the decisions that mention it "
                         "(`report.show(\"d.…\")` for one of them)."))
        display(report.first_look())
    for _d, spec in shown_figures(report):
        _show_figure(spec, ip)


def _show_decision(report: Any, did: str, ip: Any) -> None:
    import pandas as pd

    d = report.decision(did)
    ev = report.details.evidence
    rows = [{"measure": ev[e].label, "value": fmt(ev[e].value), "95% interval": fmt_ci(ev[e].ci),
             "method": ev[e].method} for e in d.evidence_ids if e in ev]
    lines = [f"{d.title}  [{status_word(d.status)} · {d.cls.value}]", f"why: {d.why_short}"
             + (f" ({key_evidence(report, d)})" if key_evidence(report, d) else ""), f"do:  {d.do_short}"]
    lines += [f"or:  {a}" for a in d.alternatives]
    if d.risk:
        lines.append(f"risk: {d.risk}")
    if d.note:
        lines.append(f"note: {d.note}")
    if ip is None:
        print("\n".join(lines))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    else:
        display, _HTML, Markdown = ip
        display(Markdown("\n\n".join(f"**{x}**" if i == 0 else x for i, x in enumerate(lines))))
        if rows:
            display(pd.DataFrame(rows))
    figs = {f.id: f for f in report.details.figures}
    for fid in d.figure_ids:
        if fid in figs:
            _show_figure(figs[fid], ip)
