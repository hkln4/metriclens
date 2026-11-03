"""Single-file HTML report.

Reading order: Summary (what this is, how much needs you, the next step) ->
Actions (what needs your attention, open questions, metrics) -> Roadmap (one
short card per decision: status, title, one-line why with the decisive number,
one-line do, experiment summary, at most one figure) -> Details (collapsed:
evidence, findings, all diagnostics, profile, manifest). Everything that
explains a decision is one click away inside its card.

Figures are plotly JSON with color tokens resolved from CSS custom properties,
so light and dark mode each use their own steps. With plotlyjs="cdn" (default)
the file is a single file but needs internet for the charts; "inline" embeds
plotly.js for fully offline use. Every figure also has a data table, which is
what shows when plotly.js is unavailable. No raw rows are embedded.
"""

from __future__ import annotations

import html
import json
from typing import Any

from ..enums import DecisionClass, DecisionStatus, Severity
from ..rules.metrics import label, natural
from .common import (
    CLASS_HELP,
    CLASS_LABEL,
    ICON,
    SEV_TONE,
    attention,
    effect,
    experiment_summary,
    fmt,
    fmt_ci,
    headline,
    key_evidence,
    next_step,
    partition_note,
    status_word,
    summary_counts,
    task_dependence_note,
    task_note,
    tone,
)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def aid(x: str) -> str:
    """Anchor id: ids are already restricted, but escape anyway."""
    return esc(x)


CSS = """
:root{color-scheme:light;
--surface:#fcfcfb;--surface-2:#f3f2ef;--border:#dddcd6;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#76756f;--grid:#e6e5e0;--link:#1c5cab;
--series-1:#2a78d6;--series-1-soft:rgba(42,120,214,.18);--series-2:#eb6834;--series-3:#1baf7a;
--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--seq-0:#cde2fb;--seq-1:#86b6ef;--seq-2:#3987e5;--seq-3:#1c5cab;--seq-4:#0d366b;
--div-neg:#2a78d6;--div-mid:#f0efec;--div-pos:#e34948;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--surface-2:#242423;--border:#3a3a36;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--text-muted:#94938b;--grid:#33332f;--link:#86b6ef;
--series-1:#3987e5;--series-1-soft:rgba(57,135,229,.28);--series-2:#d95926;--series-3:#199e70;
--seq-0:#0d366b;--seq-1:#184f95;--seq-2:#2a78d6;--seq-3:#6da7ec;--seq-4:#cde2fb;
--div-neg:#3987e5;--div-mid:#383835;--div-pos:#e66767;}}
:root[data-theme="dark"]{color-scheme:dark;
--surface:#1a1a19;--surface-2:#242423;--border:#3a3a36;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--text-muted:#94938b;--grid:#33332f;--link:#86b6ef;
--series-1:#3987e5;--series-1-soft:rgba(57,135,229,.28);--series-2:#d95926;--series-3:#199e70;
--seq-0:#0d366b;--seq-1:#184f95;--seq-2:#2a78d6;--seq-3:#6da7ec;--seq-4:#cde2fb;
--div-neg:#3987e5;--div-mid:#383835;--div-pos:#e66767;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text-primary);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
a{color:var(--link)}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 12px;padding-top:8px;
border-top:1px solid var(--border)}h3{font-size:1.02rem;margin:0}
.sub{color:var(--text-secondary);margin:0 0 16px}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.9rem;margin:12px 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.chip{border:1px solid var(--border);border-radius:999px;padding:2px 10px;font-size:.85rem;color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--surface-2);border-radius:10px;padding:12px}
.tile b{display:block;font-size:1.6rem;line-height:1.1}.tile span{color:var(--text-secondary);font-size:.85rem}
.badge{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--border);border-radius:6px;
padding:1px 8px;font-size:.78rem;white-space:nowrap;color:var(--text-primary);background:var(--surface)}
.badge i{font-style:normal;font-weight:700}
.t-good i{color:var(--good)}.t-warning i{color:#b07a00}.t-critical i{color:var(--critical)}
.t-serious i{color:var(--serious)}.t-neutral i{color:var(--text-muted)}
.t-good{border-left:4px solid var(--good)}.t-warning{border-left:4px solid var(--warning)}
.t-critical{border-left:4px solid var(--critical)}.t-serious{border-left:4px solid var(--serious)}
.card{border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:12px 0;background:var(--surface)}
.card:target{outline:2px solid var(--series-1)}
.card header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px}
.card header h3{flex:1 1 260px}
.kv{display:grid;grid-template-columns:110px 1fr;gap:4px 12px;margin:8px 0}
.kv dt{color:var(--text-secondary);font-size:.88rem}.kv dd{margin:0}
.tw{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:6px 0}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--text-secondary);font-weight:600}
td.num{font-variant-numeric:tabular-nums}
details{margin:8px 0}summary{cursor:pointer;color:var(--text-secondary)}
.fig{border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:14px 0}
.fig:target{outline:2px solid var(--series-1)}
.fig p{margin:4px 0;color:var(--text-secondary);font-size:.9rem}
.plot{width:100%;min-height:180px}
.links{font-size:.85rem;color:var(--text-secondary)}
.muted{color:var(--text-muted)}
.stage{margin:10px 0}.stage ul{margin:4px 0;padding-left:0;list-style:none}
.stage li{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:4px 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85rem}
pre{background:var(--surface-2);padding:10px;border-radius:8px;overflow-x:auto}
.llm{border:1px dashed var(--border);border-radius:12px;padding:14px 16px}
.noplot .plot{display:none}
.partition{border:1px solid var(--border);border-left:4px solid var(--series-1);border-radius:8px;
background:var(--surface-2);padding:10px 12px;margin:12px 0;font-size:.92rem}
.next{font-size:1.02rem;margin:12px 0;padding:10px 12px;border-radius:8px;background:var(--series-1-soft)}
.card p{margin:4px 0}.card .lbl{color:var(--text-secondary);font-size:.85rem;margin-right:6px}
.card details{margin:4px 0}.card details>summary{font-size:.88rem}
.card .plot{min-height:160px;margin:6px 0}
.attn{list-style:none;padding-left:0}.attn li{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:3px 0}
.q{border:1px solid var(--border);border-radius:10px;padding:10px 14px;margin:10px 0}
.suggest{border-left:4px solid var(--warning);padding:4px 10px;margin:6px 0;background:var(--surface-2)}
section.details>details{border:1px solid var(--border);border-radius:10px;padding:8px 12px;margin:8px 0}
section.details>details>summary{font-weight:600;color:var(--text-primary)}
"""

JS = r"""
(function(){
  var css = function(n){return getComputedStyle(document.documentElement).getPropertyValue('--'+n).trim();};
  function scale(name){
    if(name==='scale:diverging') return [[0,css('div-neg')],[0.5,css('div-mid')],[1,css('div-pos')]];
    if(name==='scale:sequential') return [0,1,2,3,4].map(function(i){return [i/4, css('seq-'+i)];});
    return name;
  }
  function resolve(x){
    if(typeof x==='string'){
      if(x.indexOf('var:')===0) return css(x.slice(4));
      if(x.indexOf('scale:')===0) return scale(x);
      return x;
    }
    if(Array.isArray(x)) return x.map(resolve);
    if(x && typeof x==='object'){var o={}; for(var k in x){o[k]=resolve(x[k]);} return o;}
    return x;
  }
  var specs = JSON.parse(document.getElementById('prefit-figures').textContent);
  function draw(){
    if(typeof Plotly==='undefined'){document.body.classList.add('noplot');
      document.querySelectorAll('.fig details.data').forEach(function(d){d.open=true;}); return;}
    document.querySelectorAll('.plot[data-fig]').forEach(function(el){
      if(el.offsetParent===null) return;            // inside a closed <details>: drawn when opened
      var spec=specs[el.getAttribute('data-fig')]; if(!spec) return;
      var s=resolve(spec); var lay=s.layout||{};
      lay.paper_bgcolor='rgba(0,0,0,0)'; lay.plot_bgcolor='rgba(0,0,0,0)';
      lay.font={family:'system-ui,-apple-system,Segoe UI,Roboto,sans-serif',size:12,color:css('text-secondary')};
      lay.hoverlabel={bgcolor:css('surface'),bordercolor:css('border'),font:{color:css('text-primary')}};
      Plotly.react(el,s.data,lay,{displaylogo:false,responsive:true,
        modeBarButtonsToRemove:['lasso2d','select2d']});
    });
  }
  window.addEventListener('load',draw);
  document.addEventListener('toggle',function(e){ if(e.target.open) draw(); },true);
  if(window.matchMedia){var m=window.matchMedia('(prefers-color-scheme: dark)');
    if(m.addEventListener) m.addEventListener('change',draw);}
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();
"""


def _badge(text: str, t: str, title: str = "") -> str:
    tt = f' title="{esc(title)}"' if title else ""
    return f'<span class="badge t-{t}"{tt}><i aria-hidden="true">{ICON.get(t, "·")}</i>{esc(text)}</span>'


def status_badge(s: DecisionStatus) -> str:
    return _badge(status_word(s), tone(s))


def class_badge(c: DecisionClass) -> str:
    return _badge(CLASS_LABEL[c], "neutral", CLASS_HELP[c])


def sev_badge(s: Severity, near: bool = False) -> str:
    return _badge(s.value + (" · near threshold" if near else ""), SEV_TONE[s])


def table(columns: list[str], rows: list[list[Any]], raw: bool = False) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(
        f'<td class="num">{c if raw else esc(fmt(c))}</td>' for c in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _link(target: str, text: str | None = None) -> str:
    return f'<a href="#{aid(target)}">{esc(text or target)}</a>'


def _figure(f: Any, data_open: bool = False) -> str:
    links = []
    if f.evidence_ids:
        links.append("evidence: " + ", ".join(_link(e) for e in f.evidence_ids))
    if f.decision_ids:
        links.append("decisions: " + ", ".join(_link(d) for d in f.decision_ids))
    tbl = table(f.table["columns"], f.table["rows"]) if f.table else ""
    return (f'<figure class="fig" id="{aid(f.id)}" style="margin:14px 0"><h3>{esc(f.title)}</h3>'
            f'<p><b>Shows:</b> {esc(f.what)}</p><p><b>Why this figure:</b> {esc(f.why)}</p>'
            f'<div class="plot" data-fig="{aid(f.id)}" role="img" aria-label="{esc(f.title)}"></div>'
            f'<p class="links">{" · ".join(links)}</p>'
            + (f'<details class="data"><summary>Data table</summary>{tbl}</details>' if tbl else "") + "</figure>")


def _evidence_table(r: Any, ids: Any) -> str:
    rows = []
    for eid in ids:
        e = r.details.evidence.get(eid)
        if e is None:
            continue
        rows.append([_link(eid, e.label), esc(fmt(e.value)), esc(fmt_ci(e.ci)), esc(e.method),
                     ", ".join(_link(x) for x in e.figure_ids)])
    return table(["evidence", "value", "95% interval", "method", "figures"], rows, raw=True) if rows else ""


def _decision_card(r: Any, d: Any, inline_figs: set[str]) -> str:
    """Short card, open: status, title, why (+ decisive number), do, experiment summary,
    at most one figure. Everything else sits in closed <details>."""
    parts = [f'<article class="card" id="{aid(d.id)}" data-decision="{aid(d.id)}"><header>'
             f"{status_badge(d.status)}{class_badge(d.cls)}"
             + (_badge("standard practice", "neutral", "Appears on every dataset; applied as a matter of course.")
                if d.standard else "")
             + f"<h3>{esc(d.title)}</h3></header>"]
    ke = key_evidence(r, d)
    parts.append(f'<p class="why"><span class="lbl">Why</span>{esc(d.why_short)}'
                 + (f' <span class="muted">({esc(ke)})</span>' if ke else "") + "</p>")
    parts.append(f'<p class="do"><span class="lbl">Do</span>{esc(d.do_short)}</p>')
    ex = experiment_summary(r, d)
    if ex:
        parts.append(f'<p class="exp"><span class="lbl">Experiment</span>{esc(ex)}</p>')
    if d.note:
        parts.append(f'<p class="note"><span class="lbl">Note</span>{esc(d.note)}</p>')
    if d.review_note:
        parts.append(f'<p class="note"><span class="lbl">Your note</span>{esc(d.review_note)}</p>')
    tn = task_dependence_note(r, d)
    if tn:
        parts.append(f'<p class="note"><span class="lbl">Task</span>{esc(tn)}</p>')
    if d.question_id and d.status is DecisionStatus.AWAITING_USER:
        parts.append(f'<p><span class="lbl">Answer</span>{_link(d.question_id, "open question")}</p>')

    main = next((f for f in d.figure_ids if f not in inline_figs), None) if d.figure_ids else None
    if main is not None:
        inline_figs.add(main)
        parts.append(f'<div class="plot" data-fig="{aid(main)}" data-decision="{aid(d.id)}" role="img" '
                     f'aria-label="{esc(main)}"></div>')

    ev = _evidence_table(r, d.evidence_ids)
    if ev:
        parts.append(f'<details class="more"><summary>Show evidence</summary>{ev}</details>')
    exps = [e for e in r.experiments if e.decision_id == d.id]
    if exps:
        erows = []
        for e in exps:
            change, ci, direction = effect(e)
            sign = -1 if e.metric.startswith("neg_") else 1
            erows.append([e.family, e.estimator, f"{label(e.metric)} ({direction})", fmt(change), fmt_ci(ci),
                          fmt(e.p_adjusted), fmt(e.p_equivalence), f"±{fmt(e.min_effect)}",
                          f"{e.folds_improved}/{len(e.fold_diffs)}",
                          ", ".join(f"{sign * x:+.3g}" for x in e.fold_diffs), e.outcome])
        parts.append('<details class="more"><summary>Show experiment details</summary>'
                     + table(["family", "estimator", "metric", "change", "95% CI", "p (Holm)", "equivalence p (Holm)",
                              "practical threshold", "folds improved", "per-fold changes", "outcome"], erows)
                     + f'<p class="links">CV: {esc(exps[0].cv)} · {esc(exps[0].method)} Development evidence on '
                       "the development rows only.</p></details>")
    others = [f for f in d.figure_ids if f != main]
    if others:
        parts.append('<details class="more"><summary>Show related figures</summary><p class="links">'
                     + " · ".join(_link(f) for f in others) + "</p></details>")
    rule = []
    if d.rule_id:
        fnds = [f for f in r.details.findings if f.id in d.finding_ids]
        ver = r.details.manifest.rule_versions.get(d.rule_id, "?")
        thr = "; ".join(f"{k} = {fmt(v)}" for k, v in (fnds[0].thresholds if fnds else {}).items())
        rule.append(f"<dt>Rule</dt><dd><code>{esc(d.rule_id)}</code> v{esc(ver)}"
                    + (f" · thresholds: {esc(thr)}" if thr else "") + "</dd>")
        tr = r.details.rule_track_record.get(d.rule_id)
        if tr:
            rule.append(f"<dt>Benchmark</dt><dd>rule detection pass rate on benchmark "
                        f"{esc(r.details.manifest.benchmark or '?')}: {tr['passed']}/{tr['checks']} checks "
                        "(known-hazard fixtures; not a real-world accuracy claim)</dd>")
        if fnds:
            rule.append("<dt>Findings</dt><dd>" + "; ".join(f"{_link(f.id)} {esc(f.message)}" for f in fnds) + "</dd>")
            cols = sorted({c for f in fnds for c in f.columns})
            if cols:
                rule.append(f"<dt>Columns</dt><dd>{esc(', '.join(cols))}</dd>")
    rule.append(f"<dt>Full action</dt><dd>{esc(d.action)}</dd><dt>Full reason</dt><dd>{esc(d.rationale)}</dd>")
    if d.risk:
        rule.append(f"<dt>Risk</dt><dd>{esc(d.risk)}</dd>")
    if d.verify_hint:
        rule.append(f"<dt>Check</dt><dd>{esc(d.verify_hint)}</dd>")
    rule.append(f'<dt>Class</dt><dd>{esc(CLASS_LABEL[d.cls])}: {esc(CLASS_HELP[d.cls])}</dd>')
    parts.append('<details class="more"><summary>Show rule and benchmark details</summary><dl class="kv">'
                 + "".join(rule) + "</dl></details></article>")
    return "".join(parts)


def _question(q: Any) -> str:
    opts = "".join(f"<li><code>{esc(o.value)}</code> — {esc(o.label)}"
                   + (f' <span class="muted">({esc(o.effect)})</span>' if o.effect else "") + "</li>"
                   for o in q.options)
    sug = ""
    if q.suggestion:
        sug = (f'<div class="suggest"><b>Suggested (awaiting your confirmation):</b> <code>{esc(q.suggestion)}</code>'
               f" — {esc(q.suggestion_reason or '')} <span class=\"muted\">(LLM interpretation; not applied)</span>"
               "</div>")
    dl = " · ".join(_link(x, "decision") for x in q.decision_ids)
    return (f'<div class="q" id="{aid(q.id)}"><b>{esc(q.text)}</b> <span class="links">{dl}</span><ul>{opts}</ul>'
            f'{sug}<code>report = report.answer({{"{esc(q.id[2:])}": "&lt;option&gt;"}})</code></div>')


def summary_card(r: Any) -> str:
    """Compact HTML for notebooks (Report._repr_html_): no plots, no details."""
    c = summary_counts(r)
    att = "".join(f"<li>{status_badge(d.status)} {esc(d.title)}</li>" for d in attention(r)[:8])
    return (f'<div style="font:14px/1.5 system-ui,sans-serif;border:1px solid #ccc;border-radius:10px;padding:12px">'
            f"<b>prefit report · {esc(r.target)}</b> (revision {r.details.manifest.revision})<br>"
            f"{esc(headline(r))}<br><b>Next step:</b> {esc(next_step(r))}"
            + (f"<br><b>Needs your attention</b><ul style='margin:4px 0'>{att}</ul>" if att else "")
            + f"<span style='color:#666'>{c['questions']} open question(s) · print(report) for the roadmap · "
              "report.decisions_table() · report.to_html(path)</span></div>")


def report_body(r: Any) -> tuple[str, dict[str, Any]]:
    p = r.details.profile
    c = summary_counts(r)
    figs = {f.id: f.plot for f in r.details.figures}
    ctx = r.details.context
    chips = [f"target: {r.target}", f"task: {r.task.value}" + task_note(r),
             f"{p.dataset.n_rows:,} development rows", f"{p.dataset.n_features} features"]
    if ctx.groups:
        chips.append(f"groups: {ctx.groups}")
    if ctx.time_col:
        chips.append(f"time: {ctx.time_col}")

    out = [f'<section id="summary"><h1>prefit report · {esc(r.target)}</h1><div class="chips">'
           + "".join(f'<span class="chip">{esc(x)}</span>' for x in chips) + "</div>"
           + '<div class="tiles">'
           f'<div class="tile"><b>{c["questions"]}</b><span>questions awaiting your answer</span></div>'
           f'<div class="tile"><b>{c["critical_or_review"]}</b><span>critical or review decisions</span></div>'
           "</div>"
           f'<p class="next"><b>Next step:</b> {esc(next_step(r))}</p>'
           f'<div class="partition" role="note"><b>Data partition.</b> {esc(partition_note(r))}</div></section>']

    # actions
    act = ['<section id="actions"><h2>Actions</h2>']
    att = attention(r)
    if att:
        act.append("<h3>Needs your attention</h3><ul class=\"attn\">" + "".join(
            f"<li>{status_badge(d.status)} {_link(d.id, d.title)}</li>" for d in att) + "</ul>")
    else:
        act.append("<p>Nothing needs your attention right now.</p>")
    pq = r.pending_questions()
    if pq:
        act.append(f"<h3>Questions ({len(pq)})</h3><p class=\"sub\">Contextual decisions stay out of the model recipe "
                   "until you answer. Answers re-derive the decisions from the stored profile and create a new "
                   "revision.</p>" + "".join(_question(q) for q in pq))
    mp = r.metric_plan
    act.append("<h3>Metrics</h3><dl class=\"kv\">"
               f"<dt>Optimize</dt><dd>{esc(', '.join(label(m) for m in mp.primary))}"
               + (' <span class="muted">(provisional until you answer)</span>' if mp.dev_metric_provisional else "")
               + "</dd>"
               + (f"<dt>Alongside</dt><dd>{esc(', '.join(label(m) for m in mp.report_alongside))}</dd>"
                  if mp.report_alongside else "")
               + "".join(f"<dt>Avoid</dt><dd><b>{esc(m)}</b> — {esc(w)}</dd>" for m, w in mp.avoid.items())
               + "</dl>")
    b = r.baseline
    if b.get("families"):
        mt = b["metric"]
        rows = [["dummy", "class prior" if r.task.value == "classification" else "median",
                 fmt(natural(mt, b.get("dummy", {}).get("mean"))),
                 fmt(b.get("dummy", {}).get("std"))]]
        rows += [[k, v["estimator"], fmt(natural(mt, v["mean"])), fmt(v["std"])] for k, v in b["families"].items()]
        act.append(f"<details class=\"more\"><summary>Baseline ({esc(label(mt))}, {esc(b['cv'])}, "
                   f"{b['n_rows']:,} dev rows)</summary>" + table(["model", "estimator", "mean", "std"], rows)
                   + "</details>")
    L = r.details.llm
    if L:
        act.append('<h3>Optional interpretation · LLM</h3><div class="llm">'
                   f'<p class="sub">Written by {esc(L.provider)} / {esc(L.model)} from the computed report only. It '
                   "cannot change any decision; every claim cites existing evidence or decision ids. "
                   f"{L.dropped_claims} item(s) were dropped (unknown ids or columns, invalid answer options)."
                   f"</p><p>{esc(L.summary)}</p>")
        if L.priorities:
            act.append("<ul>" + "".join(f"<li>{esc(cl.text)} <span class=\"links\">"
                                        + ", ".join(_link(x) for x in cl.evidence_ids + cl.decision_ids)
                                        + "</span></li>" for cl in L.priorities) + "</ul>")
        if L.semantic_flags:
            act.append("<p><b>Semantic flags · manual review required</b></p><ul>" + "".join(
                f"<li><b>{esc(s.column)}</b>: {esc(s.concern)} — {esc(s.reason)}</li>" for s in L.semantic_flags)
                + "</ul>")
        if L.dropped:
            act.append("<details class=\"more\"><summary>What was dropped</summary><ul>"
                       + "".join(f"<li>{esc(x)}</li>" for x in L.dropped) + "</ul></details>")
        if L.raw_response:
            act.append("<details class=\"more\"><summary>Raw response (as received)</summary><pre>"
                       + esc(L.raw_response) + "</pre></details>")
        act.append("<details class=\"more\"><summary>What was sent</summary>"
                   + table(["field", "value"], [[k, v] for k, v in L.disclosure.items()]) + "</details></div>")
    act.append("</section>")
    out += act

    # roadmap: short cards, consecutive stage numbers
    inline: set[str] = set()
    out.append('<section id="roadmap"><h2>Roadmap</h2><p class="sub">Ordered from split to final evaluation. '
               "Only applied, confirmed or verified steps enter the exported model recipe.</p>")
    for st in r.roadmap:
        out.append(f'<h3 style="margin-top:22px">{esc(st.title)} <span class="muted">· {esc(st.summary)}</span></h3>')
        ds = [r.decision(i) for i in st.decision_ids]
        out += [_decision_card(r, d, inline) for d in ds if not d.standard]
        std = [d for d in ds if d.standard]
        if std:
            out.append(f'<details class="more"><summary>Standard practice ({len(std)}): '
                       + esc("; ".join(d.title for d in std)) + "</summary>"
                       + "".join(_decision_card(r, d, inline) for d in std) + "</details>")
    out.append("</section>")

    # details: limitations visible, everything else collapsed
    d = r.details
    out.append('<section id="details" class="details"><h2>Details</h2><h3>Limitations</h3><ul>'
               + "".join(f"<li>{esc(x)}</li>" for x in d.limitations) + "</ul>")
    rest = [f for f in d.figures if f.id not in inline]
    out.append(f"<details id=\"all-diagnostics\"><summary>All diagnostics ({len(d.figures)} figures)</summary>"
               + "".join(_figure(f) for f in d.figures) + "</details>" if d.figures else "")
    del rest
    frows = [[sev_badge(f.severity, f.trigger.value == "uncertain"), f'<span id="{aid(f.id)}">{esc(f.message)}</span>',
              ", ".join(_link(e) for e in f.evidence_ids), f"<code>{esc(f.rule_id)}</code> v{esc(f.rule_version)}",
              esc("; ".join(f"{k}={fmt(v)}" for k, v in f.thresholds.items()))] for f in d.findings]
    out.append(f"<details id=\"findings\"><summary>Findings ({len(frows)})</summary>"
               + (table(["severity", "finding", "evidence", "rule", "thresholds"], frows, raw=True) if frows
                  else "<p>No rule fired.</p>") + "</details>")
    erows = [[f'<span id="{aid(e.id)}"><code>{esc(e.id)}</code></span>', esc(e.label), esc(fmt(e.value)),
              esc(fmt_ci(e.ci)), esc(e.method), ", ".join(_link(x) for x in e.figure_ids)]
             for e in d.evidence.values()]
    out.append(f"<details id=\"evidence\"><summary>Evidence ({len(erows)})</summary>"
               + table(["id", "measure", "value", "95% interval", "method", "figures"], erows, raw=True)
               + "</details>")
    ds = p.dataset
    crow = [[esc(cp.name), esc(cp.kind.value), esc(cp.kind_reason), esc(cp.dtype), esc(f"{cp.missing_fraction:.1%}"),
             esc(cp.cardinality), esc(fmt(cp.stats.get("skew"))), esc(fmt(cp.association))]
            for cp in p.columns.values()]
    out.append("<details id=\"profile\"><summary>Dataset profile</summary>" + table(["measure", "value"], [
        ["rows passed to analyze", ds.n_rows_input], ["development rows profiled", ds.n_rows],
        ["features", ds.n_features], ["column kinds", ds.kind_counts],
        ["exact duplicate rows", f"{ds.duplicate_rows} ({ds.duplicate_fraction:.2%})"],
        ["identical features, different target", ds.conflicting_duplicate_groups],
        ["target missing", f"{p.target.n_missing} ({p.target.missing_fraction:.1%})"],
        ["task", f"{p.task.value} — {p.target.task_evidence.get('reason', '')}"]])
        + "<h3>All columns</h3>"
        + table(["column", "kind", "why this kind", "dtype", "missing", "distinct", "skew",
                 "single-feature 2-fold CV score"], crow, raw=True) + "</details>")
    m = d.manifest
    out.append("<details id=\"manifest\"><summary>Manifest and history</summary>"
               + table(["field", "value"], [["prefit", m.prefit_version], ["python", m.python],
                                            ["packages", m.packages], ["seed", m.seed],
                                            ["rule versions", m.rule_versions], ["profile options", m.profile_options],
                                            ["budget", m.budget], ["benchmark manifest", m.benchmark],
                                            ["source fingerprint", m.source_fingerprint],
                                            ["external holdout", m.external_holdout], ["revision", m.revision]])
               + "<h3>History</h3>" + table(["revision", "step", "details"],
                                             [[h_["revision"], h_["action"],
                                               {k: v for k, v in h_.items() if k not in ("revision", "action")}]
                                              for h_ in m.history]) + "</details></section>")
    return "".join(out), figs


NAV = [("summary", "Summary"), ("actions", "Actions"), ("roadmap", "Roadmap"), ("details", "Details")]


def shell(title: str, body: str, figures: dict[str, Any], plotlyjs: str, nav: list[tuple[str, str]] | None = None) -> str:
    if plotlyjs == "cdn":
        script = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    elif plotlyjs == "inline":
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as exc:
            raise ImportError("plotlyjs='inline' needs the plotly package: pip install 'prefit[report]'") from exc
        script = f"<script>{get_plotlyjs()}</script>"
    elif plotlyjs == "none":
        script = ""
    else:
        raise ValueError("plotlyjs must be 'cdn', 'inline' or 'none'.")
    nav_html = "".join(f'<a href="#{k}">{v}</a>' for k, v in (nav if nav is not None else NAV))
    data = json.dumps(figures, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style>{script}</head><body><main>"
            f"<nav aria-label=\"sections\">{nav_html}</nav>{body}</main>"
            f'<script type="application/json" id="prefit-figures">{data}</script><script>{JS}</script></body></html>')


def render_report_html(r: Any, review: Any = None, plotlyjs: str = "cdn") -> str:
    body, figs = report_body(r)
    if review is not None:
        rb, rf = review.html_section()
        idx = body.index('<section id="details"')
        body = body[:idx] + rb + body[idx:]
        figs.update(rf)
    return shell(f"prefit · {r.target}", body, figs, plotlyjs)
