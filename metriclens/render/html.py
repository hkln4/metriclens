"""Single-file HTML report.

The unit of the report is the decision card: every card links to the evidence
it rests on (value, interval, method), the rule version and thresholds, any
experiment, and the figures that show it. Figures are plotly JSON with color
tokens resolved from CSS custom properties, so light and dark mode each use their
own steps; every figure also has a data table, which is what you see if
plotly.js is unavailable. No raw rows are embedded.
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
    fmt,
    fmt_ci,
    headline,
    priorities,
    status_word,
    summary_counts,
    tone,
)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MAIN_FIGURES = 6


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
  var specs = JSON.parse(document.getElementById('ml-figures').textContent);
  function draw(){
    if(typeof Plotly==='undefined'){document.body.classList.add('noplot');
      document.querySelectorAll('.fig details').forEach(function(d){d.open=true;}); return;}
    Object.keys(specs).forEach(function(id){
      var el=document.getElementById('plot-'+id); if(!el) return;
      var s=resolve(specs[id]); var lay=s.layout||{};
      lay.paper_bgcolor='rgba(0,0,0,0)'; lay.plot_bgcolor='rgba(0,0,0,0)';
      lay.font={family:'system-ui,-apple-system,Segoe UI,Roboto,sans-serif',size:12,color:css('text-secondary')};
      lay.hoverlabel={bgcolor:css('surface'),bordercolor:css('border'),font:{color:css('text-primary')}};
      Plotly.react(el,s.data,lay,{displaylogo:false,responsive:true,
        modeBarButtonsToRemove:['lasso2d','select2d']});
    });
  }
  window.addEventListener('load',draw);
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


def _figure(f: Any) -> str:
    links = []
    if f.evidence_ids:
        links.append("evidence: " + ", ".join(_link(e) for e in f.evidence_ids))
    if f.decision_ids:
        links.append("decisions: " + ", ".join(_link(d) for d in f.decision_ids))
    tbl = table(f.table["columns"], f.table["rows"]) if f.table else ""
    return (f'<figure class="fig" id="{aid(f.id)}" style="margin:14px 0"><h3>{esc(f.title)}</h3>'
            f'<p><b>Shows:</b> {esc(f.what)}</p><p><b>Why this figure:</b> {esc(f.why)}</p>'
            f'<div class="plot" id="plot-{aid(f.id)}" role="img" aria-label="{esc(f.title)}"></div>'
            f'<p class="links">{" · ".join(links)}</p>'
            + (f"<details><summary>Data table</summary>{tbl}</details>" if tbl else "") + "</figure>")


def _decision_card(r: Any, d: Any) -> str:
    rows = []
    for eid in d.evidence_ids:
        e = r.evidence.get(eid)
        if e is None:
            continue
        figs = ", ".join(_link(x) for x in e.figure_ids)
        rows.append([f'<span id="{aid(eid)}-{aid(d.id)}">{esc(e.label)}</span>', esc(fmt(e.value)),
                     esc(fmt_ci(e.ci)), esc(e.method), figs])
    ev_html = table(["evidence", "value", "95% interval", "method", "figures"], rows, raw=True) if rows else ""
    rule_html = ""
    findings = [f for f in r.findings if f.id in d.finding_ids]
    if d.rule_id:
        ver = r.manifest.rule_versions.get(d.rule_id, "?")
        thr = "; ".join(f"{k} = {fmt(v)}" for k, v in (findings[0].thresholds if findings else {}).items())
        tr = r.rule_track_record.get(d.rule_id)
        trs = (f" · benchmark: correct on {tr['correct']}/{tr['cases']} fixture cases "
               f"(manifest {esc(r.manifest.benchmark or '?')})" if tr else "")
        rule_html = (f'<dt>Rule</dt><dd><code>{esc(d.rule_id)}</code> v{esc(ver)}'
                     + (f" · thresholds: {esc(thr)}" if thr else "") + f"{trs}</dd>")
    exp_html = ""
    exps = [e for e in r.experiments if e.decision_id == d.id]
    if exps:
        erows = [[e.family, e.estimator, label(e.metric), fmt(e.mean_diff), fmt_ci(e.ci), fmt(e.p_adjusted), e.outcome]
                 for e in exps]
        exp_html = ("<dt>Experiment</dt><dd>" + table(["family", "estimator", "metric", "mean diff", "95% CI",
                                                       "p (Holm)", "outcome"], erows)
                    + f'<span class="links">development evidence · paired CV, corrected t-test · '
                    f'{_link("fig.exp." + d.id[2:], "figure")}</span></dd>')
    q_html = ""
    if d.question_id:
        q = next((q for q in r.questions if q.id == d.question_id), None)
        if q:
            opts = "".join(f"<li><code>{esc(o.value)}</code> — {esc(o.label)}"
                           + (f' <span class="muted">({esc(o.effect)})</span>' if o.effect else "") + "</li>"
                           for o in q.options)
            ans = (f"answered: <code>{esc(q.answer)}</code>" if q.answer else
                   f'answer with <code>report.answer({{"{esc(q.id[2:])}": "&lt;option&gt;"}})</code>')
            sug = (f'<br><span class="muted">LLM suggestion (not applied): {esc(q.suggestion)} — '
                   f"{esc(q.suggestion_reason or '')}</span>" if q.suggestion else "")
            q_html = f"<dt>Question</dt><dd>{esc(q.text)}<ul>{opts}</ul>{ans}{sug}</dd>"
    figs = " · ".join(_link(f) for f in d.figure_ids)
    fl = " · ".join(_link(f.id, f.id) for f in findings)
    parts = [
        f'<article class="card" id="{aid(d.id)}"><header><h3>{esc(d.title)}</h3>{class_badge(d.cls)}'
        f"{status_badge(d.status)}</header><dl class=\"kv\">",
        f"<dt>Do</dt><dd>{esc(d.action)}</dd><dt>Why</dt><dd>{esc(d.rationale)}</dd>",
        f"<dt>Risk</dt><dd>{esc(d.risk)}</dd>" if d.risk else "",
        f"<dt>Check</dt><dd>{esc(d.verify_hint)}</dd>" if d.verify_hint else "",
        f"<dt>Note</dt><dd>{esc(d.note)}</dd>" if d.note else "",
        f"<dt>Findings</dt><dd>{fl}</dd>" if fl else "",
        rule_html, exp_html, q_html,
        f"<dt>Figures</dt><dd>{figs}</dd>" if figs else "",
        "</dl>", ev_html, "</article>",
    ]
    return "".join(parts)


def _main_figure_ids(r: Any) -> set[str]:
    important = {f.id for f in r.findings if f.severity in (Severity.WARN, Severity.CRITICAL)}
    keep = []
    for f in r.figures:
        linked = [d for d in r.decisions if f.id in d.figure_ids]
        if f.id in ("fig.target",) or f.id.startswith("fig.exp.") or any(set(d.finding_ids) & important for d in linked):
            keep.append(f.id)
    return set(keep[:MAIN_FIGURES])


def report_body(r: Any) -> tuple[str, dict[str, Any]]:
    p = r.profile
    c = summary_counts(r)
    figs = {f.id: f.plot for f in r.figures}
    chips = [f"{p.dataset.n_rows:,} dev rows", f"{p.dataset.n_features} features",
             f"task: {r.task.value}" + (" (inferred)" if p.task_inferred else "")]
    if p.holdout.get("used"):
        chips.append(f"holdout: {p.holdout['n_holdout']:,} rows, {p.holdout['method']}")
    else:
        chips.append("no holdout")
    if p.sampling.get("sampled"):
        chips.append(f"profiled a seeded sample of {p.sampling['n_rows']:,} rows")
    if r.context.groups:
        chips.append(f"groups: {r.context.groups}")
    if r.context.time_col:
        chips.append(f"time: {r.context.time_col}")

    out = [f'<header><h1>metriclens report · {esc(r.target)}</h1>'
           f'<p class="sub">{esc(headline(r))}</p><div class="chips">'
           + "".join(f'<span class="chip">{esc(x)}</span>' for x in chips) + "</div></header>"]

    # executive summary
    out.append('<section id="summary"><h2>Executive summary</h2><div class="tiles">'
               f'<div class="tile"><b>{c["applied"]}</b><span>applied / confirmed / verified</span></div>'
               f'<div class="tile"><b>{c["awaiting"]}</b><span>awaiting your answer</span></div>'
               f'<div class="tile"><b>{c["review"]}</b><span>to review by hand</span></div>'
               f'<div class="tile"><b>{c["conditional"]}</b><span>conditional (testable)</span></div>'
               f'<div class="tile"><b>{c["critical"]}</b><span>critical findings</span></div></div>')
    pr = priorities(r)
    if pr:
        out.append("<h3>Needs your attention</h3><ul>" + "".join(
            f"<li>{status_badge(d.status)} {_link(d.id, d.title)}</li>" for d in pr) + "</ul>")
    out.append("</section>")

    # roadmap
    dmap = {d.id: d for d in r.decisions}
    out.append('<section id="roadmap"><h2>Roadmap</h2><p class="sub">Ordered steps from split to final '
               "evaluation. Only applied, confirmed or verified steps enter the exported pipeline.</p>")
    for st in r.roadmap:
        items = "".join(f"<li>{class_badge(dmap[i].cls)}{status_badge(dmap[i].status)} {_link(i, dmap[i].title)}</li>"
                        for i in st.decision_ids)
        out.append(f'<div class="stage"><h3>{esc(st.title)} <span class="muted">· {esc(st.summary)}</span></h3>'
                   f"<ul>{items}</ul></div>")
    out.append("</section>")

    # questions
    if r.pending_questions:
        snippet = ", ".join(f'"{q.id[2:]}": "<option>"' for q in r.pending_questions[:3])
        out.append('<section id="questions"><h2>Questions for you</h2><p class="sub">Contextual decisions stay out '
                   "of the pipeline until you answer. Answers re-derive the decisions from the stored profile.</p><ul>"
                   + "".join(f"<li>{_link(q.decision_ids[0] if q.decision_ids else q.id, q.text)} "
                             f'<span class="muted">({", ".join(esc(o.value) for o in q.options)})</span></li>'
                             for q in r.pending_questions)
                   + f"</ul><pre>report = report.answer({{{esc(snippet)}}})</pre></section>")

    # metrics
    mp = r.metric_plan
    out.append('<section id="metrics"><h2>Metrics</h2><dl class="kv">'
               f"<dt>Optimize</dt><dd>{esc(', '.join(label(m) for m in mp.primary))}"
               + (' <span class="muted">(provisional until you answer)</span>' if mp.dev_metric_provisional else "")
               + "</dd>"
               + (f"<dt>Alongside</dt><dd>{esc(', '.join(label(m) for m in mp.report_alongside))}</dd>"
                  if mp.report_alongside else "")
               + "".join(f"<dt>Avoid</dt><dd><b>{esc(m)}</b> — {esc(w)}</dd>" for m, w in mp.avoid.items())
               + "".join(f"<dt>Note</dt><dd>{esc(n)}</dd>" for n in mp.notes) + "</dl>")
    b = r.baseline
    if b.get("families"):
        mt = b["metric"]
        rows = [["dummy", "prior / median", fmt(natural(mt, b.get("dummy", {}).get("mean"))),
                 fmt(b.get("dummy", {}).get("std"))]]
        rows += [[k, v["estimator"], fmt(natural(mt, v["mean"])), fmt(v["std"])] for k, v in b["families"].items()]
        out.append(f"<h3>Baseline ({esc(label(b['metric']))}, {b['n_folds']} folds on {b['n_rows']:,} dev rows)</h3>"
                   + table(["model", "estimator", "mean", "std"], rows))
    out.append("</section>")

    # decisions
    out.append('<section id="decisions"><h2>Decisions</h2><p class="sub">Each card: what to do, why, the evidence '
               "(value, interval, method), the rule version and thresholds, and any experiment.</p>")
    for st in r.roadmap:
        out.append(f"<h3 style=\"margin-top:20px\">{esc(st.title)}</h3>")
        out += [_decision_card(r, dmap[i]) for i in st.decision_ids]
    out.append("</section>")

    # figures
    main = _main_figure_ids(r)
    out.append('<section id="figures"><h2>Visual diagnostics</h2><p class="sub">Selected by applicability and '
               "importance; each figure says what it shows and which evidence it supports.</p>")
    out += [_figure(f) for f in r.figures if f.id in main]
    rest = [f for f in r.figures if f.id not in main]
    if rest:
        out.append(f"<details><summary>All diagnostics ({len(rest)} more)</summary>"
                   + "".join(_figure(f) for f in rest) + "</details>")
    out.append("</section>")

    # findings & evidence
    frows = []
    for f in r.findings:
        frows.append([sev_badge(f.severity, f.trigger.value == "uncertain"),
                      f'<span id="{aid(f.id)}">{esc(f.message)}</span>',
                      ", ".join(_link(e) for e in f.evidence_ids),
                      f"<code>{esc(f.rule_id)}</code> v{esc(f.rule_version)}",
                      esc("; ".join(f"{k}={fmt(v)}" for k, v in f.thresholds.items()))])
    out.append('<section id="findings"><h2>Findings</h2>'
               + (table(["severity", "finding", "evidence", "rule", "thresholds"], frows, raw=True) if frows
                  else "<p>No rule fired.</p>") + "</section>")
    erows = [[f'<span id="{aid(e.id)}"><code>{esc(e.id)}</code></span>', esc(e.label), esc(fmt(e.value)),
              esc(fmt_ci(e.ci)), esc(e.method), ", ".join(_link(x) for x in e.figure_ids)]
             for e in r.evidence.values()]
    out.append('<section id="evidence"><h2>Evidence</h2>'
               + table(["id", "measure", "value", "95% interval", "method", "figures"], erows, raw=True)
               + "</section>")

    # profile
    ds = p.dataset
    out.append('<section id="profile"><h2>Dataset profile</h2>' + table(["measure", "value"], [
        ["rows passed in", ds.n_rows_input], ["development rows profiled", ds.n_rows], ["features", ds.n_features],
        ["column kinds", ds.kind_counts], ["exact duplicate rows", f"{ds.duplicate_rows} ({ds.duplicate_fraction:.2%})"],
        ["identical features, different target", ds.conflicting_duplicate_groups],
        ["target missing", f"{p.target.n_missing} ({p.target.missing_fraction:.1%})"],
        ["task", f"{p.task.value} — {p.target.task_evidence.get('reason', '')}"]]))
    crow = [[esc(cp.name), esc(cp.kind.value), esc(cp.kind_reason), esc(cp.dtype), esc(f"{cp.missing_fraction:.1%}"),
             esc(cp.cardinality), esc(fmt(cp.stats.get("skew"))), esc(fmt(cp.association))]
            for cp in p.columns.values()]
    out.append(f"<details><summary>All columns ({len(crow)})</summary>"
               + table(["column", "kind", "why this kind", "dtype", "missing", "distinct", "skew", "single-feature score"],
                       crow, raw=True) + "</details></section>")

    # llm
    if r.llm:
        L = r.llm
        dis = L.disclosure
        out.append('<section id="llm"><h2>Optional interpretation · LLM</h2><div class="llm">'
                   f'<p class="sub">Written by {esc(L.provider)} / {esc(L.model)} from the computed report only. '
                   "It cannot change any decision; every claim below cites existing evidence or decision ids. "
                   f"{L.dropped_claims} claim(s) were dropped for citing unknown ids.</p>"
                   f"<p>{esc(L.summary)}</p>")
        if L.priorities:
            out.append("<ul>" + "".join(f"<li>{esc(cl.text)} <span class=\"links\">"
                                        + ", ".join(_link(x) for x in cl.evidence_ids + cl.decision_ids)
                                        + "</span></li>" for cl in L.priorities) + "</ul>")
        if L.semantic_flags:
            out.append("<h3>Semantic flags · manual review required</h3><ul>" + "".join(
                f"<li><b>{esc(s.column)}</b>: {esc(s.concern)} — {esc(s.reason)}</li>" for s in L.semantic_flags)
                + "</ul>")
        if L.questions_for_user:
            out.append("<h3>Questions the LLM suggests asking</h3><ul>"
                       + "".join(f"<li>{esc(q)}</li>" for q in L.questions_for_user) + "</ul>")
        out.append("<details><summary>What was sent</summary>"
                   + table(["field", "value"], [[k, v] for k, v in dis.items()]) + "</details></div></section>")

    out.append('<section id="limitations"><h2>Limitations</h2><ul>'
               + "".join(f"<li>{esc(x)}</li>" for x in r.limitations) + "</ul></section>")
    m = r.manifest
    out.append('<section id="manifest"><h2>Reproducibility</h2><details><summary>Manifest</summary>'
               + table(["field", "value"], [["metriclens", m.metriclens_version], ["python", m.python],
                                            ["packages", m.packages], ["seed", m.seed],
                                            ["rule versions", m.rule_versions], ["profile options", m.profile_options],
                                            ["budget", m.budget], ["benchmark manifest", m.benchmark]])
               + "</details></section>")
    return "".join(out), figs


NAV = [("summary", "Summary"), ("roadmap", "Roadmap"), ("questions", "Questions"), ("metrics", "Metrics"),
       ("decisions", "Decisions"), ("figures", "Figures"), ("findings", "Findings"), ("evidence", "Evidence"),
       ("profile", "Profile"), ("review", "Review"), ("llm", "LLM"), ("limitations", "Limitations")]


def shell(title: str, body: str, figures: dict[str, Any], plotlyjs: str, present: set[str]) -> str:
    if plotlyjs == "cdn":
        script = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    elif plotlyjs == "inline":
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as exc:
            raise ImportError("plotlyjs='inline' needs the plotly package: pip install 'metriclens[report]'") from exc
        script = f"<script>{get_plotlyjs()}</script>"
    elif plotlyjs == "none":
        script = ""
    else:
        raise ValueError("plotlyjs must be 'cdn', 'inline' or 'none'.")
    nav = "".join(f'<a href="#{k}">{v}</a>' for k, v in NAV if k in present)
    data = json.dumps(figures, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style>{script}</head><body><main>"
            f"<nav aria-label=\"sections\">{nav}</nav>{body}</main>"
            f'<script type="application/json" id="ml-figures">{data}</script><script>{JS}</script></body></html>')


def _present(body: str) -> set[str]:
    return {k for k, _ in NAV if f'id="{k}"' in body}


def render_report_html(r: Any, review: Any = None, plotlyjs: str = "cdn") -> str:
    body, figs = report_body(r)
    if review is not None:
        rb, rf = review.html_section()
        idx = body.index('<section id="limitations">')
        body = body[:idx] + rb + body[idx:]
        figs.update(rf)
    return shell(f"metriclens · {r.target}", body, figs, plotlyjs, _present(body))
