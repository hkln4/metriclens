"""Decision-bound figures.

Figures are chosen by applicability and priority — never "every chart for every
column". Each one states what it shows, why it was selected, and which evidence /
decisions it supports. Only aggregates are stored (bin counts, quantiles,
correlations), never raw rows, so the report stays shareable.

Plots are plotly JSON. Colors are *tokens* ("var:series-1", "scale:diverging")
that the HTML renderer resolves from CSS custom properties, so light and dark
mode use their own validated steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .enums import ColumnKind, Task
from .frozen import evolve
from .profiling.profile import DataProfile
from .profiling.stats import r
from .report.model import Decision, Evidence, FigureSpec

MAX_COLUMN_FIGURES = 6


def _layout(x_title: str = "", y_title: str = "", **extra: Any) -> dict[str, Any]:
    lay = {
        "xaxis": {"title": {"text": x_title}, "gridcolor": "var:grid", "zeroline": False},
        "yaxis": {"title": {"text": y_title}, "gridcolor": "var:grid", "zeroline": False},
        "margin": {"l": 56, "r": 16, "t": 16, "b": 48},
        "height": 300, "bargap": 0.08, "showlegend": False,
    }
    lay.update(extra)
    return lay


def _threshold_line(value: float, axis: str = "x", text: str = "") -> dict[str, Any]:
    if axis == "x":
        shape = {"type": "line", "x0": value, "x1": value, "y0": 0, "y1": 1, "yref": "paper"}
    else:
        shape = {"type": "line", "y0": value, "y1": value, "x0": 0, "x1": 1, "xref": "paper"}
    shape["line"] = {"color": "var:text-muted", "width": 1.5, "dash": "dash"}
    return shape


def _hist(v: np.ndarray, bins: int = 40) -> tuple[list[float], list[int], float]:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [], [], 1.0
    counts, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return [r(c) for c in centers], [int(c) for c in counts], float(edges[1] - edges[0])


def _box_summary(v: np.ndarray) -> dict[str, float]:
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo = float(v[v >= q1 - 1.5 * iqr].min())
    hi = float(v[v <= q3 + 1.5 * iqr].max())
    return {"q1": r(q1), "median": r(med), "q3": r(q3), "lowerfence": r(lo), "upperfence": r(hi)}


def _hist_with_box(v: np.ndarray, x_title: str) -> dict[str, Any]:
    centers, counts, width = _hist(v)
    box = _box_summary(v[np.isfinite(v)])
    traces = [
        {"type": "bar", "x": centers, "y": counts, "width": width * 0.92, "marker": {"color": "var:series-1"},
         "hovertemplate": "%{x:.4g}: %{y} rows<extra></extra>", "yaxis": "y"},
        {"type": "box", "q1": [box["q1"]], "median": [box["median"]], "q3": [box["q3"]],
         "lowerfence": [box["lowerfence"]], "upperfence": [box["upperfence"]], "y": [""], "orientation": "h",
         "yaxis": "y2", "marker": {"color": "var:series-1"}, "line": {"color": "var:series-1", "width": 1.5},
         "fillcolor": "var:series-1-soft", "hoverinfo": "x"},
    ]
    lay = _layout(x_title, "rows", height=320)
    lay["yaxis"]["domain"] = [0, 0.78]
    lay["yaxis2"] = {"domain": [0.84, 1], "showticklabels": False, "gridcolor": "var:grid"}
    return {"data": traces, "layout": lay}


def build_figures(df: pd.DataFrame, p: DataProfile, evidence: dict[str, Evidence],
                  decisions: list[Decision]) -> list[FigureSpec]:
    figs: list[FigureSpec] = []
    tgt = p.target.name
    y = df[tgt]

    def has(eid: str) -> list[str]:
        return [eid] if eid in evidence else []

    # 1. target distribution -------------------------------------------------
    if p.task is Task.CLASSIFICATION and p.target.stats.get("class_counts"):
        cc = p.target.stats["class_counts"]
        labels, counts = list(cc), list(cc.values())
        maj = p.target.stats.get("majority_fraction", 0)
        figs.append(FigureSpec(
            id="fig.target", title=f"Class distribution · {tgt}", priority=10,
            what="Rows per class on the development data.",
            why=("Imbalance decides the split (stratified) and which metrics mislead; the tallest bar's share is the "
                 f"accuracy of a model that always predicts it ({maj:.1%})."),
            evidence_ids=has("ev.target.imbalance") + has("ev.target.majority"),
            plot={"data": [{"type": "bar", "x": labels, "y": counts, "marker": {"color": "var:series-1"},
                            "hovertemplate": "%{x}: %{y} rows<extra></extra>"}],
                  "layout": _layout("class", "rows")},
            table={"columns": ["class", "rows", "share"],
                   "rows": [[lab, c, f"{c / sum(counts):.1%}"] for lab, c in zip(labels, counts)]}))
    elif p.task is Task.REGRESSION:
        v = y.dropna().astype(float).to_numpy()
        if len(v):
            q = p.target.stats.get("quantiles", {})
            figs.append(FigureSpec(
                id="fig.target", title=f"Target distribution · {tgt}", priority=10,
                what="Histogram of the target with a quantile box (whiskers at 1.5 IQR).",
                why="Skew, tails and zeros in the target decide which error metrics mislead and whether a target "
                    "transform is worth testing.",
                evidence_ids=has("ev.target.skew") + has("ev.target.outliers") + has("ev.target.kurtosis")
                + has("ev.target.zeros"),
                plot=_hist_with_box(v, tgt),
                table={"columns": ["quantile", "value"], "rows": [[k, val] for k, val in q.items()]}))
            if "d.log_target" in {d.id for d in decisions} and v.min() >= 0:
                lv = np.log1p(v)
                figs.append(FigureSpec(
                    id="fig.target_log", title=f"log(1 + {tgt})", priority=12,
                    what="The same target on the log scale the transform decision would model.",
                    why="Shows how much the transform reduces the skew before any experiment is run.",
                    evidence_ids=has("ev.target.skew"), decision_ids=["d.log_target"],
                    plot=_hist_with_box(lv, f"log1p({tgt})"),
                    table={"columns": ["statistic", "raw", "log1p"],
                           "rows": [["skew", r(pd.Series(v).skew(), 3), r(pd.Series(lv).skew(), 3)]]}))

    # 2. single-feature association -----------------------------------------
    assoc = sorted(((c, cp.association) for c, cp in p.columns.items() if cp.association is not None),
                   key=lambda t: -t[1])[:20]
    if assoc:
        names = [c for c, _ in assoc][::-1]
        vals = [a for _, a in assoc][::-1]
        lay = _layout("2-fold CV single-feature score (0 = no signal, 1 = predicts the target alone)", "",
                      height=max(220, 26 * len(names) + 70), margin={"l": 140, "r": 16, "t": 16, "b": 48})
        lay["shapes"] = [_threshold_line(0.95)]
        lay["xaxis"]["range"] = [0, 1.02]
        figs.append(FigureSpec(
            id="fig.association", title="How well each feature predicts the target on its own", priority=15,
            what="2-fold cross-validated single-feature score per feature (top 20). Dashed line: leakage-hint threshold 0.95.",
            why="A feature near 1.0 alone is the classic signature of target leakage; the ranking also shows "
                "where signal is.",
            evidence_ids=[e for e in evidence if e.endswith(".association")],
            plot={"data": [{"type": "bar", "orientation": "h", "x": vals, "y": names,
                            "marker": {"color": ["var:critical" if a >= 0.95 else "var:series-1" for a in vals]},
                            "hovertemplate": "%{y}: %{x:.3f}<extra></extra>"}], "layout": lay},
            table={"columns": ["feature", "score"], "rows": [[c, a] for c, a in assoc]}))

    # 3. missingness ---------------------------------------------------------
    miss = p.dataset.missing_columns[:30]
    if miss:
        fr = [p.columns[c].missing_fraction for c in miss][::-1]
        lay = _layout("missing fraction", "", height=max(220, 24 * len(miss) + 70),
                      margin={"l": 140, "r": 16, "t": 16, "b": 48})
        lay["shapes"] = [_threshold_line(0.2), _threshold_line(0.5)]
        lay["xaxis"]["range"] = [0, 1]
        lay["xaxis"]["tickformat"] = ".0%"
        figs.append(FigureSpec(
            id="fig.missingness", title="Missing values per column", priority=20,
            what="Share of missing values per column (columns with any). Dashed lines: 20% review and 50% "
                 "drop-candidate thresholds.",
            why="Missingness decides imputation (fit inside folds) and which columns are candidates to drop.",
            evidence_ids=[e for e in evidence if e.endswith(".missing") and e.startswith("ev.col.")],
            plot={"data": [{"type": "bar", "orientation": "h", "x": fr, "y": miss[::-1],
                            "marker": {"color": "var:series-1"}, "hovertemplate": "%{y}: %{x:.1%}<extra></extra>"}],
                  "layout": lay},
            table={"columns": ["column", "missing"], "rows": [[c, f"{p.columns[c].missing_fraction:.1%}"] for c in miss]}))
    cooc = p.dataset.missing_cooccurrence
    if cooc:
        figs.append(FigureSpec(
            id="fig.missing_pattern", title="Which columns go missing together", priority=40,
            what="Jaccard overlap of missing rows between the most-missing columns (1 = always missing together).",
            why="Columns missing together often share a cause (one form, one sensor); that informs whether "
                "missingness is informative.",
            evidence_ids=has("ev.columns.missing"),
            plot={"data": [{"type": "heatmap", "z": cooc["jaccard"], "x": cooc["columns"], "y": cooc["columns"],
                            "zmin": 0, "zmax": 1, "colorscale": "scale:sequential",
                            "hovertemplate": "%{x} & %{y}: %{z:.2f}<extra></extra>"}],
                  "layout": _layout(height=380, margin={"l": 120, "r": 16, "t": 16, "b": 100})},
            table={"columns": ["column", *cooc["columns"]],
                   "rows": [[c, *row] for c, row in zip(cooc["columns"], cooc["jaccard"])]}))

    # 4. per-column distributions for flagged numeric columns ---------------
    flagged: list[tuple[str, str]] = []
    for d in decisions:
        if d.id in ("d.power_transform",) and d.step:
            flagged += [(c, d.id) for c in d.step.params.get("columns", [])]
        if d.id.startswith("d.drop_leak."):
            flagged += [(c, d.id) for c in (d.step.params.get("columns", []) if d.step else [])]
    seen: set[str] = set()
    for c, did in flagged:
        if c in seen or len(seen) >= MAX_COLUMN_FIGURES or p.columns[c].kind is not ColumnKind.NUMERIC:
            continue
        seen.add(c)
        v = df[c].dropna().astype(float).to_numpy()
        if not len(v):
            continue
        eids = [e for e in (f"ev.col.{c}.skew", f"ev.col.{c}.association", f"ev.col.{c}.pearson") if e in evidence]
        figs.append(FigureSpec(
            id=f"fig.col.{c}", title=f"Distribution · {c}", priority=30,
            what=f"Histogram of '{c}' with a quantile box.",
            why=f"'{c}' is named by a decision ({did}); the shape shows what that decision would act on.",
            evidence_ids=eids, decision_ids=[did], plot=_hist_with_box(v, c),
            table={"columns": ["quantile", "value"],
                   "rows": [[k, val] for k, val in p.columns[c].stats.get("quantiles", {}).items()]}))

    # 5. correlation ---------------------------------------------------------
    rel = p.relations
    if len(rel.corr_columns) >= 3:
        figs.append(FigureSpec(
            id="fig.correlation", title="Rank correlation between numeric features", priority=45,
            what=f"Spearman correlation among the {len(rel.corr_columns)} numeric features most associated with the "
                 "target.",
            why="Near-collinear pairs make coefficients and importances unstable.",
            evidence_ids=has("ev.relations.high_corr"),
            plot={"data": [{"type": "heatmap", "z": rel.corr_matrix, "x": rel.corr_columns, "y": rel.corr_columns,
                            "zmin": -1, "zmax": 1, "colorscale": "scale:diverging",
                            "hovertemplate": "%{x} vs %{y}: %{z:.2f}<extra></extra>"}],
                  "layout": _layout(height=420, margin={"l": 120, "r": 16, "t": 16, "b": 110})},
            table={"columns": ["feature", *rel.corr_columns],
                   "rows": [[c, *row] for c, row in zip(rel.corr_columns, rel.corr_matrix)]}))

    # 6. high-cardinality categoricals (rank-frequency, no labels) ----------
    for c in [c for c, cp in p.columns.items() if cp.kind is ColumnKind.CATEGORICAL and cp.cardinality > 50][:3]:
        vc = df[c].dropna().astype(str).value_counts().to_numpy()
        figs.append(FigureSpec(
            id=f"fig.card.{c}", title=f"Category frequencies · {c}", priority=50,
            what=f"Rows per category of '{c}', sorted by frequency (labels hidden).",
            why="A long tail of rare categories is what one-hot encoding with grouping discards and target encoding "
                "keeps.",
            evidence_ids=has(f"ev.col.{c}.cardinality"), decision_ids=["d.target_encode"],
            plot={"data": [{"type": "scatter", "mode": "lines", "x": list(range(1, len(vc) + 1)),
                            "y": [int(x) for x in vc], "line": {"color": "var:series-1", "width": 2},
                            "hovertemplate": "rank %{x}: %{y} rows<extra></extra>"}],
                  "layout": _layout("category rank", "rows (log scale)",
                                    yaxis={"type": "log", "gridcolor": "var:grid", "title": {"text": "rows (log)"}})},
            table={"columns": ["rank", "rows"], "rows": [[i + 1, int(x)] for i, x in enumerate(vc[:50])]}))

    _link(figs, evidence, decisions)
    return sorted(figs, key=lambda f: (f.priority, f.id))


def _link(figs: list[FigureSpec], evidence: dict[str, Evidence], decisions: list[Decision]) -> None:
    """Back-link figures onto the evidence and decisions they support.

    Report objects are immutable, so linked copies replace the originals inside
    the (mutable, builder-owned) `evidence` dict and `decisions` list. A
    decision's own figures (naming it in decision_ids) come first, so the first
    figure id is the one shown on its card."""
    for eid in list(evidence):
        ids = [f.id for f in figs if eid in f.evidence_ids]
        new = tuple(dict.fromkeys([*evidence[eid].figure_ids, *ids]))
        if new != evidence[eid].figure_ids:
            evidence[eid] = evolve(evidence[eid], figure_ids=new)
    for i, d in enumerate(decisions):
        own = [f.id for f in figs if d.id in f.decision_ids]
        via = [f.id for f in figs if set(f.evidence_ids) & set(d.evidence_ids)]
        new = tuple(dict.fromkeys([*own, *d.figure_ids, *via]))
        if new != d.figure_ids:
            decisions[i] = evolve(d, figure_ids=new)


def experiment_figure(decision: Decision, experiments: list[Any]) -> FigureSpec:
    """Per-fold paired differences for one decision, one row per model family,
    with the practical-effect band shaded."""
    traces, rows = [], []
    for i, e in enumerate(experiments):
        diffs = e.fold_diffs
        traces.append({"type": "scatter", "mode": "markers", "x": [r(x) for x in diffs], "y": [e.family] * len(diffs),
                       "marker": {"color": f"var:series-{i + 1}", "size": 9,
                                  "line": {"color": "var:surface", "width": 2}},
                       "name": e.family, "hovertemplate": e.family + " fold diff: %{x:.4g}<extra></extra>"})
        mean_tr: dict[str, Any] = {
            "type": "scatter", "mode": "markers", "x": [r(e.mean_diff)], "y": [e.family],
            "marker": {"color": "var:text-primary", "size": 12, "symbol": "diamond"},
            "name": f"{e.family} mean", "hovertemplate": "mean diff %{x:.4g}<extra></extra>"}
        if e.ci is not None:
            mean_tr["error_x"] = {"type": "data", "symmetric": False, "array": [r(e.ci[1] - e.mean_diff)],
                                  "arrayminus": [r(e.mean_diff - e.ci[0])], "color": "var:text-primary",
                                  "thickness": 2}
        traces.append(mean_tr)
        ci = f"[{r(e.ci[0])}, {r(e.ci[1])}]" if e.ci is not None else "—"
        rows.append([e.family, e.metric, r(e.mean_diff), ci, r(e.p_adjusted), r(e.min_effect),
                     f"{e.folds_improved}/{len(diffs)}", ", ".join(f"{x:+.4g}" for x in diffs), e.outcome])
    lay = _layout("candidate - current (higher = the change helps)", "",
                  height=max(180, 70 * len(experiments) + 90), margin={"l": 110, "r": 16, "t": 16, "b": 48})
    me = max((e.min_effect for e in experiments), default=0.0)
    lay["shapes"] = [_threshold_line(0.0)]
    if me > 0:
        lay["shapes"].insert(0, {"type": "rect", "x0": -me, "x1": me, "y0": 0, "y1": 1, "yref": "paper",
                                 "fillcolor": "var:series-1-soft", "opacity": 0.5, "line": {"width": 0},
                                 "layer": "below"})
    return FigureSpec(
        id=f"fig.exp.{decision.id[2:]}", title=f"Experiment · {decision.title}", priority=25,
        what="Per-fold score difference (dots), mean with its corrected 95% interval (diamond), and the "
             "practical-effect band (shaded): differences inside it count as negligible. Dashed line: no change.",
        why="Shows whether the conditional decision helps beyond fold-to-fold noise and beyond the practical "
            "threshold, per model family.",
        decision_ids=[decision.id], plot={"data": traces, "layout": lay},
        table={"columns": ["family", "metric", "mean diff", "95% CI", "p (Holm)", "practical threshold",
                           "folds improved", "per-fold differences", "outcome"], "rows": rows})
