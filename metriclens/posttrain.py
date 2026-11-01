"""Second stage: after training, check the roadmap's assumptions against the
model's actual errors.

Deterministic, numpy-only, and it sees only y_true / y_pred (/ y_score and
optional segment labels). Judgements use explicit reference points rather than
bare thresholds — e.g. "tails dominate" is measured against what Gaussian
errors would give — and the reference is shown with every check. Only scores
computed on the untouched holdout are an unbiased estimate; the review says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .enums import DecisionStatus
from .figures import _hist_with_box, _layout
from .profiling.stats import r
from .report.model import FigureSpec

# Gaussian-error references: RMSE/MAE = sqrt(pi/2); worst 5% of |N(0,1)| carry ~14.7% of total |error|
GAUSS_RMSE_MAE = float(np.sqrt(np.pi / 2))
GAUSS_WORST5 = 0.147
UNVERIFIED = {DecisionStatus.AWAITING_USER, DecisionStatus.PROPOSED, DecisionStatus.UNCERTAIN,
              DecisionStatus.INCONCLUSIVE, DecisionStatus.SKIPPED_BUDGET, DecisionStatus.SUPPORTED_BY_PROXY,
              DecisionStatus.NEEDS_REVIEW}


class ReviewCheck(BaseModel):
    finding_id: str | None = None
    hypothesis: str
    held: bool | None = None          # True held, False did not, None = observation
    evidence: dict[str, Any] = Field(default_factory=dict)
    reference: str = ""
    comment: str = ""


class ReviewReport(BaseModel):
    task: str
    n: int
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)
    per_class: list[dict[str, Any]] = Field(default_factory=list)
    confusion: dict[str, Any] = Field(default_factory=dict)
    thresholds: list[dict[str, float]] = Field(default_factory=list)
    segments: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[ReviewCheck] = Field(default_factory=list)
    unverified: list[dict[str, str]] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        verdict = {True: "HELD", False: "DID NOT HOLD", None: "OBSERVED"}
        lines = ["REVIEW — did the roadmap hold?", f"Task: {self.task} · n={self.n}", ""]
        if self.metrics:
            lines.append("Measured:")
            lines += [f"- {k}: {v:.4g}" for k, v in self.metrics.items()]
        if self.baseline:
            lines += [f"- baseline {k}: {v:.4g}" for k, v in self.baseline.items()]
        lines += ["", "Checks:"]
        for c in self.checks:
            lines.append(f"- [{verdict[c.held]}] {c.hypothesis}")
            if c.comment:
                lines.append(f"    {c.comment}")
            if c.reference:
                lines.append(f"    reference: {c.reference}")
        if self.segments:
            lines += ["", "Worst segments:"]
            lines += [f"- {s['segment']}={s['value']} (n={s['n']}): {s['metric']}={s['score']:.4g}"
                      for s in self.segments[:5]]
        if self.unverified:
            lines += ["", "Roadmap assumptions that were never verified:"]
            lines += [f"- [{u['status']}] {u['title']} ({u['id']})" for u in self.unverified]
        if self.notes:
            lines += [""] + [f"note: {n}" for n in self.notes]
        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.to_text()

    def html_section(self) -> tuple[str, dict[str, Any]]:
        from .render.html import _figure, esc, table

        verdict = {True: ("held", "good"), False: ("did not hold", "critical"), None: ("observed", "neutral")}
        from .render.html import _badge

        rows = [[_badge(*verdict[c.held]), esc(c.hypothesis), esc(c.comment), esc(c.reference)] for c in self.checks]
        parts = ['<section id="review"><h2>Review · after training</h2>',
                 f'<p class="sub">{esc(self.task)} · n={self.n}. ' + " ".join(esc(n) for n in self.notes) + "</p>",
                 table(["metric", "value"], [[k, v] for k, v in {**self.metrics, **{f"baseline {k}": v for k, v in
                                                                                   self.baseline.items()}}.items()]),
                 "<h3>Did the roadmap's hypotheses hold?</h3>",
                 table(["verdict", "hypothesis", "result", "reference"], rows, raw=True)]
        if self.per_class:
            parts.append("<h3>Per class</h3>" + table(list(self.per_class[0]), [list(x.values()) for x in self.per_class]))
        if self.thresholds:
            parts.append("<h3>Threshold trade-off</h3>" + table(list(self.thresholds[0]),
                                                                [list(x.values()) for x in self.thresholds]))
        if self.segments:
            parts.append("<h3>Worst segments</h3>" + table(list(self.segments[0]),
                                                           [list(x.values()) for x in self.segments]))
        if self.unverified:
            parts.append("<h3>Assumptions that were never verified</h3><ul>" + "".join(
                f'<li><a href="#{esc(u["id"])}">{esc(u["title"])}</a> — {esc(u["status"])}</li>'
                for u in self.unverified) + "</ul>")
        parts += [_figure(f) for f in self.figures]
        parts.append("</section>")
        return "".join(parts), {f.id: f.plot for f in self.figures}

    def to_html(self, path: str | None = None, plotlyjs: str = "cdn") -> str:
        from .render.html import shell

        body, figs = self.html_section()
        out = shell("metriclens review", f"<header><h1>metriclens review</h1></header>{body}", figs, plotlyjs,
                    {"review"})
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(out)
        return out


def _to_1d(a: Any) -> np.ndarray:
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    return np.asarray(a).ravel()


def _py(x: Any) -> Any:
    return x.item() if isinstance(x, np.generic) else x


def _unverified(report: Any) -> list[dict[str, str]]:
    if report is None:
        return []
    out = [{"id": d.id, "title": d.title, "status": d.status.value} for d in report.decisions
           if d.status in UNVERIFIED]
    if report.metric_plan.dev_metric_provisional and not any(u["id"] == "d.metric" for u in out):
        out.append({"id": "d.metric", "title": "Metric priority (provisional)", "status": "awaiting_user"})
    return out


def _segments(seg: Any, n: int, err: np.ndarray, metric: str, higher_worse: bool) -> list[dict[str, Any]]:
    if seg is None:
        return []
    df = seg if isinstance(seg, pd.DataFrame) else pd.DataFrame({getattr(seg, "name", None) or "segment": _to_1d(seg)})
    if len(df) != n:
        raise ValueError(f"segments has {len(df)} rows; y_true has {n}.")
    rows = []
    for col in df.columns:
        s = df[col].reset_index(drop=True).astype(str)
        for val, idx in s.groupby(s).indices.items():
            if len(idx) < 20:
                continue
            rows.append({"segment": str(col), "value": val, "n": int(len(idx)), "metric": metric,
                         "score": r(float(err[idx].mean()), 5)})
    return sorted(rows, key=lambda x: -x["score"] if higher_worse else x["score"])[:10]


def _review_regression(yt: np.ndarray, yp: np.ndarray, fids: set[str], segments: Any) -> ReviewReport:
    resid = yp - yt
    ae = np.abs(resid)
    n = int(yt.size)
    mae, rmse = float(ae.mean()), float(np.sqrt((resid ** 2).mean()))
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    ratio = rmse / mae if mae > 0 else float("nan")
    k = max(1, int(np.ceil(0.05 * n)))
    share = float(np.sort(ae)[::-1][:k].sum() / ae.sum()) if ae.sum() > 0 else 0.0
    base_mae = float(np.abs(yt - np.median(yt)).mean())
    rep = ReviewReport(task="regression", n=n,
                       metrics={"MAE": r(mae), "RMSE": r(rmse), "median_abs_error": r(float(np.median(ae))),
                                "R2": r(r2), "RMSE_MAE_ratio": r(ratio), "worst5pct_error_share": r(share)},
                       baseline={"MAE (predict the median)": r(base_mae)})
    tails = ratio >= 1.2 * GAUSS_RMSE_MAE or share >= 2 * GAUSS_WORST5
    ev = {"RMSE_MAE_ratio": r(ratio, 3), "worst5pct_error_share": r(share, 3)}
    ref = (f"Gaussian errors give RMSE/MAE = {GAUSS_RMSE_MAE:.2f} and worst-5% share = {GAUSS_WORST5:.2f}; "
           "'tails dominate' = ratio >= 1.2x or share >= 2x those.")
    skew = [f for f in ("f.target_skew", "f.heavy_tailed_target") if f in fids]
    if skew:
        rep.checks.append(ReviewCheck(
            finding_id=skew[0], hypothesis="A few large errors dominate squared-error metrics (skewed/heavy-tailed target).",
            held=tails, evidence=ev, reference=ref,
            comment=(f"Held: RMSE/MAE={ratio:.2f}, the worst 5% carry {share:.0%} of total error. Keep MAE primary "
                     "unless large errors are what you care about." if tails else
                     "Did not hold: errors are spread fairly evenly, so RMSE is less misleading than the target "
                     "shape suggested.")))
    else:
        rep.checks.append(ReviewCheck(hypothesis="Error-distribution shape (no skew warning was raised).", held=None,
                                      evidence=ev, reference=ref,
                                      comment=(f"RMSE/MAE={ratio:.2f}; worst 5% carry {share:.0%}. "
                                               + ("Tails dominate more than the profile suggested."
                                                  if tails else "Errors look well-behaved."))))
    rep.checks.append(ReviewCheck(
        hypothesis="The model beats a constant (median) prediction.", held=mae < base_mae,
        evidence={"MAE": r(mae), "baseline_MAE": r(base_mae)}, reference="MAE of always predicting the median of y_true",
        comment=f"MAE {mae:.4g} vs {base_mae:.4g} ({(1 - mae / base_mae):+.1%})." if base_mae else ""))
    rep.segments = _segments(segments, n, ae, "MAE", True)

    rep.figures.append(FigureSpec(
        id="fig.review.residuals", title="Residuals (prediction − truth)", priority=60,
        what="Histogram of residuals with a quantile box.", why="Shows bias (off-centre) and tail weight of the errors.",
        plot=_hist_with_box(resid, "residual"),
        table={"columns": ["quantile", "residual"],
               "rows": [[q, r(float(np.quantile(resid, q)))] for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]}))
    srt = np.sort(ae)[::-1]
    cum = np.cumsum(srt) / max(srt.sum(), 1e-12)
    xs = np.linspace(0, 1, 51)
    ys = [0.0] + [float(cum[max(0, int(np.ceil(x * n)) - 1)]) for x in xs[1:]]
    lay = _layout("share of rows (worst first)", "share of total absolute error")
    lay["xaxis"]["tickformat"] = ".0%"
    lay["yaxis"]["tickformat"] = ".0%"
    lay["showlegend"] = True
    lay["legend"] = {"orientation": "h", "y": -0.25}
    rep.figures.append(FigureSpec(
        id="fig.review.concentration", title="How concentrated the error is", priority=61,
        what="Cumulative share of total absolute error carried by the worst x% of rows; dashed: equal errors.",
        why="A steep start means a few rows drive the error metrics.",
        plot={"data": [{"type": "scatter", "mode": "lines", "x": [r(x) for x in xs], "y": [r(y) for y in ys],
                        "line": {"color": "var:series-1", "width": 2}, "name": "model"},
                       {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1],
                        "line": {"color": "var:text-muted", "width": 1.5, "dash": "dash"}, "name": "equal errors"}],
              "layout": lay},
        table={"columns": ["worst share of rows", "share of error"],
               "rows": [[f"{x:.0%}", f"{y:.1%}"] for x, y in zip(xs[::5], ys[::5])]}))
    if rep.segments:
        segs = rep.segments[::-1]
        rep.figures.append(FigureSpec(
            id="fig.review.segments", title="MAE by segment (worst 10)", priority=62,
            what="Mean absolute error per segment value with at least 20 rows.",
            why="Average error can hide segments where the model fails.",
            plot={"data": [{"type": "bar", "orientation": "h", "x": [s["score"] for s in segs],
                            "y": [f"{s['segment']}={s['value']}" for s in segs], "marker": {"color": "var:series-1"},
                            "hovertemplate": "%{y}: %{x:.4g}<extra></extra>"}],
                  "layout": _layout("MAE", "", height=max(220, 26 * len(segs) + 70),
                                    margin={"l": 160, "r": 16, "t": 16, "b": 48})},
            table={"columns": ["segment", "value", "n", "MAE"],
                   "rows": [[s["segment"], s["value"], s["n"], s["score"]] for s in rep.segments]}))
    return rep


def _binary_curves(y: np.ndarray, s: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    n_pos = max(int(y.sum()), 1)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    ap = float((precision * ys).sum() / n_pos)
    ranks = pd.Series(s).rank().to_numpy()
    n_neg = len(y) - n_pos
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_neg > 0 else float("nan")
    return ap, auc, precision, recall, ss


def _review_classification(yt: np.ndarray, yp: np.ndarray, fids: set[str], y_score: Any,
                           pos_label: Any, segments: Any) -> ReviewReport:
    n = int(yt.size)
    labels = list(pd.unique(np.concatenate([yt, yp])))
    try:
        labels = sorted(labels)
    except TypeError:
        labels = sorted(labels, key=str)
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(yt, yp):
        cm[idx[a], idx[b]] += 1
    support = cm.sum(1)
    pred_n = cm.sum(0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels)), where=support > 0)
    precision = np.divide(np.diag(cm), pred_n, out=np.zeros(len(labels)), where=pred_n > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=(precision + recall) > 0)
    present = support > 0
    n_present = int(present.sum())
    acc = float(np.trace(cm) / n)
    bacc = float(recall[present].mean())
    maj = float(support.max() / n)
    minority = labels[int(np.argmin(np.where(present, support, np.iinfo(int).max)))]
    rep = ReviewReport(task="classification", n=n,
                       metrics={"accuracy": r(acc), "balanced_accuracy": r(bacc), "macro_f1": r(float(f1[present].mean())),
                                "minority_recall": r(float(recall[idx[minority]]))},
                       baseline={"accuracy (always predict the majority class)": r(maj)})
    rep.per_class = [{"class": str(_py(lab)), "support": int(support[i]), "precision": r(float(precision[i]), 4),
                      "recall": r(float(recall[i]), 4), "f1": r(float(f1[i]), 4)} for i, lab in enumerate(labels)]
    rep.confusion = {"labels": [str(_py(x)) for x in labels], "matrix": cm.tolist()}

    gap = acc - bacc
    if "f.class_imbalance" in fids:
        held = gap >= 0.10
        rep.checks.append(ReviewCheck(
            finding_id="f.class_imbalance", hypothesis="Accuracy hides poor performance on the rare class(es).",
            held=held, evidence={"accuracy": r(acc, 3), "balanced_accuracy": r(bacc, 3),
                                 "minority_recall": r(float(recall[idx[minority]]), 3)},
            reference="held when accuracy exceeds balanced accuracy by >= 0.10",
            comment=(f"Held: accuracy {acc:.1%} vs balanced accuracy {bacc:.1%}; minority class "
                     f"'{_py(minority)}' recall {recall[idx[minority]]:.1%}." if held else
                     f"Did not hold: accuracy {acc:.1%} and balanced accuracy {bacc:.1%} are close.")))
    rep.checks.append(ReviewCheck(
        hypothesis="The model beats always predicting the majority class.", held=bacc > 1 / max(n_present, 1) + 0.02
        and acc >= maj, evidence={"accuracy": r(acc, 3), "majority_baseline": r(maj, 3), "balanced_accuracy": r(bacc, 3)},
        reference=f"majority baseline accuracy {maj:.1%}; balanced accuracy of chance = {1 / max(n_present, 1):.2f}",
        comment=f"accuracy {acc:.1%} vs baseline {maj:.1%}."))

    if y_score is not None and len(labels) == 2:
        s = np.asarray(y_score, dtype=float)
        if s.ndim == 2:
            s = s[:, -1]
        pl = pos_label if pos_label in idx else (minority if pos_label is None else labels[-1])
        yb = (yt == pl).astype(int)
        ap, auc, prec, rec, ss = _binary_curves(yb, s.ravel())
        rep.metrics.update({"average_precision": r(ap), "roc_auc": r(auc)})
        rep.baseline["average_precision (random)"] = r(float(yb.mean()))
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pred = s >= t
            tp = int((pred & (yb == 1)).sum())
            p_ = tp / max(int(pred.sum()), 1)
            r_ = tp / max(int(yb.sum()), 1)
            rep.thresholds.append({"threshold": t, "precision": r(p_, 4), "recall": r(r_, 4),
                                   "f1": r(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, 4)})
        rep.notes.append(f"Scores treated as P(class = '{_py(pl)}'). Choose a threshold on validation data, "
                         "not on the holdout these predictions came from, or the holdout estimate becomes optimistic.")
        step = max(1, len(prec) // 200)
        lay = _layout("recall", "precision")
        lay["shapes"] = [{"type": "line", "x0": 0, "x1": 1, "y0": float(yb.mean()), "y1": float(yb.mean()),
                          "line": {"color": "var:text-muted", "dash": "dash", "width": 1.5}}]
        rep.figures.append(FigureSpec(
            id="fig.review.pr", title="Precision–recall curve", priority=61,
            what="Precision vs recall over all thresholds; dashed: a random model (= prevalence).",
            why="The trade-off you choose a threshold from; PR-AUC is the area under it.",
            plot={"data": [{"type": "scatter", "mode": "lines", "x": [r(x, 4) for x in rec[::step]],
                            "y": [r(x, 4) for x in prec[::step]], "line": {"color": "var:series-1", "width": 2},
                            "hovertemplate": "recall %{x:.2f}, precision %{y:.2f}<extra></extra>"}], "layout": lay},
            table={"columns": list(rep.thresholds[0]), "rows": [list(x.values()) for x in rep.thresholds]}))

    norm = np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float), where=support[:, None] > 0)
    lab_s = [str(_py(x)) for x in labels]
    rep.figures.insert(0, FigureSpec(
        id="fig.review.confusion", title="Confusion matrix (row-normalized)", priority=60,
        what="Share of each true class (rows) predicted as each class (columns).",
        why="Shows which classes are confused, independent of how many rows each has.",
        plot={"data": [{"type": "heatmap", "z": [[r(x, 4) for x in row] for row in norm], "x": lab_s, "y": lab_s,
                        "zmin": 0, "zmax": 1, "colorscale": "scale:sequential",
                        "hovertemplate": "true %{y} → predicted %{x}: %{z:.1%}<extra></extra>"}],
              "layout": _layout("predicted", "true", height=max(260, 40 * len(labels) + 120),
                                yaxis={"autorange": "reversed", "title": {"text": "true"}})},
        table={"columns": ["true \\ predicted", *lab_s], "rows": [[lab, *row] for lab, row in zip(lab_s, cm.tolist())]}))
    err = (yt != yp).astype(float)
    rep.segments = _segments(segments, n, err, "error rate", True)
    return rep


def review(y_true: Any, y_pred: Any, report: Any = None, *, y_score: Any = None, segments: Any = None,
           task: Any = None) -> ReviewReport:
    """Check the report's assumptions against actual predictions.

    y_true, y_pred : truth and predictions (labels or values), same length.
    report : the Report from `analyze`; supplies the task, findings and the
        list of assumptions that were never verified.
    y_score : optional scores/probabilities (binary: P(positive) or an (n, 2) array).
    segments : optional Series/DataFrame of segment labels aligned to y_true.
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if yt.size == 0:
        raise ValueError("y_true / y_pred are empty.")
    if task is not None:
        t = getattr(task, "value", task)
    elif report is not None:
        t = report.task.value
    else:
        t = "regression" if (np.issubdtype(yt.dtype, np.floating) and len(np.unique(yt)) > 20) else "classification"
    fids = {f.id for f in report.findings} if report is not None else set()
    if t == "regression":
        yt_f, yp_f = yt.astype(float), yp.astype(float)
        keep = ~(np.isnan(yt_f) | np.isnan(yp_f))
        seg = segments
        if seg is not None and not keep.all():
            seg = (seg if isinstance(seg, pd.DataFrame) else pd.Series(_to_1d(seg))).reset_index(drop=True)[keep]
        rep = _review_regression(yt_f[keep], yp_f[keep], fids, seg)
    elif t == "classification":
        pos = None
        if report is not None and report.metric_plan.positive_label is not None:
            from .export import positive_label

            pos = positive_label(report.profile, report.metric_plan.positive_label)
        rep = _review_classification(yt, yp, fids, y_score, pos, segments)
    else:
        raise ValueError(f"review() supports 'regression' and 'classification'; got {t!r}.")
    rep.unverified = _unverified(report)
    rep.notes.insert(0, "Only predictions on the untouched holdout give an unbiased estimate.")
    return rep
