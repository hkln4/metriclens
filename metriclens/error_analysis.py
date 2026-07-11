"""Second pass (Step 5): did the init roadmap's hypothesis hold?

`analyze` produces a hypothesis BEFORE training — e.g. "the target is skewed, so a
few large errors will dominate squared-error metrics; report MAE primary". After
training, `review` looks at the ACTUAL errors and checks whether that hypothesis
held. It is the natural closing half of init's promise, not a separate feature.

Like the rest of the library it stays deterministic and above no trust boundary
it shouldn't cross: it sees only y_true / y_pred (the model's outputs), never the
raw training data, and computes every number itself."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field


class ReviewCheck(BaseModel):
    """One hypothesis from the init roadmap, judged against the real errors."""

    # the init Finding this check corresponds to, when it came from one
    finding_id: str | None = None
    hypothesis: str
    # True = held as predicted, False = did not hold, None = observation only
    held: bool | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    comment: str = ""


class ReviewReport(BaseModel):
    """Outcome of the second pass: measured errors + did-it-hold checks."""

    task: str
    n: int
    metrics: dict[str, float] = Field(default_factory=dict)
    checks: list[ReviewCheck] = Field(default_factory=list)

    def to_text(self) -> str:
        verdict = {True: "HELD", False: "DID NOT HOLD", None: "OBSERVED"}
        lines = ["REVIEW — did the roadmap hold?",
                 f"Task: {self.task} · n={self.n}", ""]
        if self.metrics:
            lines.append("Measured metrics:")
            lines += [f"- {k}: {v:.4g}" for k, v in self.metrics.items()]
            lines.append("")
        lines.append("Checks:")
        for c in self.checks:
            lines.append(f"- [{verdict[c.held]}] {c.hypothesis}")
            if c.comment:
                lines.append(f"    {c.comment}")
        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.to_text()


# --- helpers -----------------------------------------------------------------

def _to_1d(a: Any) -> np.ndarray:
    if hasattr(a, "to_numpy"):
        a = a.to_numpy()
    return np.asarray(a).ravel()

def _py(x: Any) -> Any:
    """Cast numpy scalars to plain Python so the report stays JSON-serializable."""
    if isinstance(x, np.generic):
        return x.item()
    return x

def _finding_ids(roadmap) -> set[str]:
    if roadmap is None:
        return set()
    return {f.id for f in roadmap.findings}

def _infer_task_str(y: np.ndarray) -> str:
    if not np.issubdtype(y.dtype, np.number):
        return "classification"
    y = y[~np.isnan(y)]
    if y.size == 0:
        return "classification"
    n_unique = int(np.unique(y).size)
    whole = bool(np.all(np.equal(np.mod(y, 1), 0)))
    if n_unique == 2 or (whole and n_unique <= 20):
        return "classification"
    return "regression"

def _resolve_task(task, roadmap, y: np.ndarray) -> str:
    if task is not None:
        return getattr(task, "value", task)  # Task enum or str
    if roadmap is not None and roadmap.profile_summary.get("task"):
        return roadmap.profile_summary["task"]
    return _infer_task_str(y)


# --- task-specific reviews ---------------------------------------------------

def _review_regression(yt: np.ndarray, yp: np.ndarray, fids: set[str]) -> ReviewReport:
    resid = yp - yt
    abs_err = np.abs(resid)
    n = int(yt.size)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    medae = float(np.median(abs_err))
    ratio = float(rmse / mae) if mae > 0 else float("nan")
    k = max(1, int(np.ceil(0.05 * n)))
    worst = np.sort(abs_err)[::-1][:k]
    total = float(abs_err.sum())
    conc = float(worst.sum() / total) if total > 0 else 0.0
    metrics = {"MAE": mae, "RMSE": rmse, "median_abs_error": medae,
               "RMSE_MAE_ratio": ratio, "worst5pct_error_share": conc}

    tail_dominated = (ratio >= 1.5) or (conc >= 0.30)
    evidence = {"RMSE_MAE_ratio": round(ratio, 3),
                "worst5pct_error_share": round(conc, 3)}
    skew_findings = sorted({"target_skew", "heavy_tailed_target"} & fids)
    if skew_findings:
        checks = [ReviewCheck(
            finding_id=skew_findings[0],
            hypothesis="A few large errors dominate squared-error metrics (skew / heavy tails).",
            held=tail_dominated, evidence=evidence,
            comment=(f"As predicted: RMSE/MAE={ratio:.2f} and the worst 5% of errors carry "
                     f"{conc:.0%} of total absolute error — report MAE primary, RMSE alongside."
                     if tail_dominated else
                     "Unexpected: errors are spread fairly evenly, so the skew warning did not "
                     "translate into tail-dominated errors. RMSE is less misleading than feared."))]
    else:
        checks = [ReviewCheck(
            hypothesis="Error-distribution shape (no skew warning was raised at init).",
            held=None, evidence=evidence,
            comment=(f"RMSE/MAE={ratio:.2f}; worst 5% hold {conc:.0%} of total error. "
                     + ("Tails dominate more than the init profile suggested — revisit the target."
                        if tail_dominated else
                        "Errors look well-behaved; conventional metrics are fine.")))]
    return ReviewReport(task="regression", n=n, metrics=metrics, checks=checks)


def _review_classification(yt: np.ndarray, yp: np.ndarray, fids: set[str]) -> ReviewReport:
    n = int(yt.size)
    classes, counts = np.unique(yt, return_counts=True)
    acc = float((yp == yt).mean())
    recalls = {}
    for c in classes:
        mask = yt == c
        recalls[c] = float((yp[mask] == c).mean()) if mask.sum() else float("nan")
    minority = classes[int(np.argmin(counts))]
    min_recall = recalls[minority]
    macro_recall = float(np.nanmean(list(recalls.values())))
    metrics = {"accuracy": acc, "minority_recall": min_recall, "macro_recall": macro_recall}

    if "class_imbalance" in fids:
        held = acc >= 0.8 and min_recall < 0.5
        checks = [ReviewCheck(
            finding_id="class_imbalance",
            hypothesis="Accuracy looks high while the minority class is missed.",
            held=held,
            evidence={"accuracy": round(acc, 3), "minority_recall": round(min_recall, 3),
                      "minority_class": _py(minority)},
            comment=("As predicted: accuracy is high but minority-class recall is poor — accuracy "
                     "is hiding the failure. Trust PR-AUC / balanced accuracy instead."
                     if held else
                     f"Mixed: accuracy {acc:.0%}, minority recall {min_recall:.0%}. The model is not "
                     "obviously gaming the majority class, but still report imbalance-aware metrics."))]
    else:
        checks = [ReviewCheck(
            hypothesis="Per-class performance (no imbalance warning at init).",
            held=None,
            evidence={"accuracy": round(acc, 3), "minority_recall": round(min_recall, 3)},
            comment=f"Accuracy {acc:.0%}; minority-class recall {min_recall:.0%}; "
                    f"macro recall {macro_recall:.0%}.")]
    return ReviewReport(task="classification", n=n, metrics=metrics, checks=checks)


def review(y_true: Sequence, y_pred: Sequence, roadmap=None, task=None) -> ReviewReport:
    """Check whether the init roadmap's hypothesis held against the real errors.

    Parameters
    ----------
    y_true, y_pred : ground truth and model predictions (labels for classification,
        values for regression). Same length.
    roadmap : the Roadmap returned by `analyze`. Used to pull the task and the
        init findings so each hypothesis can be judged. Optional — without it the
        report still measures the error distribution.
    task : override the task ("regression" / "classification" or a Task enum).
    """
    yt = _to_1d(y_true)
    yp = _to_1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if yt.size == 0:
        raise ValueError("y_true / y_pred are empty.")

    task_str = _resolve_task(task, roadmap, yt)
    fids = _finding_ids(roadmap)
    if task_str == "regression":
        yt = yt.astype(float)
        yp = yp.astype(float)
        keep = ~(np.isnan(yt) | np.isnan(yp))
        return _review_regression(yt[keep], yp[keep], fids)
    if task_str == "classification":
        return _review_classification(yt, yp, fids)
    raise ValueError(
        f"review() supports 'regression' and 'classification'; got task={task_str!r}.")
