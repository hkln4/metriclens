"""Metric policy: one contextual decision, driven by the cost of errors.

The data can say which metrics will *mislead* (evidence-backed "avoid" entries),
but which metric to *optimize* depends on what an error costs — a question for
the user. Until it is answered, the plan is provisional and a broad reporting set
is used.
"""

from __future__ import annotations

from ..enums import DecisionClass, DecisionStatus, Task
from ..profiling.profile import DataProfile
from ..report.model import MetricPlan
from .registry import Emitter, RuleContext

QID = "q.metric_priority"

_BINARY = {
    "miss_costly": (["recall", "average_precision"], ["precision", "balanced_accuracy"], "average_precision",
                    "Missing a positive (false negative) is costlier"),
    "false_alarm_costly": (["precision", "average_precision"], ["recall", "balanced_accuracy"], "average_precision",
                           "A false alarm (false positive) is costlier"),
    "symmetric": (["balanced_accuracy", "f1_macro"], ["matthews_corrcoef", "accuracy"], "balanced_accuracy",
                  "Both errors cost about the same"),
    "ranking": (["roc_auc", "average_precision"], ["balanced_accuracy"], "roc_auc",
                "Only the ranking / scores matter (threshold chosen later)"),
}
_MULTI = {
    "symmetric": (["f1_macro", "balanced_accuracy"], ["accuracy"], "f1_macro", "All classes matter equally"),
    "accuracy_ok": (["accuracy"], ["f1_macro", "balanced_accuracy"], "accuracy",
                    "Overall hit rate is what matters"),
}
_REG = {
    "large_errors_costly": (["neg_root_mean_squared_error"], ["neg_mean_absolute_error", "r2"],
                            "neg_root_mean_squared_error", "Large errors are disproportionately costly"),
    "typical_error": (["neg_mean_absolute_error"], ["neg_median_absolute_error", "neg_root_mean_squared_error", "r2"],
                      "neg_mean_absolute_error", "The typical (absolute) error matters most"),
    "relative_error": (["neg_mean_absolute_percentage_error"], ["neg_mean_absolute_error"],
                       "neg_mean_absolute_percentage_error", "Errors relative to the true value matter"),
}

# short answers accepted by report.answer(metric=...), per option
ALIASES = {
    "miss_costly": ("recall",), "false_alarm_costly": ("precision",), "symmetric": ("balanced_accuracy", "f1"),
    "ranking": ("roc_auc", "auc"), "accuracy_ok": ("accuracy",),
    "large_errors_costly": ("rmse", "mse"), "typical_error": ("mae",), "relative_error": ("mape",),
}
_MULTI_ALIASES = {"symmetric": ("f1_macro", "macro_f1", "balanced_accuracy")}

LABELS = {
    "recall": "recall (positive class)", "precision": "precision (positive class)",
    "average_precision": "PR-AUC / average precision", "balanced_accuracy": "balanced accuracy",
    "f1_macro": "macro F1", "matthews_corrcoef": "MCC", "accuracy": "accuracy", "roc_auc": "ROC-AUC",
    "neg_root_mean_squared_error": "RMSE", "neg_mean_absolute_error": "MAE",
    "neg_median_absolute_error": "median absolute error", "r2": "R²",
    "neg_mean_absolute_percentage_error": "MAPE",
}


def label(scorer: str) -> str:
    return LABELS.get(scorer, scorer)


def natural(scorer: str, value: float | None) -> float | None:
    """sklearn negates error scorers (neg_*); show errors as positive numbers."""
    if value is None:
        return None
    return -value if scorer.startswith("neg_") else value


def metric_policy(p: DataProfile, ctx: RuleContext, out: Emitter, finding_ids: set[str]) -> MetricPlan:
    s = p.target.stats
    notes: list[str] = []
    positive_label: str | None = None
    avoid: dict[str, str] = {}
    evs: list[str] = []
    answer = ctx.answers.get(QID)

    if p.task is Task.CLASSIFICATION:
        binary = s.get("n_classes") == 2
        imbalanced = "f.class_imbalance" in finding_ids
        table = _BINARY if binary else _MULTI
        maj = s.get("majority_fraction")
        evs.append(out.ev("target.majority", "Majority-class share (= accuracy of always predicting it)", maj,
                          "class counts on dev rows", columns=[p.target.name]))
        if imbalanced:
            avoid["accuracy (alone)"] = (f"Always predicting the majority class already scores {maj:.1%}; "
                                         "report it only next to that baseline.")
        if binary:
            positive_label = s.get("minority_class")
            prev = s.get("minority_fraction")
            notes.append(f"Positive class assumed to be the minority class '{positive_label}' "
                              f"(prevalence {prev:.1%}). PR-AUC of a random model equals the prevalence; "
                              "ROC-AUC of a random model is 0.5 regardless of imbalance — report both.")
        default = ("balanced_accuracy" if imbalanced else "accuracy") if binary else (
            "f1_macro" if imbalanced else "accuracy")
        provisional_primary = (["balanced_accuracy", "f1_macro", "average_precision"] if binary and imbalanced
                               else ["balanced_accuracy", "f1_macro"] if imbalanced else ["accuracy", "f1_macro"])
    elif p.task is Task.REGRESSION:
        table = dict(_REG)
        zeros = s.get("zero_fraction") or 0
        neg = s.get("negative_fraction") or 0
        if zeros > 0 or neg > 0:
            table.pop("relative_error")
            avoid["MAPE / MSLE"] = (f"{zeros:.1%} zeros and {neg:.1%} negatives in the target make relative and "
                                    "log-based errors undefined or explosive.")
            evs.append(out.ev("target.zeros_negatives", "Target zero / negative share", {"zero": zeros, "negative": neg},
                              "share of target values == 0 and < 0", columns=[p.target.name]))
        if {"f.target_skew", "f.heavy_tailed_target"} & finding_ids:
            avoid["RMSE (alone)"] = ("The target is skewed / heavy-tailed: RMSE is dominated by a few large values. "
                                     "Report it next to MAE; the RMSE/MAE gap is itself informative.")
            avoid["R² (alone)"] = "Can look healthy while large-error behaviour is poor."
        default = "neg_mean_absolute_error"
        provisional_primary = ["neg_mean_absolute_error", "neg_root_mean_squared_error"]
    else:
        return MetricPlan()

    if answer in table:
        primary, alongside, dev, _ = table[answer]
        status = DecisionStatus.CONFIRMED
        provisional = False
    else:
        primary, alongside, dev = provisional_primary, [], default
        status = DecisionStatus.AWAITING_USER
        provisional = True
        notes.append("Metric priority is unanswered: a broad reporting set is shown and experiments use "
                     f"{label(default)} provisionally.")
    plan = MetricPlan(primary=list(primary), report_alongside=[m for m in alongside if m not in primary],
                      avoid=avoid, notes=notes, dev_metric=dev, dev_metric_provisional=provisional,
                      positive_label=positive_label)

    aliases = {**ALIASES, **(_MULTI_ALIASES if table is _MULTI else {})}
    options = [(k, v[3], "primary: " + ", ".join(label(m) for m in v[0]), {"aliases": aliases.get(k, ())})
               for k, v in table.items()]
    d = out.decision("metric", "Choose the metric from the cost of errors", DecisionClass.CONTEXTUAL, "metrics",
                     "Optimize and report: " + ", ".join(label(m) for m in plan.primary)
                     + (f"; report alongside: {', '.join(label(m) for m in plan.report_alongside)}"
                        if plan.report_alongside else "") + ".",
                     "The data shows which metrics mislead here (see 'avoid'); which one to optimize depends on what "
                     "an error costs, which only you know.",
                     status=status, question_id=QID, evidence_ids=evs,
                     why="Which metric to optimize depends on what an error costs; the data only says which ones "
                         "mislead.",
                     do=("Answer the metric question." if provisional else
                         "Optimize " + label(plan.primary[0]) + "; report the others alongside."),
                     risk="Optimizing a metric that does not match the real cost produces a model that looks good and "
                          "fails where it matters.")
    out.question("metric_priority", "Which kind of error matters most in your application?", options,
                 decision_ids=[d.id], answer=answer, keyword="metric")
    return plan
