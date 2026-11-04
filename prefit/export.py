"""Turn eligible decisions into scikit-learn objects.

The exported pipeline contains only:
- PROCEDURAL decisions (status APPLIED),
- CONTEXTUAL decisions the user CONFIRMED,
- CONDITIONAL decisions an experiment VERIFIED with the user's estimator
  (or SUPPORTED_BY_PROXY, only when explicitly requested).
Everything else is listed in `excluded` with its status, so nothing unconfirmed
enters the pipeline silently. scikit-learn is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .enums import ColumnKind, DecisionClass, DecisionStatus, Task
from .profiling.profile import DataProfile
from .report.model import Step

_INSTALL = "This needs scikit-learn: pip install 'prefit[verify]'"


def _sk():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_INSTALL) from exc


# module-level so pipelines stay picklable
def _to_float(X: Any) -> np.ndarray:
    import pandas as pd

    v = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float).to_numpy(copy=True)
    v[~np.isfinite(v)] = np.nan  # +/-inf is treated as missing (see the infinite_values review item)
    return v


def _to_str(X: Any) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(X)
    return df.astype(object).where(df.notna(), "__missing__").astype(str).to_numpy(dtype=object)


class _PerColumnFill:
    """Fill NaN with one constant per column (module-level, so pipelines pickle)."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def fit(self, X: Any, y: Any = None) -> _PerColumnFill:
        return self

    def transform(self, X: Any) -> np.ndarray:
        v = np.array(X, dtype=float, copy=True)
        for j, val in enumerate(self.values):
            v[np.isnan(v[:, j]), j] = float(val)
        return v

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"values": self.values}

    def set_params(self, **params: Any) -> _PerColumnFill:
        self.values = params.get("values", self.values)
        return self


ESTIMATOR_STEPS = {"log_target", "class_weight"}


@dataclass
class ExportResult:
    """What `Report.export()` builds.

    model         the full estimator recipe, ready for fit / cross_validate: a
                  Pipeline(preprocess, estimator), wrapped in a
                  TransformedTargetRegressor when a target transform is included.
                  None when no estimator was given.
    preprocessor  the feature preprocessing alone (a Pipeline), always present.
    """

    model: Any
    preprocessor: Any
    cv: Any
    scoring: dict[str, Any]
    feature_columns: list[str]
    groups: str | None
    sort_by: str | None
    included: list[dict[str, str]] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    needs_estimator: list[dict[str, str]] = field(default_factory=list)
    unsupported: list[dict[str, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    reviewed: list[dict[str, str]] = field(default_factory=list)
    scaled: bool = True
    report: Any = field(default=None, repr=False, compare=False)

    def fit(self, df: Any = None) -> Any:
        """Fit `model` on the development rows and return it.

        Uses the frame kept by the report (or `df`, the frame passed to analyze;
        fingerprint-checked), without the internal holdout, with exactly
        `feature_columns`, rows with a missing target dropped, sorted by time when
        a time column is set."""
        if self.model is None:
            raise ValueError("No model recipe: pass your estimator to analyze(..., estimator=) or "
                             "report.export(estimator).")
        if self.report is None:
            raise ValueError("This recipe has no report attached; fit it yourself: "
                             "model.fit(dev_df[feature_columns], dev_df[target]).")
        dev, _ = (self.report.holdout(df) if self.report.details.profile.holdout.get("used")
                  else (_dev_frame(self.report, df), None))
        tgt = self.report.target
        dev = dev[dev[tgt].notna()]
        if self.sort_by:
            dev = dev.sort_values(self.sort_by, kind="stable")
        return self.model.fit(dev[self.feature_columns], dev[tgt])

    @property
    def model_kind(self) -> str:
        if self.model is None:
            return "none (pass estimator= to get a model recipe)"
        name = type(self.model).__name__
        if name == "TransformedTargetRegressor":
            return "TransformedTargetRegressor(Pipeline(preprocess, estimator))"
        return "Pipeline(preprocess, estimator)"

    def describe(self) -> str:
        lines = ["EXPORTED MODEL RECIPE", f"model: {self.model_kind}",
                 f"features: {len(self.feature_columns)} columns", f"cv: {self.cv!r}",
                 f"scoring: {list(self.scoring)}",
                 "estimator-dependent default: " + ("StandardScaler on numeric features (scale=False to turn off)"
                                                    if self.scaled else "no feature scaling (scale=False)")]
        if type(self.cv).__name__ == "DuplicateAwareKFold":
            lines.append("cv keeps exact duplicate rows (features + target) in one fold; no groups= needed")
        if self.groups:
            lines.append(f"groups: pass groups=df['{self.groups}'] to cross_validate")
        if self.sort_by:
            lines.append(f"sort rows by '{self.sort_by}' before splitting (TimeSeriesSplit)")
        lines.append("included:")
        lines += [f"  + {d['id']}: {d['title']}" for d in self.included] or ["  (none)"]
        if self.needs_estimator:
            lines.append("eligible but not applied (estimator-level; pass estimator=):")
            lines += [f"  ~ {d['id']}: {d['title']}" for d in self.needs_estimator]
        lines.append("not included:")
        lines += [f"  - {d['id']} [{d['status']}]: {d['reason']}" for d in self.excluded] or ["  (none)"]
        if self.unsupported:
            lines.append("columns left out (not supported raw in v1; may carry signal):")
            lines += [f"  · {d['column']} ({d['kind']})" for d in self.unsupported]
        if self.assumptions:
            lines.append("assumptions still open:")
            lines += [f"  ? {a}" for a in self.assumptions]
        if self.reviewed:
            lines.append("review items you closed:")
            lines += [f"  ✓ {d['title']}: {d['note']}" for d in self.reviewed]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


_EXCLUDED_REASON = {
    "proposed": "not tested (run report.verify(df))",
    "uncertain": "evidence is near the threshold and it was not tested",
    "supported_by_proxy": "only a proxy model supports it (pass include_proxy_supported=True to include)",
    "inconclusive": "experiment could not separate it from noise",
    "no_practical_effect": "experiment shows no practical effect (equivalent within the practical threshold)",
    "refuted": "experiment shows it hurts",
    "not_applicable": "your estimator cannot take this change",
    "skipped_budget": "experiment budget ran out",
    "awaiting_user": "waiting for your answer",
    "declined": "you declined it",
}


def _dev_frame(report: Any, df: Any) -> Any:
    from .api import _frame

    return _frame(report, df, "export().fit")


# --------------------------------------------------------------------------- #
# Builders (shared with verify)
# --------------------------------------------------------------------------- #

def feature_plan(p: DataProfile, steps: list[Step]) -> dict[str, list[str]]:
    drop: set[str] = set()
    power: set[str] = set()
    te: set[str] = set()
    fill: dict[str, Any] = {}
    for s in steps:
        if s.kind == "drop_columns":
            drop |= set(s.params.get("columns", []))
        elif s.kind == "power_transform":
            power |= set(s.params.get("columns", []))
        elif s.kind == "target_encode":
            te |= set(s.params.get("columns", []))
        elif s.kind == "fill_constant":
            fill.update({c: s.params.get("value", 0) for c in s.params.get("columns", [])})
    num, cat = [], []
    for c, cp in p.columns.items():
        if c in drop:
            continue
        if cp.kind in (ColumnKind.NUMERIC, ColumnKind.BOOLEAN):
            num.append(c)
        elif cp.kind is ColumnKind.ID_LIKE and cp.dtype.startswith(("int", "uint", "Int", "UInt")):
            num.append(c)  # status quo until the user confirms it is an identifier
        elif cp.kind is ColumnKind.CATEGORICAL:
            cat.append(c)
    return {
        "numeric": [c for c in num if c not in power and c not in fill],
        "power": [c for c in num if c in power and c not in fill],
        "filled": [c for c in num if c in fill],
        "fill_values": [fill[c] for c in num if c in fill],
        "categorical": [c for c in cat if c not in te],
        "target_encoded": [c for c in cat if c in te],
        "dropped": sorted(drop),
    }


def build_preprocessor(p: DataProfile, steps: list[Step], scale: bool = True) -> tuple[Any, list[str]]:
    _sk()
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        OneHotEncoder,
        PowerTransformer,
        StandardScaler,
        TargetEncoder,
    )

    plan = feature_plan(p, steps)
    missing = set(p.dataset.missing_columns)
    iterative = any(s.kind == "iterative_impute" for s in steps)
    tx = []
    if plan["numeric"]:
        ind = bool(missing & set(plan["numeric"]))
        imputer: Any = SimpleImputer(strategy="median", add_indicator=ind)
        if iterative:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            imputer = IterativeImputer(random_state=0, max_iter=10, skip_complete=True, add_indicator=ind)
        num_steps: list[tuple[str, Any]] = [("to_float", FunctionTransformer(_to_float)), ("impute", imputer)]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        tx.append(("num", Pipeline(num_steps), plan["numeric"]))
    if plan["power"]:
        ind = bool(missing & set(plan["power"]))
        tx.append(("power", Pipeline([("to_float", FunctionTransformer(_to_float)),
                                      ("impute", SimpleImputer(strategy="median", add_indicator=ind)),
                                      ("yeo_johnson", PowerTransformer(method="yeo-johnson"))]), plan["power"]))
    if plan["filled"]:
        fill_steps: list[tuple[str, Any]] = [("to_float", FunctionTransformer(_to_float))]
        values = set(map(float, plan["fill_values"]))
        fill_steps.append(("fill", SimpleImputer(strategy="constant", fill_value=values.pop()) if len(values) == 1
                           else _PerColumnFill(plan["fill_values"])))
        if scale:
            fill_steps.append(("scale", StandardScaler()))
        tx.append(("filled", Pipeline(fill_steps), plan["filled"]))
    if plan["categorical"]:
        tx.append(("cat", Pipeline([("to_str", FunctionTransformer(_to_str)),
                                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                             min_frequency=0.01, sparse_output=False))]),
                   plan["categorical"]))
    if plan["target_encoded"]:
        tx.append(("te", Pipeline([("to_str", FunctionTransformer(_to_str)),
                                   ("target_encode", TargetEncoder(random_state=0))]), plan["target_encoded"]))
    if not tx:
        raise ValueError("No usable feature columns remain after the applied decisions.")
    cols = plan["numeric"] + plan["power"] + plan["filled"] + plan["categorical"] + plan["target_encoded"]
    return ColumnTransformer(tx, remainder="drop", verbose_feature_names_out=False), cols


def proxy_estimators(task: Task) -> list[tuple[str, Any]]:
    _sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if task is Task.CLASSIFICATION:
        return [("linear_proxy", LogisticRegression(max_iter=2000)),
                ("tree_proxy", HistGradientBoostingClassifier(random_state=0))]
    return [("linear_proxy", Ridge()), ("tree_proxy", HistGradientBoostingRegressor(random_state=0))]


def build_model(p: DataProfile, steps: list[Step], estimator: Any, scale: bool = True) -> Any:
    """Preprocessor + estimator, with estimator-level steps applied."""
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    pre, _ = build_preprocessor(p, steps, scale=scale)
    est = clone(estimator)
    for s in steps:
        if s.kind == "class_weight":
            est.set_params(class_weight=s.params.get("value", "balanced"))
    model: Any = Pipeline([("preprocess", pre), ("model", est)])
    if any(s.kind == "log_target" for s in steps):
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def positive_label(p: DataProfile, label: str | None) -> Any:
    if label is None:
        return None
    if "int" in p.target.task_evidence.get("dtype", "") or "float" in p.target.task_evidence.get("dtype", ""):
        v = float(label)
        return int(v) if v.is_integer() else v
    if p.target.task_evidence.get("dtype") == "bool":
        return label == "True"
    return label


def build_scorer(name: str, p: DataProfile, pos_label: Any) -> Any:
    from sklearn.metrics import average_precision_score, f1_score, make_scorer, precision_score, recall_score

    if pos_label is not None:
        if name == "recall":
            return make_scorer(recall_score, pos_label=pos_label, zero_division=0)
        if name == "precision":
            return make_scorer(precision_score, pos_label=pos_label, zero_division=0)
        if name == "f1":
            return make_scorer(f1_score, pos_label=pos_label, zero_division=0)
        if name == "average_precision":
            return make_scorer(average_precision_score, response_method=("decision_function", "predict_proba"),
                               pos_label=pos_label)
    return name


def build_cv(step: Step | None) -> Any:
    from sklearn import model_selection

    if step is None:
        return model_selection.KFold(5, shuffle=True, random_state=0)
    params = dict(step.params)
    name = params.pop("name")
    if params.pop("duplicate_aware", False):
        from .splitters import DuplicateAwareKFold

        return DuplicateAwareKFold(n_splits=params["n_splits"], n_repeats=params.get("n_repeats", 1),
                                   stratify="Stratified" in name, random_state=params.get("random_state", 0))
    return getattr(model_selection, name)(**params)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def export_pipeline(report: Any, estimator: Any = None, include_proxy_supported: bool = False,
                    scale: bool = True) -> ExportResult:
    _sk()
    from sklearn.pipeline import Pipeline

    p = report.details.profile
    ctx = report.details.context
    included, excluded, needs_est, steps = [], [], [], []
    cv_step = None
    for d in report.decisions:
        if d.step is None:
            continue
        if d.step.kind == "cv":
            cv_step = d.step
            continue
        if not d.eligible(include_proxy_supported):
            excluded.append({"id": d.id, "title": d.title, "status": d.status.value,
                             "reason": _EXCLUDED_REASON.get(d.status.value, d.status.value)})
        elif d.step.kind in ESTIMATOR_STEPS and estimator is None:
            needs_est.append({"id": d.id, "title": d.title})
        elif d.step.kind == "class_weight" and "class_weight" not in estimator.get_params():
            excluded.append({"id": d.id, "title": d.title, "status": "not_applicable",
                             "reason": f"{type(estimator).__name__} has no class_weight parameter"})
        else:
            steps.append(d.step)
            included.append({"id": d.id, "title": d.title})

    pre, cols = build_preprocessor(p, steps, scale=scale)
    preprocessor = Pipeline([("preprocess", pre)])
    model = build_model(p, steps, estimator, scale=scale) if estimator is not None else None

    plan = report.metric_plan
    pos = positive_label(p, plan.positive_label)
    names = list(dict.fromkeys([*plan.primary, *plan.report_alongside, *([plan.dev_metric] if plan.dev_metric else [])]))
    scoring = {n: build_scorer(n, p, pos) for n in names}

    from .enums import ColumnKind

    unsupported = [{"column": c, "kind": cp.kind.value} for c, cp in p.columns.items()
                   if cp.kind in (ColumnKind.DATETIME, ColumnKind.TEXT)
                   or (cp.kind is ColumnKind.ID_LIKE and not cp.dtype.startswith(("int", "uint", "Int", "UInt")))]
    assumptions = []
    if p.task_needs_answer and ctx.answers.get("q.task") is None:
        assumptions.append(f"task inferred as {p.task.value} (unconfirmed)")
    if plan.dev_metric_provisional:
        assumptions.append("metric priority unanswered; scoring is a provisional reporting set")
    if plan.positive_label is not None:
        assumptions.append(f"positive class = minority class '{plan.positive_label}'")
    reviewed = []
    for d in report.decisions:
        if d.cls is DecisionClass.REVIEW and d.status is DecisionStatus.REVIEWED:
            reviewed.append({"id": d.id, "title": d.title, "note": d.review_note})
        elif d.cls is DecisionClass.REVIEW:
            assumptions.append(f"review item open: {d.title}")
    return ExportResult(
        model=model, preprocessor=preprocessor, cv=build_cv(cv_step), scoring=scoring, feature_columns=cols,
        groups=ctx.groups if ctx.groups not in (None, "<array>") else None,
        sort_by=ctx.time_col, included=included, excluded=excluded, needs_estimator=needs_est,
        unsupported=unsupported, assumptions=assumptions, reviewed=reviewed, scaled=scale, report=report)
