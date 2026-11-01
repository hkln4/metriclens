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

from .enums import ColumnKind, DecisionClass, Task
from .profiling.profile import DataProfile
from .report.model import Step

_INSTALL = "This needs scikit-learn: pip install 'metriclens[verify]'"


def _sk():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_INSTALL) from exc


# module-level so pipelines stay picklable
def _to_float(X: Any) -> np.ndarray:
    import pandas as pd

    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()


def _to_str(X: Any) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(X)
    return df.astype(object).where(df.notna(), "__missing__").astype(str).to_numpy(dtype=object)


@dataclass
class ExportResult:
    pipeline: Any
    cv: Any
    scoring: dict[str, Any]
    feature_columns: list[str]
    groups: str | None
    sort_by: str | None
    included: list[dict[str, str]] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = ["EXPORTED PIPELINE", f"features: {len(self.feature_columns)} columns",
                 f"cv: {self.cv!r}", f"scoring: {list(self.scoring)}"]
        if self.groups:
            lines.append(f"groups: pass groups=df['{self.groups}'] to cross_validate")
        if self.sort_by:
            lines.append(f"sort rows by '{self.sort_by}' before splitting (TimeSeriesSplit)")
        lines.append("included:")
        lines += [f"  + {d['id']}: {d['title']}" for d in self.included] or ["  (none)"]
        lines.append("not included:")
        lines += [f"  - {d['id']} [{d['status']}]: {d['reason']}" for d in self.excluded] or ["  (none)"]
        if self.assumptions:
            lines.append("assumptions still open:")
            lines += [f"  ? {a}" for a in self.assumptions]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


_EXCLUDED_REASON = {
    "proposed": "not tested (run report.verify(df))",
    "uncertain": "evidence is near the threshold and it was not tested",
    "supported_by_proxy": "only a proxy model supports it (pass include_proxy_supported=True to include)",
    "inconclusive": "experiment could not separate it from noise",
    "refuted": "experiment shows it hurts",
    "not_applicable": "your estimator cannot take this change",
    "skipped_budget": "experiment budget ran out",
    "awaiting_user": "waiting for your answer",
    "declined": "you declined it",
}


# --------------------------------------------------------------------------- #
# Builders (shared with verify)
# --------------------------------------------------------------------------- #

def feature_plan(p: DataProfile, steps: list[Step]) -> dict[str, list[str]]:
    drop: set[str] = set()
    power: set[str] = set()
    te: set[str] = set()
    for s in steps:
        if s.kind == "drop_columns":
            drop |= set(s.params.get("columns", []))
        elif s.kind == "power_transform":
            power |= set(s.params.get("columns", []))
        elif s.kind == "target_encode":
            te |= set(s.params.get("columns", []))
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
        "numeric": [c for c in num if c not in power],
        "power": [c for c in num if c in power],
        "categorical": [c for c in cat if c not in te],
        "target_encoded": [c for c in cat if c in te],
        "dropped": sorted(drop),
    }


def build_preprocessor(p: DataProfile, steps: list[Step]) -> tuple[Any, list[str]]:
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
    tx = []
    if plan["numeric"]:
        ind = bool(missing & set(plan["numeric"]))
        tx.append(("num", Pipeline([("to_float", FunctionTransformer(_to_float)),
                                    ("impute", SimpleImputer(strategy="median", add_indicator=ind)),
                                    ("scale", StandardScaler())]), plan["numeric"]))
    if plan["power"]:
        ind = bool(missing & set(plan["power"]))
        tx.append(("power", Pipeline([("to_float", FunctionTransformer(_to_float)),
                                      ("impute", SimpleImputer(strategy="median", add_indicator=ind)),
                                      ("yeo_johnson", PowerTransformer(method="yeo-johnson"))]), plan["power"]))
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
    cols = plan["numeric"] + plan["power"] + plan["categorical"] + plan["target_encoded"]
    return ColumnTransformer(tx, remainder="drop", verbose_feature_names_out=False), cols


def proxy_estimators(task: Task) -> list[tuple[str, Any]]:
    _sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if task is Task.CLASSIFICATION:
        return [("linear_proxy", LogisticRegression(max_iter=2000)),
                ("tree_proxy", HistGradientBoostingClassifier(random_state=0))]
    return [("linear_proxy", Ridge()), ("tree_proxy", HistGradientBoostingRegressor(random_state=0))]


def build_model(p: DataProfile, steps: list[Step], estimator: Any) -> Any:
    """Preprocessor + estimator, with estimator-level steps applied."""
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    pre, _ = build_preprocessor(p, steps)
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
    return getattr(model_selection, name)(**params)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def export_pipeline(report: Any, estimator: Any = None, include_proxy_supported: bool = False) -> ExportResult:
    _sk()
    from sklearn.pipeline import Pipeline

    p = report.profile
    included, excluded, steps = [], [], []
    cv_step = None
    for d in report.decisions:
        if d.step is None:
            continue
        if d.step.kind == "cv":
            cv_step = d.step
            continue
        if d.eligible(include_proxy_supported):
            steps.append(d.step)
            included.append({"id": d.id, "title": d.title})
        else:
            excluded.append({"id": d.id, "title": d.title, "status": d.status.value,
                             "reason": _EXCLUDED_REASON.get(d.status.value, d.status.value)})
    if estimator is not None and any(s.kind == "class_weight" for s in steps):
        if "class_weight" not in estimator.get_params():
            steps = [s for s in steps if s.kind != "class_weight"]
    if estimator is not None:
        pipeline = build_model(p, steps, estimator)
    else:
        pipeline, _ = build_preprocessor(p, steps)
        pipeline = Pipeline([("preprocess", pipeline)])
    _, cols = build_preprocessor(p, steps)

    plan = report.metric_plan
    pos = positive_label(p, plan.positive_label)
    names = list(dict.fromkeys(plan.primary + plan.report_alongside + ([plan.dev_metric] if plan.dev_metric else [])))
    scoring = {n: build_scorer(n, p, pos) for n in names}

    assumptions = []
    if p.task_inferred and report.context.answers.get("q.task") is None:
        assumptions.append(f"task inferred as {p.task.value} (unconfirmed)")
    if plan.dev_metric_provisional:
        assumptions.append("metric priority unanswered; scoring is a provisional reporting set")
    if plan.positive_label is not None:
        assumptions.append(f"positive class = minority class '{plan.positive_label}'")
    for d in report.decisions:
        if d.cls is DecisionClass.REVIEW:
            assumptions.append(f"review item open: {d.title}")
    return ExportResult(
        pipeline=pipeline, cv=build_cv(cv_step), scoring=scoring, feature_columns=cols,
        groups=report.context.groups if report.context.groups not in (None, "<array>") else None,
        sort_by=report.context.time_col, included=included, excluded=excluded, assumptions=assumptions)
