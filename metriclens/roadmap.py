"""The Roadmap contract: the final output of `analyze`.

A Roadmap is really a *hypothesis* formed before any model is trained: "given
these data properties, report these metrics, distrust those, validate this way".
The optional `review` pass later checks whether that hypothesis held.

It renders both as a rich object (for CI / programmatic use) and as markdown
(for reading). The raw `findings` are kept for transparency so a user can always
trace a recommendation back to the number that produced it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .rules.finding import Finding


class Roadmap(BaseModel):
    """Reasoned plan for how to evaluate a project, grounded in the data."""

    # primary/secondary metric names with rationale (aggregated from findings)
    recommended_metrics: dict[str, list[str]] = Field(default_factory=dict)
    # metric name -> why it will mislead on this data
    avoid: dict[str, str] = Field(default_factory=dict)
    # e.g. "stratified k-fold due to 1:47 imbalance"
    validation_strategy: list[str] = Field(default_factory=list)
    preprocessing_notes: list[str] = Field(default_factory=list)
    # LLM-written prose, or the deterministic fallback when no LLM is used
    narrative: str = ""
    # whether the narrative came from the LLM or the deterministic fallback
    narrative_source: str = "deterministic"
    # compact view of the deterministic profiler's computed metrics, rendered
    # first so every roadmap visibly stands on the numbers that produced it.
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    # raw findings, kept for full transparency / auditability
    findings: list[Finding] = Field(default_factory=list)

    def to_text(self) -> str:
        """Plain-text render — no markdown symbols. This is what `print()` shows,
        so it stays readable wherever it lands (a terminal, a raw notebook cell)."""
        return self._render(md=False)

    def to_markdown(self) -> str:
        """Markdown render, for places that actually display markdown."""
        return self._render(md=True)

    def _render(self, md: bool) -> str:
        """One section per pipeline stage, always in order:
        1. computed metrics (deterministic profiler),
        2. the rule-based roadmap (rule engine),
        3. the interpretation (LLM, or deterministic fallback).
        A section is omitted only if that stage produced nothing."""
        title = "# Evaluation Roadmap" if md else "EVALUATION ROADMAP"
        lines: list[str] = [title, ""]
        lines += self._render_metrics(md)
        lines += self._render_rules(md)
        lines += self._render_interpretation(md)
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _head(text: str, md: bool) -> str:
        return f"## {text}" if md else text.upper()

    @staticmethod
    def _label(text: str, md: bool) -> str:
        return f"**{text}**" if md else f"{text}:"

    def _render_metrics(self, md: bool) -> list[str]:
        ps = self.profile_summary
        if not ps:
            return []
        out = [self._head("1. Computed metrics · deterministic profiler", md)]
        task = ps.get("task")
        modality = ps.get("modality")
        head = (f"- Modality: **{modality}** · Task: **{task}**" if md
                else f"- Modality: {modality} · Task: {task}")
        if ps.get("task_inferred"):
            head += " _(inferred — confirm)_" if md else " (inferred — confirm)"
        out.append(head)
        size = f"- Samples: {ps.get('n_samples')}"
        if ps.get("n_features") is not None:
            size += f" · Features: {ps.get('n_features')}"
        out.append(size)
        target = ps.get("target") or {}
        if task == "classification":
            if ps.get("n_classes") is not None:
                line = f"- Classes: {ps['n_classes']}"
                if ps.get("imbalance_ratio") is not None:
                    line += f" · Imbalance: {ps['imbalance_ratio']:.1f}:1"
                if target.get("minority_fraction") is not None:
                    line += f" · Minority: {target['minority_fraction']:.1%}"
                out.append(line)
        elif task == "regression" and target.get("skew") is not None:
            out.append(
                f"- Target: skew={target['skew']:.2f} · "
                f"IQR-outliers={target.get('outlier_fraction', 0.0):.1%} · "
                f"std={target.get('std', float('nan')):.3g}")
        feats = ps.get("features") or {}
        bits: list[str] = []
        if feats.get("max_missing_fraction"):
            bits.append(f"max missing={feats['max_missing_fraction']:.1%}")
        if feats.get("constant_columns"):
            bits.append(f"constant={feats['constant_columns']}")
        if feats.get("near_unique_columns"):
            bits.append(f"id-like={feats['near_unique_columns']}")
        if feats.get("leakage_hints"):
            bits.append(f"leakage hints={len(feats['leakage_hints'])}")
        if feats.get("duplicate_fraction"):
            bits.append(f"duplicates={feats['duplicate_fraction']:.1%}")
        if bits:
            out.append("- Features: " + " · ".join(bits))
        out.append("")
        return out

    def _render_rules(self, md: bool) -> list[str]:
        out = [self._head("2. Rule-based roadmap · rule engine", md)]
        if self.recommended_metrics:
            out.append(self._label("Recommended metrics", md))
            for tier, metrics in self.recommended_metrics.items():
                out.append(f"- {tier}: {', '.join(metrics)}")
        if self.avoid:
            out.append(self._label("Avoid", md))
            for metric, reason in self.avoid.items():
                out.append(f"- {metric} — {reason}")
        if self.validation_strategy:
            out.append(self._label("Validation", md))
            out += [f"- {v}" for v in self.validation_strategy]
        if self.preprocessing_notes:
            out.append(self._label("Preprocessing", md))
            out += [f"- {p}" for p in self.preprocessing_notes]
        if self.findings:
            out.append(self._label("Findings", md))
            out += [f"- [{f.severity.value}] {f.id}: {f.message}"
                    for f in self.findings]
        else:
            out.append("- No issues detected; conventional metrics for this "
                       "task apply.")
        out.append("")
        return out

    def _render_interpretation(self, md: bool) -> list[str]:
        if not self.narrative:
            return []
        src = "LLM" if self.narrative_source == "llm" else "deterministic"
        return [self._head(f"3. Interpretation · {src}", md), self.narrative, ""]

    # --- executable output: turn the recommendation into a runnable setup -----

    def validation_recipe(self) -> dict[str, Any]:
        """Concrete, dependency-free validation plan derived from the findings.

        Recommendations are only useful if you can act on them; this turns the
        prose into a structured recipe (which splitter, which scorers) that a
        caller can read directly or hand to `make_sklearn()`."""
        ids = {f.id for f in self.findings}
        task = self.profile_summary.get("task")
        n_splits = 5
        if {"duplicate_rows", "id_like_features"} & ids:
            splitter = {"name": "GroupKFold", "params": {"n_splits": n_splits},
                        "reason": "duplicates / id-like columns present — keep related rows "
                                  "on one side of the split; pass groups=..."}
        elif task == "classification":
            splitter = {"name": "StratifiedKFold",
                        "params": {"n_splits": n_splits, "shuffle": True, "random_state": 0},
                        "reason": "preserve class ratios across folds"}
        else:
            splitter = {"name": "KFold",
                        "params": {"n_splits": n_splits, "shuffle": True, "random_state": 0},
                        "reason": "standard k-fold cross-validation"}
        if "small_data" in ids:
            splitter["reason"] += "; consider repeated CV given the small sample"
        return {
            "splitter": splitter,
            "primary_metrics": self.recommended_metrics.get("primary", []),
            "secondary_metrics": self.recommended_metrics.get("secondary", []),
            "avoid": dict(self.avoid),
        }

    def make_sklearn(self) -> dict[str, Any]:
        """Build real scikit-learn objects from the recipe: a CV splitter plus a
        `scoring` dict ready for cross_validate. scikit-learn is imported lazily so
        the core library never depends on it."""
        try:
            from sklearn import model_selection
        except ImportError as exc:  # pragma: no cover
            raise ImportError("make_sklearn() needs scikit-learn: pip install scikit-learn") from exc
        recipe = self.validation_recipe()
        splitter_cls = getattr(model_selection, recipe["splitter"]["name"])
        params = dict(recipe["splitter"]["params"])
        if recipe["splitter"]["name"] == "GroupKFold":
            params.pop("shuffle", None)
            params.pop("random_state", None)
        cv = splitter_cls(**params)
        scoring: dict[str, str] = {}
        for metric in recipe["primary_metrics"] + recipe["secondary_metrics"]:
            mapped = _to_sklearn_scorer(metric)
            if mapped is not None:
                scoring[mapped] = mapped
        return {"cv": cv, "scoring": scoring}

    def __str__(self) -> str:  # so `print(roadmap)` shows clean plain text
        return self.to_text()
    
# keyword (lowercased, matched as substring) -> sklearn scoring string. Order
# matters: the first keyword found in a metric label wins.
_SKLEARN_SCORERS: list[tuple[str, str]] = [
    ("average precision", "average_precision"),
    ("pr-auc", "average_precision"),
    ("balanced accuracy", "balanced_accuracy"),
    ("macro f1", "f1_macro"),
    ("mcc", "matthews_corrcoef"),
    ("roc-auc", "roc_auc"),
    ("median absolute error", "neg_median_absolute_error"),
    ("mae", "neg_mean_absolute_error"),
    ("rmse", "neg_root_mean_squared_error"),
    ("r²", "r2"),
]

def _to_sklearn_scorer(metric: str) -> str | None:
    """Map a recommended metric label to a built-in sklearn scoring name, or None
    if there is no direct built-in (e.g. Tweedie deviance, Huber loss)."""
    low = metric.lower()
    for key, scorer in _SKLEARN_SCORERS:
        if key in low:
            return scorer
    return None

def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def summarize_profile(profile) -> dict:
    """Compact, render-friendly view of the profiler output. Numbers only —
    the same trust-boundary data the LLM sees, never raw rows."""
    if profile is None:
        return {}
    core = profile.core
    summary: dict[str, Any] = {
        "modality": profile.modality.value,
        "task": profile.task.value,
        "task_inferred": profile.task_inferred,
        "n_samples": core.n_samples,
        "n_features": core.n_features,
    }
    if core.n_classes is not None:
        summary["n_classes"] = core.n_classes
        summary["imbalance_ratio"] = core.imbalance_ratio
    target = profile.modality_stats.get("target") or {}
    if target:
        summary["target"] = target
    features = profile.modality_stats.get("features") or {}
    feat_flags = {k: features[k] for k in
                  ("max_missing_fraction", "constant_columns", "near_unique_columns",
                   "leakage_hints", "duplicate_fraction") if features.get(k)}
    if feat_flags:
        summary["features"] = feat_flags
    return summary

def assemble_roadmap(findings, narrative, narrative_source, profile=None):
    primary, secondary = [], []
    avoid, validation, preprocessing = {}, [], []
    for f in findings:  # already sorted critical-first
        primary += f.candidate_metrics.primary
        secondary += f.candidate_metrics.secondary
        for pit in f.pitfalls:
            avoid.setdefault(pit.metric, pit.reason)
        validation += f.validation_notes
        preprocessing += f.preprocessing_notes
    recommended = {}
    if primary:   recommended["primary"] = _dedup(primary)
    if secondary: recommended["secondary"] = _dedup(secondary)
    return Roadmap(recommended_metrics=recommended, avoid=avoid,
                   validation_strategy=_dedup(validation),
                   preprocessing_notes=_dedup(preprocessing),
                   narrative=narrative, narrative_source=narrative_source,
                   profile_summary=summarize_profile(profile),
                   findings=findings)