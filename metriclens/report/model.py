"""The report contract: Evidence -> Finding -> Decision, plus everything that
explains them (figures, experiments, questions) and the manifest that makes the
run reproducible.

Traceability is structural:
- a Finding cannot exist without at least one Evidence id,
- a Decision names the findings/evidence it rests on, its class and its status,
- only some (class, status) pairs are pipeline-eligible (see `Decision.eligible`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from ..enums import DecisionClass, DecisionStatus, Severity, Task, Trigger
from ..profiling.profile import DataProfile


class Evidence(BaseModel):
    """One measured value, how it was measured, and where it is visualized."""

    id: str
    label: str
    value: Any
    ci: tuple[float, float] | None = None
    method: str
    columns: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """A rule fired (or nearly fired) on the evidence."""

    id: str
    rule_id: str
    rule_version: str
    severity: Severity
    trigger: Trigger
    message: str
    evidence_ids: list[str] = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)


class Step(BaseModel):
    """A concrete pipeline change, applied only if the decision is eligible.

    kinds: drop_columns, power_transform, target_encode, log_target, class_weight
    """

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    id: str
    title: str
    cls: DecisionClass
    status: DecisionStatus
    stage: str
    action: str                 # what to do
    rationale: str              # why
    risk: str = ""              # what can go wrong if followed / ignored
    verify_hint: str = ""       # how a user can check it
    rule_id: str = ""
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    question_id: str | None = None
    step: Step | None = None
    trigger: Trigger | None = None
    note: str = ""

    def eligible(self, include_proxy_supported: bool = False) -> bool:
        """Whether this decision's step may enter an exported pipeline."""
        if self.cls is DecisionClass.PROCEDURAL:
            return self.status is DecisionStatus.APPLIED
        if self.cls is DecisionClass.CONTEXTUAL:
            return self.status is DecisionStatus.CONFIRMED
        if self.cls is DecisionClass.CONDITIONAL:
            if self.status is DecisionStatus.VERIFIED:
                return True
            return include_proxy_supported and self.status is DecisionStatus.SUPPORTED_BY_PROXY
        return False


class Option(BaseModel):
    value: str
    label: str
    effect: str = ""


class Question(BaseModel):
    id: str
    text: str
    options: list[Option]
    decision_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    # an LLM may suggest an answer; it is never applied without the user
    suggestion: str | None = None
    suggestion_reason: str | None = None


class MetricPlan(BaseModel):
    primary: list[str] = Field(default_factory=list)
    report_alongside: list[str] = Field(default_factory=list)
    avoid: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    # sklearn scorer used for experiments, and whether it is still provisional
    dev_metric: str = ""
    dev_metric_provisional: bool = True
    positive_label: str | None = None


class Experiment(BaseModel):
    """'Current setup' vs 'current setup + one change', on identical folds."""

    id: str
    decision_id: str
    family: str                      # "user", "linear_proxy", "tree_proxy"
    estimator: str
    metric: str
    n_rows: int
    n_splits: int
    n_repeats: int
    baseline_scores: list[float]
    candidate_scores: list[float]
    mean_diff: float                 # candidate - baseline, in "higher is better" units
    ci: tuple[float, float] | None = None
    p_value: float | None = None     # None when the split protocol admits no valid test
    p_adjusted: float | None = None
    min_effect: float = 0.0          # practical threshold in metric units
    folds_improved: int = 0
    protocol: str = ""               # repeated_kfold / grouped_kfold / time_series
    outcome: str = ""                # improved / worse / negligible / no_difference / not_tested
    seconds: float = 0.0
    method: str = ""

    @property
    def fold_diffs(self) -> list[float]:
        return [c - b for b, c in zip(self.baseline_scores, self.candidate_scores)]


class FigureSpec(BaseModel):
    id: str
    title: str
    what: str                        # what the figure shows
    why: str                         # why it was selected
    evidence_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    priority: int = 50               # lower = shown earlier
    plot: dict[str, Any]             # plotly JSON with color tokens ("var:series-1")
    table: dict[str, Any] = Field(default_factory=dict)   # {"columns": [...], "rows": [[...]]}


class RoadmapStage(BaseModel):
    stage: str
    title: str
    decision_ids: list[str]
    summary: str


class LLMClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)


class SemanticFlag(BaseModel):
    column: str
    concern: str
    reason: str


class LLMInterpretation(BaseModel):
    provider: str
    model: str
    summary: str
    priorities: list[LLMClaim] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    semantic_flags: list[SemanticFlag] = Field(default_factory=list)
    dropped_claims: int = 0          # claims rejected for citing unknown ids/columns
    disclosure: dict[str, Any] = Field(default_factory=dict)


class ContextState(BaseModel):
    text: str | None = None
    answers: dict[str, str] = Field(default_factory=dict)
    groups: str | None = None        # column name, "<array>", or None
    time_col: str | None = None
    estimator: str | None = None


class Manifest(BaseModel):
    metriclens_version: str
    python: str
    packages: dict[str, str]
    seed: int
    rule_versions: dict[str, str]
    profile_options: dict[str, Any]
    budget: dict[str, Any] | None = None
    benchmark: str | None = None
    # every step that produced this report version: analyze, answer, verify, interpret
    revision: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)


class Report(BaseModel):
    """Everything `analyze` produced. JSON-serializable; holds no raw rows."""

    target: str
    task: Task
    profile: DataProfile
    evidence: dict[str, Evidence]
    findings: list[Finding]
    decisions: list[Decision]
    questions: list[Question]
    metric_plan: MetricPlan
    roadmap: list[RoadmapStage]
    figures: list[FigureSpec]
    experiments: list[Experiment] = Field(default_factory=list)
    baseline: dict[str, Any] = Field(default_factory=dict)
    llm: LLMInterpretation | None = None
    limitations: list[str] = Field(default_factory=list)
    context: ContextState
    manifest: Manifest
    rule_track_record: dict[str, Any] = Field(default_factory=dict)

    _holdout_positions: Any = PrivateAttr(default=None)
    _estimator: Any = PrivateAttr(default=None)

    # ---- lookups ---------------------------------------------------------- #
    def decision(self, decision_id: str) -> Decision:
        for d in self.decisions:
            if d.id == decision_id:
                return d
        raise KeyError(decision_id)

    def finding(self, finding_id: str) -> Finding:
        for f in self.findings:
            if f.id == finding_id:
                return f
        raise KeyError(finding_id)

    @property
    def pending_questions(self) -> list[Question]:
        return [q for q in self.questions if q.answer is None]

    # ---- workflow --------------------------------------------------------- #
    def answer(self, answers: dict[str, str]) -> Report:
        """Return a new report with these question answers applied.

        Contextual decisions are re-derived from the stored profile; earlier
        experiment results are kept and flagged if the metric changed."""
        from ..api import _apply_answers

        return _apply_answers(self, answers)

    def verify(self, df: Any, estimator: Any = None, budget: Any = None) -> Report:
        """Run the bounded experiments for conditional decisions on the dev rows of `df`."""
        from ..api import _verify

        return _verify(self, df, estimator=estimator, budget=budget)

    def split(self, df: Any) -> tuple[Any, Any]:
        """Return (dev_df, holdout_df) exactly as reserved by `analyze`."""
        import numpy as np

        if self._holdout_positions is None:
            raise RuntimeError("Holdout positions are not available (report was loaded from JSON). "
                               "Re-run ml.analyze with the same data and seed.")
        hold = np.asarray(self._holdout_positions, dtype=int)
        mask = np.zeros(len(df), dtype=bool)
        mask[hold] = True
        return df.iloc[~mask], df.iloc[mask]

    def export(self, estimator: Any = None, include_proxy_supported: bool = False) -> Any:
        """Build the sklearn Pipeline + CV + scoring from eligible decisions only."""
        from ..export import export_pipeline

        return export_pipeline(self, estimator=estimator or self._estimator,
                               include_proxy_supported=include_proxy_supported)

    def interpret(self, llm: Any) -> Report:
        """Add the optional LLM interpretation (see `llm_payload` for what is sent)."""
        from ..llm.interpreter import interpret_report

        return interpret_report(self, llm)

    def llm_payload(self, llm: Any = None) -> dict[str, Any]:
        """Exactly what would leave the machine if `interpret` were called."""
        from ..llm.config import LLMConfig
        from ..llm.payload import build_payload

        cfg = llm if isinstance(llm, LLMConfig) else LLMConfig()
        return build_payload(self, cfg)[0]

    # ---- rendering -------------------------------------------------------- #
    def to_text(self) -> str:
        from ..render.text import render_text

        return render_text(self, md=False)

    def to_markdown(self) -> str:
        from ..render.text import render_text

        return render_text(self, md=True)

    def to_html(self, path: str | None = None, review: Any = None, plotlyjs: str = "cdn") -> str:
        """Render the single-file HTML report; write it to `path` if given."""
        from ..render.html import render_report_html

        html = render_report_html(self, review=review, plotlyjs=plotlyjs)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = self.model_dump_json(indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def __str__(self) -> str:
        return self.to_text()

    # ---- v0 compatibility ------------------------------------------------- #
    @property
    def recommended_metrics(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if self.metric_plan.primary:
            out["primary"] = list(self.metric_plan.primary)
        if self.metric_plan.report_alongside:
            out["secondary"] = list(self.metric_plan.report_alongside)
        return out

    @property
    def avoid(self) -> dict[str, str]:
        return dict(self.metric_plan.avoid)

    @property
    def validation_strategy(self) -> list[str]:
        return [d.action for d in self.decisions if d.stage == "split"]

    @property
    def preprocessing_notes(self) -> list[str]:
        return [d.action for d in self.decisions if d.stage in ("preprocessing", "target")]
