"""The report contract: Evidence -> Finding -> Decision, plus everything that
explains them (figures, experiments, questions) and the manifest that makes the
run reproducible.

Traceability is structural:
- a Finding cannot exist without at least one Evidence id,
- a Decision names the findings/evidence it rests on, its class and its status,
- only some (class, status) pairs are pipeline-eligible (see `Decision.eligible`).

Every model here is deeply immutable (see `prefit.frozen`). answer / verify /
interpret / register_external_holdout return a new Report with an incremented
`details.manifest.revision` and a history entry; the old report is untouched.

Surface:
- reading:     print(report), to_html, to_markdown, roadmap, pending_questions(),
               decisions_table(), target, task
- flow:        answer, resolve, change_task, verify, export, interpret, holdout,
               register_external_holdout
- automation:  decisions, decision(id), questions, experiments, metric_plan,
               baseline, cv_estimate(), to_json, llm_payload
- internals:   report.details.{findings, finding(id), evidence, profile, figures,
               manifest, context, limitations, rule_track_record, llm}
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

from ..enums import DecisionClass, DecisionStatus, Severity, Task, Trigger
from ..frozen import Frozen
from ..profiling.profile import DataProfile


def _short(v: Any, n: int = 60) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


class Evidence(Frozen):
    """One measured value, how it was measured, and where it is visualized."""

    id: str
    label: str
    value: Any
    ci: tuple[float, float] | None = None
    method: str
    columns: tuple[str, ...] = ()
    figure_ids: tuple[str, ...] = ()

    def __repr__(self) -> str:
        ci = f" [{self.ci[0]:.4g}, {self.ci[1]:.4g}]" if self.ci else ""
        return f"<Evidence {self.label} = {_short(self.value)}{ci}>"


class Finding(Frozen):
    """A rule fired (or nearly fired) on the evidence."""

    id: str
    rule_id: str
    rule_version: str
    severity: Severity
    trigger: Trigger
    message: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    thresholds: dict[str, float] = Field(default_factory=dict)
    columns: tuple[str, ...] = ()
    task_dependent: bool = False     # only meaningful under the current task (e.g. class imbalance)

    def __repr__(self) -> str:
        near = ", near threshold" if self.trigger is Trigger.UNCERTAIN else ""
        return f"<Finding {self.severity.value}{near} · {self.rule_id}: {_short(self.message, 90)}>"


class Step(Frozen):
    """A concrete pipeline change, applied only if the decision is eligible.

    kinds: drop_columns, power_transform, target_encode, log_target, class_weight, cv
    """

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class Decision(Frozen):
    id: str
    title: str
    cls: DecisionClass
    status: DecisionStatus
    stage: str
    action: str                 # what to do (full)
    rationale: str              # why (full)
    why_short: str = ""         # one sentence, with the decisive number when there is one
    do_short: str = ""          # one sentence
    risk: str = ""              # what can go wrong if followed / ignored
    verify_hint: str = ""       # how a user can check it
    rule_id: str = ""
    finding_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    figure_ids: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    question_id: str | None = None
    step: Step | None = None
    trigger: Trigger | None = None
    note: str = ""
    task_dependent: bool = False  # only meaningful under the current task; flagged while it is unconfirmed
    standard: bool = False        # standard practice that appears on every dataset (fit in fold, holdout, ...)
    review_note: str = ""         # the user's note when a review item was closed with report.resolve
    # the other ways to go, each with when it is the right one ("median imputation — when ...")
    alternatives: tuple[str, ...] = ()

    @property
    def state(self) -> str:
        """The three states shown to the user: "done", "needs_you" or "tested"."""
        return STATE.get(self.status, "done")

    def eligible(self, include_proxy_supported: bool = False) -> bool:
        """Whether this decision's step may enter an exported model recipe."""
        if self.cls is DecisionClass.PROCEDURAL:
            return self.status is DecisionStatus.APPLIED
        if self.cls is DecisionClass.CONTEXTUAL:
            return self.status is DecisionStatus.CONFIRMED
        if self.cls is DecisionClass.CONDITIONAL:
            if self.status is DecisionStatus.VERIFIED:
                return True
            return include_proxy_supported and self.status is DecisionStatus.SUPPORTED_BY_PROXY
        return False

    def __repr__(self) -> str:
        return f"<Decision {self.status.value} · {self.cls.value} · {self.title}>"


STATE: dict[DecisionStatus, str] = {
    # done: nothing left to do
    DecisionStatus.APPLIED: "done", DecisionStatus.CONFIRMED: "done", DecisionStatus.DECLINED: "done",
    DecisionStatus.REVIEWED: "done", DecisionStatus.NOT_APPLICABLE: "done",
    # needs you: an answer, a review, or an experiment you have not run
    DecisionStatus.AWAITING_USER: "needs_you", DecisionStatus.NEEDS_REVIEW: "needs_you",
    DecisionStatus.PROPOSED: "needs_you", DecisionStatus.UNCERTAIN: "needs_you",
    DecisionStatus.SKIPPED_BUDGET: "needs_you",
    # tested: an experiment settled it (in or out of the recipe, see the status)
    DecisionStatus.VERIFIED: "tested", DecisionStatus.REFUTED: "tested", DecisionStatus.INCONCLUSIVE: "tested",
    DecisionStatus.NO_PRACTICAL_EFFECT: "tested", DecisionStatus.SUPPORTED_BY_PROXY: "tested",
}


class Option(Frozen):
    value: str
    label: str
    effect: str = ""
    aliases: tuple[str, ...] = ()    # other accepted answers, e.g. "mae" for "typical_error"
    flag: bool | None = None         # the True/False answer that selects this option (column questions)


class Question(Frozen):
    id: str
    text: str
    options: tuple[Option, ...]
    decision_ids: tuple[str, ...] = ()
    answer: str | None = None
    # report.answer(<keyword>=...) answers it; column questions take {column: True/False}
    keyword: str = ""
    column: str | None = None
    # an LLM may suggest an answer; it is never applied without the user
    suggestion: str | None = None
    suggestion_reason: str | None = None

    def __repr__(self) -> str:
        state = f"answer={self.answer!r}" if self.answer else "unanswered"
        sug = f", suggested={self.suggestion!r}" if self.suggestion and not self.answer else ""
        return f"<Question {self.id[2:]}: {self.text} options={[o.value for o in self.options]} {state}{sug}>"


class MetricPlan(Frozen):
    primary: tuple[str, ...] = ()
    report_alongside: tuple[str, ...] = ()
    avoid: dict[str, str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    # sklearn scorer used for experiments, and whether it is still provisional
    dev_metric: str = ""
    dev_metric_provisional: bool = True
    positive_label: str | None = None

    def __repr__(self) -> str:
        prov = " (provisional)" if self.dev_metric_provisional else ""
        return f"<MetricPlan primary={list(self.primary)}{prov} avoid={list(self.avoid)}>"


class Experiment(Frozen):
    """'Current setup' vs 'current setup + one change', on identical folds."""

    id: str
    decision_id: str
    family: str                      # "user", "linear_proxy", "tree_proxy"
    estimator: str
    metric: str
    n_rows: int
    n_splits: int
    n_repeats: int
    cv: str = ""                     # the splitter actually used
    baseline_scores: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    mean_diff: float                 # candidate - baseline, in "higher is better" units
    ci: tuple[float, float] | None = None
    p_value: float | None = None     # None when the split protocol admits no valid test
    p_adjusted: float | None = None
    # equivalence (TOST against ±min_effect, same corrected SE), Holm-adjusted across the run
    p_equivalence: float | None = None
    min_effect: float = 0.0          # practical threshold in metric units
    folds_improved: int = 0
    protocol: str = ""               # repeated_kfold / grouped_kfold / time_series
    outcome: str = ""  # improved / worse / negligible / no_practical_effect / no_difference / not_tested
    seconds: float = 0.0
    method: str = ""

    @property
    def fold_diffs(self) -> list[float]:
        return [c - b for b, c in zip(self.baseline_scores, self.candidate_scores)]

    def __repr__(self) -> str:
        return (f"<Experiment {self.decision_id} · {self.family}: diff {self.mean_diff:+.4g} "
                f"({self.folds_improved}/{len(self.baseline_scores)} folds better) -> {self.outcome}>")


class FigureSpec(Frozen):
    id: str
    title: str
    what: str                        # what the figure shows
    why: str                         # why it was selected
    evidence_ids: tuple[str, ...] = ()
    decision_ids: tuple[str, ...] = ()
    priority: int = 50               # lower = shown earlier
    plot: dict[str, Any]             # plotly JSON with color tokens ("var:series-1")
    table: dict[str, Any] = Field(default_factory=dict)   # {"columns": [...], "rows": [[...]]}

    def __repr__(self) -> str:
        return f"<FigureSpec {self.id}: {self.title}>"


class RoadmapStage(Frozen):
    stage: str
    title: str
    decision_ids: tuple[str, ...]
    summary: str

    def __repr__(self) -> str:
        return f"<RoadmapStage {self.title} · {self.summary}>"


class LLMClaim(Frozen):
    text: str
    evidence_ids: tuple[str, ...] = ()
    decision_ids: tuple[str, ...] = ()


class SemanticFlag(Frozen):
    column: str
    concern: str
    reason: str


class LLMInterpretation(Frozen):
    provider: str
    model: str
    summary: str
    priorities: tuple[LLMClaim, ...] = ()
    questions_for_user: tuple[str, ...] = ()
    semantic_flags: tuple[SemanticFlag, ...] = ()
    dropped_claims: int = 0          # claims rejected for citing unknown ids/columns
    dropped: tuple[str, ...] = ()    # what was rejected, and why (one line each)
    raw_response: str = ""           # the model's reply exactly as received, for audit
    disclosure: dict[str, Any] = Field(default_factory=dict)


class ContextState(Frozen):
    text: str | None = None
    answers: dict[str, str] = Field(default_factory=dict)
    groups: str | None = None        # column name, "<array>", or None
    time_col: str | None = None
    estimator: str | None = None
    split_columns_as_features: bool = False
    # review items closed by the user: decision id -> note (report.resolve)
    resolutions: dict[str, str] = Field(default_factory=dict)


class Manifest(Frozen):
    prefit_version: str
    python: str
    packages: dict[str, str]
    seed: int
    rule_versions: dict[str, str]
    profile_options: dict[str, Any]
    budget: dict[str, Any] | None = None
    benchmark: str | None = None
    # digests of the DataFrame passed to analyze (no data): see prefit.fingerprint
    source_fingerprint: dict[str, Any] = Field(default_factory=dict)
    # metadata of an evaluation partition kept out of the analysis (no data)
    external_holdout: dict[str, Any] | None = None
    # every step that produced this report version: analyze, answer, verify, interpret, ...
    revision: int = 0
    history: tuple[dict[str, Any], ...] = ()


class ReportDetails(Frozen):
    """Internals: the traceability chain and reproducibility record behind the decisions."""

    findings: tuple[Finding, ...]
    evidence: dict[str, Evidence]
    profile: DataProfile
    figures: tuple[FigureSpec, ...]
    manifest: Manifest
    context: ContextState
    limitations: tuple[str, ...] = ()
    rule_track_record: dict[str, Any] = Field(default_factory=dict)
    llm: LLMInterpretation | None = None

    def finding(self, finding_id: str) -> Finding:
        for f in self.findings:
            if f.id == finding_id:
                return f
        raise KeyError(finding_id)

    def __repr__(self) -> str:
        return (f"<ReportDetails {len(self.findings)} findings, {len(self.evidence)} evidence, "
                f"{len(self.figures)} figures, revision {self.manifest.revision}>")


class Report(Frozen):
    """Everything `analyze` produced. JSON-serializable; holds no raw rows."""

    target: str
    task: Task
    decisions: tuple[Decision, ...]
    questions: tuple[Question, ...]
    metric_plan: MetricPlan
    roadmap: tuple[RoadmapStage, ...]
    experiments: tuple[Experiment, ...] = ()
    baseline: dict[str, Any] = Field(default_factory=dict)
    details: ReportDetails

    # in-memory only (never serialized): holdout row positions, the estimator,
    # and per-row hashes of the source frame for precise mismatch messages
    _holdout_positions: Any = PrivateAttr(default=None)
    _estimator: Any = PrivateAttr(default=None)
    _row_hashes: Any = PrivateAttr(default=None)
    # the frame passed to analyze (a reference, not a copy; still fingerprint-checked on use)
    # and per-session state shared by every revision (how often the test data was reviewed)
    _data: Any = PrivateAttr(default=None)
    _session: Any = PrivateAttr(default=None)

    # ---- reading ---------------------------------------------------------- #
    def decision(self, decision_id: str) -> Decision:
        for d in self.decisions:
            if d.id == decision_id:
                return d
        raise KeyError(decision_id)

    def pending_questions(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.answer is None)

    def decisions_table(self) -> Any:
        """One row per decision, with the decisive evidence resolved (a pandas DataFrame)."""
        from ..render.tables import decisions_table

        return decisions_table(self)

    def cv_estimate(self, family: str | None = None) -> dict[str, Any] | None:
        """The development cross-validation estimate from `verify`, in natural units
        (errors positive): {"metric", "label", "mean", "std", "n_folds", "cv", "family",
        "estimator", "recipe"}. `family` defaults to your estimator ("user"), else the first proxy.
        When `verify` verified changes, the estimate is for the recipe with them (`recipe` lists
        them), scored on the same folds, so it describes the model `export()` builds.
        None before `verify`."""
        from ..rules.metrics import label, natural

        b = self.baseline
        fams = b.get("families") or {}
        if not fams:
            return None
        fam = family or ("user" if "user" in fams else next(iter(fams)))
        if fam not in fams:
            raise KeyError(f"No CV estimate for family {fam!r}; available: {sorted(fams)}")
        f, mt = fams[fam], b["metric"]
        rec = f.get("recipe")
        src = rec or f
        return {"metric": mt, "label": label(mt), "mean": natural(mt, src["mean"]), "std": src["std"],
                "n_folds": b.get("n_folds"), "cv": b.get("cv"), "family": fam, "estimator": f["estimator"],
                "recipe": list(rec["decisions"]) if rec else []}

    # ---- flow ------------------------------------------------------------- #
    def answer(self, answers: dict[str, Any] | None = None, /, **by_keyword: Any) -> Report:
        """New revision with these answers applied; contextual decisions are re-derived.

            report.answer(task="regression", metric="mae")
            report.answer(known_at_prediction={"discount": False}, identifier=["user_id"])
            report.answer({"metric_priority": "typical_error"})      # question ids work too

        Each open question names its keyword (`print(report)` shows the line to run).
        A different `task` re-runs the analysis on the frame kept in memory (see
        `change_task`) and then applies the other answers."""
        from ..api import _apply_answers

        return _apply_answers(self, dict(answers or {}), by_keyword)

    def resolve(self, decision_id: str, note: str) -> Report:
        """New revision closing a review item with your note (status `reviewed`).

        Nothing in the data or the model recipe changes; the note is recorded, shown
        in the report and in `pf.review`, and the item no longer counts as open."""
        from ..api import _resolve

        return _resolve(self, decision_id, note)

    def change_task(self, task: Any, df: Any = None) -> Report:
        """New revision analyzed as `task`, keeping what still applies.

        Uses the frame kept in memory (or `df`, which must be the frame passed to
        `analyze`; checked by fingerprint either way). The holdout keeps the same
        rows. Valid answers, review notes, the registered external holdout and the
        options are carried over; experiments are dropped, since they belong to the
        old task. `change_task(df, task)` (frame first) is accepted too."""
        from ..api import _change_task

        if not isinstance(task, (str, Task)) and isinstance(df, (str, Task)):
            task, df = df, task  # the 0.3 order: change_task(df, task)
        return _change_task(self, df, task)

    def verify(self, df: Any = None, estimator: Any = None, budget: Any = None, groups: Any = None) -> Report:
        """New revision with bounded experiments for the conditional decisions.

        Uses the frame kept in memory, or `df` (the frame passed to `analyze`,
        checked by fingerprint either way); any internal holdout is excluded."""
        from ..api import _verify

        return _verify(self, df, estimator=estimator, budget=budget, groups=groups)

    def holdout(self, df: Any = None) -> tuple[Any, Any]:
        """(dev_df, holdout_df) exactly as reserved by `analyze(..., holdout=...)`.

        Re-derives a partition made before profiling; it never splits anew. Uses the
        frame kept in memory, or `df` (checked by fingerprint)."""
        from ..api import _holdout

        return _holdout(self, df)

    def register_external_holdout(self, holdout_df: Any, dev_df: Any = None, name: str = "external holdout") -> Report:
        """New revision recording an evaluation partition kept out of the analysis.

        Only metadata is stored (rows, columns, digests); the frame is not
        profiled. Exact row overlap with the development frame (kept in memory, or
        `dev_df`) is counted."""
        from ..api import _register_external_holdout

        return _register_external_holdout(self, holdout_df, dev_df=dev_df, name=name)

    def review(self, model: Any, test_df: Any = None, *, segments: Any = None, n_bootstrap: int = 1000,
               seed: int = 0) -> Any:
        """Score a fitted model once on the untouched test data and check the roadmap.

        `test_df` defaults to the internal holdout (`analyze(..., holdout=0.2)`);
        pass your test frame otherwise. Predictions use the model's own feature
        columns. A second review of the same test data in one session warns: a
        test set looked at twice is no longer an unbiased estimate."""
        from ..api import _review_model

        return _review_model(self, model, test_df, segments=segments, n_bootstrap=n_bootstrap, seed=seed)

    def next_steps(self) -> list[str]:
        """The lines to run next, one per open item, ready to copy."""
        from ..render.common import next_lines

        return next_lines(self)

    def show(self, decision_id: str | None = None) -> None:
        """Notebook view: the status card, the first-look table and the figures
        behind the decisions that need you. `show("d.xxx")` shows one decision with
        its evidence and figure. Outside a notebook it prints the same as text."""
        from ..render.notebook import show

        show(self, decision_id)

    def first_look(self) -> Any:
        """One row per column: kind, missing %, distinct values, skew, the
        single-feature score, and the decisions that mention it (a DataFrame)."""
        from ..render.tables import first_look

        return first_look(self)

    def export(self, estimator: Any = None, include_proxy_supported: bool = False, scale: bool = True) -> Any:
        """Model recipe + preprocessor + CV + scoring from eligible decisions only."""
        from ..export import export_pipeline

        return export_pipeline(self, estimator=estimator if estimator is not None else self._estimator,
                               include_proxy_supported=include_proxy_supported, scale=scale)

    def interpret(self, llm: Any) -> Report:
        """New revision with the optional LLM interpretation (see `llm_payload`)."""
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

    def __repr__(self) -> str:
        from ..render.common import headline, next_step

        return f"<prefit Report r{self.details.manifest.revision}: {headline(self)} Next: {next_step(self)}>"

    def _repr_html_(self) -> str:
        from ..render.html import summary_card

        return summary_card(self)
