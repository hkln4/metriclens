"""The v1 rule catalog.

Conventions every rule follows:
- thresholds are triggers, not verdicts: a confidence interval that straddles
  the threshold yields Trigger.UNCERTAIN, never a confident finding;
- data-modifying actions are never PROCEDURAL — they are REVIEW items, or
  CONDITIONAL (testable) / CONTEXTUAL (needs the user) decisions;
- "leakage" is only ever a *hint* that requires manual review;
- GroupKFold-style splitting is only used when the user supplies real groups.
"""

from __future__ import annotations

from ..enums import ColumnKind, DecisionClass, DecisionStatus, Severity, Task, Trigger
from ..profiling.profile import DataProfile
from ..report.model import Step
from .registry import Emitter, RuleContext, abs_ci, contextual, rule, trigger

CLS, REG = Task.CLASSIFICATION, Task.REGRESSION
WILSON = "Wilson 95% interval"
BOOT = "bootstrap 95% percentile interval (seeded)"


# --------------------------------------------------------------------------- #
# Setup & split
# --------------------------------------------------------------------------- #

@rule("task_confirmation", "1", description="Ask the user to confirm an inferred task.")
def task_confirmation(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    if not p.task_inferred:
        return
    te = p.target.task_evidence
    other = REG if p.task is CLS else CLS
    e = out.ev("task.inference", "Task inference", te.get("reason"),
               "dtype + distinct-value heuristic on the target", columns=[p.target.name])
    sev = Severity.WARN if te.get("ambiguous") else Severity.INFO
    f = out.finding("task_inferred", sev, Trigger.FIRED,
                    f"Task inferred as {p.task.value} ({te.get('reason')}).", [e], [p.target.name])
    contextual(out, ctx, qid="task", question=f"Is this a {p.task.value} task?",
               options=[(p.task.value, f"Yes, {p.task.value}", "keeps the current analysis"),
                        (other.value, f"No, {other.value}", f"re-run pf.analyze(..., task='{other.value}')")],
               confirm=p.task.value, decision_id="task", title=f"Treat the task as {p.task.value}",
               stage="split", action=f"Analyze and evaluate as a {p.task.value} problem.",
               rationale=f"Inferred from the target: {te.get('reason')}.",
               risk="A wrong task changes every metric and split recommendation.",
               finding_ids=[f], evidence_ids=[e])


@rule("validation_split", "2", thresholds={"small_data_rows": 1000},
      description="Choose the cross-validation splitter from task, groups and time.")
def validation_split(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    n = p.dataset.n_rows
    ev = [out.ev("dataset.n_rows", "Rows profiled (dev)", n, "row count after holdout/sampling")]
    k = ctx.n_splits
    small = n < 1000
    if ctx.time_col:
        name, params = "TimeSeriesSplit", {"n_splits": k}
        why = f"rows are ordered in time by '{ctx.time_col}'; each fold trains on the past and tests on the future"
        if p.task is CLS:
            why += ("; class stratification is not applied under time ordering, so check that every class "
                    "appears in each test window")
    elif ctx.groups:
        name = "StratifiedGroupKFold" if p.task is CLS else "GroupKFold"
        params = {"n_splits": k}
        if p.task is CLS:
            params.update(shuffle=True, random_state=0)
        why = f"user-supplied groups ({ctx.groups}) keep every group on one side of each split"
        if p.task is CLS:
            why += "; class ratios are preserved only approximately, since whole groups move together"
    elif p.task is CLS:
        min_class = min(p.target.stats.get("class_counts", {"_": k}).values())
        ev.append(out.ev("target.min_class_count", "Smallest class count", min_class, "class counts on dev rows"))
        k = max(2, min(k, min_class))
        name = "RepeatedStratifiedKFold" if small else "StratifiedKFold"
        params = {"n_splits": k, "random_state": 0}
        params.update({"n_repeats": 3} if small else {"shuffle": True})
        why = "stratification keeps class ratios equal across folds"
    else:
        name = "RepeatedKFold" if small else "KFold"
        params = {"n_splits": k, "random_state": 0}
        params.update({"n_repeats": 3} if small else {"shuffle": True})
        why = "standard shuffled k-fold"
    if small and not ctx.time_col and not ctx.groups:
        why += f"; repeated because {n} rows give noisy single-split estimates"
    spec = f"{name}({', '.join(f'{a}={b}' for a, b in params.items())})"
    if ctx.time_col or ctx.groups or name.startswith("Repeated"):
        verif = f"Verification experiments and baselines use the same splitter ({spec})."
    else:
        verif = (f"Verification experiments and baselines use the same splitter repeated "
                 f"{ctx.n_repeats}x (Repeated{name}), for statistical power.")
    out.decision("cv", f"Cross-validate with {name}", DecisionClass.PROCEDURAL, "split",
                 f"Use {spec}. {verif}", why.capitalize() + ".",
                 risk=("Without groups or a time column, rows that belong together (same patient, "
                       "same customer, later in time) can still leak across folds."
                       if not (ctx.groups or ctx.time_col) else ""),
                 evidence_ids=ev, step=Step(kind="cv", params={"name": name, **params}),
                 do=f"Use {spec}.", why=why.capitalize() + ".")
    split_cols = [c for c in (ctx.groups if ctx.groups != "<array>" else None, ctx.time_col) if c]
    if split_cols and not ctx.split_columns_as_features:
        e = out.ev("columns.split", "Split columns", split_cols, "groups= / time_col= arguments", columns=split_cols)
        out.decision("split_columns", f"Keep split column(s) {split_cols} out of the features",
                     DecisionClass.PROCEDURAL, "columns",
                     f"{split_cols} define the split and are not model features.",
                     "A column used to define folds encodes which fold a row is in (group identity, position in "
                     "time); as a feature it lets the model exploit the split itself.",
                     risk="If one is also a genuine predictor, pass split_columns_as_features=True and review the "
                          "warning that follows.", evidence_ids=[e],
                     why="A split column as a feature lets the model exploit the split itself.",
                     do=f"Exclude {split_cols} from the features.")
    elif split_cols:
        e = out.ev("columns.split", "Split columns", split_cols, "groups= / time_col= arguments", columns=split_cols)
        f = out.finding("split_columns_as_features", Severity.WARN, Trigger.FIRED,
                        f"{split_cols} are used both to define the split and as features "
                        "(split_columns_as_features=True).", [e], split_cols)
        out.decision("split_columns", "Split column(s) also used as features", DecisionClass.REVIEW, "columns",
                     "Check that the model is not exploiting the split: compare scores with and without these "
                     "columns.",
                     "You asked for split columns to be features. A time column used both ways lets the model "
                     "learn the fold boundaries; a group id lets it memorize groups seen in training.",
                     finding_ids=[f], evidence_ids=[e])


@rule("holdout_policy", "2", description="Keep the holdout out of every decision.")
def holdout_policy(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    h = p.holdout
    ext = ctx.external_holdout
    if h.get("used"):
        e = out.ev("holdout", "Internal holdout", f"{h['n_holdout']} rows ({h['method']})",
                   "reserved before profiling; never read by rules or experiments")
        out.decision("holdout", "Evaluate on the holdout once, at the end", DecisionClass.PROCEDURAL,
                     "final", "Use report.holdout(df) to get the holdout; score the final model on it once.",
                     "Every prefit decision was made on the dev rows only, so the holdout is an "
                     "unbiased check of the final model.",
                     risk="Re-using the holdout to choose between models turns it into a second dev set.",
                     evidence_ids=[e], do="Score the final model on report.holdout(df) once, at the end.")
    elif ext:
        e = out.ev("holdout", "External holdout (registered)", f"{ext['n_rows']} rows ({ext['name']})",
                   "registered with report.register_external_holdout; metadata only, never profiled")
        out.decision("holdout", f"Evaluate on the {ext['name']} once, at the end", DecisionClass.PROCEDURAL,
                     "final", f"Score the final model on the {ext['name']} once.",
                     "Prefit did not reserve an internal holdout; the analysis ran on the frame you passed, "
                     f"and the registered {ext['name']} ({ext['n_rows']} rows) was kept out of it.",
                     risk="Re-using it to choose between models turns it into a second dev set.", evidence_ids=[e])
    else:
        e = out.ev("holdout", "Internal holdout", "none", h.get("reason", ""))
        out.decision("holdout", "Prefit did not reserve an internal holdout", DecisionClass.PROCEDURAL,
                     "final",
                     "Keep a test set out of this analysis yourself (pf.split before pf.analyze) and register it with "
                     "report.register_external_holdout(test_df); without one, estimate final performance with nested "
                     "cross-validation.",
                     f"Prefit did not reserve an internal holdout ({h.get('reason', '')}). If you split off a test "
                     "set before calling analyze, register it so the report records it.",
                     risk="Scores from the same CV used to make decisions are optimistic.", evidence_ids=[e],
                     do="Register your test set, or use nested CV for the final estimate.")


@rule("external_holdout_overlap", "1", thresholds={"overlap_rows": 1})
def external_holdout_overlap(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    ext = ctx.external_holdout
    if not ext or not ext.get("overlap_rows"):
        return
    k = ext["overlap_rows"]
    e = out.ev("holdout.overlap", f"Rows of the {ext['name']} also present in development", k,
               f"exact row match on {ext.get('overlap_columns_compared')} shared columns (row hashes)")
    f = out.finding("external_holdout_overlap", Severity.WARN, Trigger.FIRED,
                    f"Exact row overlap detected between development and {ext['name']} ({k} rows). This may inflate "
                    "evaluation; inspect whether the duplicated observations are legitimate repeats.", [e])
    out.decision("external_holdout_overlap", f"Inspect {k} row(s) shared with the {ext['name']}", DecisionClass.REVIEW,
                 "final",
                 "Find out whether the shared rows are the same observation recorded twice (remove them from one "
                 "side) or legitimate repeats (keep them, and expect a slightly optimistic score).",
                 f"Exact row overlap detected between development and {ext['name']} ({k} rows). This may inflate "
                 "evaluation.", finding_ids=[f], evidence_ids=[e],
                 why=f"{k} row(s) appear in both development and the {ext['name']}.",
                 do="Inspect whether the duplicated observations are legitimate repeats.")


@rule("small_data", "1", thresholds={"warn_rows": 1000, "critical_rows": 200})
def small_data(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    n = p.dataset.n_rows
    if n >= 1000:
        return
    e = out.ev("dataset.n_rows", "Rows profiled (dev)", n, "row count after holdout/sampling")
    sev = Severity.CRITICAL if n < 200 else Severity.WARN
    f = out.finding("small_data", sev, Trigger.FIRED,
                    f"Only {n} development rows: fold-to-fold variance will be large.", [e])
    out.decision("small_data", "Treat small score differences as noise", DecisionClass.REVIEW, "split",
                 "Report the spread across folds and compare models only when the gap exceeds it.",
                 f"With {n} rows a single split's score swings with the random seed.",
                 finding_ids=[f], evidence_ids=[e], why=f"Only {n} development rows: fold-to-fold noise is large.", do="Compare models only when the gap exceeds the fold spread.")


@rule("temporal_order", "1", description="Datetime columns exist but no time_col was given.")
def temporal_order(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    dts = p.kind_of(ColumnKind.DATETIME)
    if not dts or ctx.time_col:
        return
    e = out.ev("columns.datetime", "Datetime columns", dts, "column-kind inference", columns=dts)
    f = out.finding("datetime_columns", Severity.WARN, Trigger.FIRED,
                    f"{len(dts)} datetime column(s) {dts} but no time_col: random CV assumes rows are exchangeable in time.",
                    [e], dts)
    contextual(out, ctx, qid="temporal", question=(
        f"Will the model predict future rows from past ones (time order matters for {dts[0]})?"),
        options=[("yes", "Yes, predict the future", f"re-run pf.analyze(..., time_col='{dts[0]}') for a time-ordered holdout and TimeSeriesSplit"),
                 ("no", "No, rows are exchangeable", "keep random cross-validation")],
        confirm="yes", decision_id="temporal", title="Split by time", stage="split",
        action=f"Re-run with time_col='{dts[0]}' so the holdout is the latest period and CV uses TimeSeriesSplit.",
        rationale="Random splits let a model train on the future and test on the past, which flatters every metric.",
        risk="Random CV on time-ordered data is a common source of over-optimistic scores.",
        finding_ids=[f], evidence_ids=[e])


# --------------------------------------------------------------------------- #
# Data review
# --------------------------------------------------------------------------- #

@rule("target_missing", "1", thresholds={"missing_fraction": 0.0})
def target_missing(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    t = p.target
    if not t.n_missing:
        return
    e = out.ev("target.missing", "Target missing rows", t.n_missing,
               f"count of missing target values ({t.missing_fraction:.1%})", columns=[t.name])
    f = out.finding("target_missing", Severity.WARN, Trigger.FIRED,
                    f"{t.n_missing} rows ({t.missing_fraction:.1%}) have no target and cannot be used for supervised training.",
                    [e], [t.name])
    out.decision("target_missing", "Check why the target is missing", DecisionClass.REVIEW, "data_review",
                 "Rows without a target are excluded from training and evaluation; check whether they differ "
                 "systematically from labelled rows.",
                 "If labels are missing for a reason (e.g. outcome not yet observed), the labelled rows are a "
                 "biased sample and metrics will not transfer.",
                 risk="Selection bias: performance on labelled rows may not match production.",
                 finding_ids=[f], evidence_ids=[e], why=f"{t.n_missing} rows ({t.missing_fraction:.1%}) have no target and are excluded.", do="Check whether unlabelled rows differ systematically from labelled ones.")


@rule("infinite_values", "1", thresholds={"inf_count": 1})
def infinite_values(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols = {c: cp.stats.get("inf_count", 0) for c, cp in p.columns.items() if cp.stats.get("inf_count")}
    t_inf = p.target.stats.get("inf_count", 0) if p.task is REG else 0
    if not cols and not t_inf:
        return
    evs = [out.ev(f"col.{c}.inf", f"Infinite values · {c}", k, "count of +inf / -inf", columns=[c])
           for c, k in cols.items()]
    if t_inf:
        evs.append(out.ev("target.inf", "Infinite target values", t_inf, "count of +inf / -inf",
                          columns=[p.target.name]))
    names = list(cols) + ([p.target.name] if t_inf else [])
    f = out.finding("infinite_values", Severity.WARN, Trigger.FIRED,
                    f"{len(names)} column(s) contain +/-inf values: {names}. Distribution statistics exclude them.",
                    evs, names)
    out.decision("infinite_values", "Decide what the infinite values mean", DecisionClass.REVIEW, "data_review",
                 "Find out where the infinities come from (often a division by zero) and replace them with a "
                 "meaningful value or a missing marker before modelling.",
                 "Infinite values break most estimators; the exported preprocessing treats them as missing (imputed), "
                 "which may hide a real data problem.",
                 finding_ids=[f], evidence_ids=evs,
                 why=f"{len(names)} column(s) contain +/-inf; export treats them as missing.",
                 do="Trace the source of the infinities and fix them upstream.")


@rule("duplicate_rows", "1", thresholds={"warn_fraction": 0.01})
def duplicate_rows(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    d = p.dataset
    if not d.duplicate_rows:
        return
    e = out.ev("dataset.duplicates", "Exact duplicate rows", d.duplicate_rows,
               f"exact full-row duplicates incl. target ({d.duplicate_fraction:.2%})")
    sev = Severity.WARN if d.duplicate_fraction >= 0.01 else Severity.INFO
    f = out.finding("duplicate_rows", sev, Trigger.FIRED,
                    f"{d.duplicate_rows} exact duplicate rows ({d.duplicate_fraction:.2%}).", [e])
    hint = ("" if ctx.groups else " If you have an entity id, pass it as groups= so copies stay on one side of each split.")
    out.decision("duplicate_rows", "Review duplicates before splitting", DecisionClass.REVIEW, "data_review",
                 "Find out whether duplicates are the same real-world record (collection artefact) or legitimate "
                 "repeats." + hint,
                 "Copies on both sides of a split leak the answer and inflate validation scores.",
                 risk="Dropping legitimate repeats changes the data distribution; keeping artefacts inflates scores. "
                      "prefit does not decide this for you.",
                 finding_ids=[f], evidence_ids=[e], why=f"{d.duplicate_rows} exact duplicate rows ({d.duplicate_fraction:.2%}) can land on both sides of a split.", do="Decide whether they are the same record or legitimate repeats.")


@rule("conflicting_duplicates", "1", thresholds={"groups": 1})
def conflicting_duplicates(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    k = p.dataset.conflicting_duplicate_groups
    if not k:
        return
    e = out.ev("dataset.conflicting", "Identical features, different target", k,
               "groups of rows with identical feature values but more than one target value")
    f = out.finding("conflicting_duplicates", Severity.WARN, Trigger.FIRED,
                    f"{k} group(s) of rows share all feature values but disagree on the target.", [e])
    out.decision("conflicting_duplicates", "Check for label noise or missing features", DecisionClass.REVIEW,
                 "data_review", "Inspect these groups: they indicate label noise, or that the features miss information.",
                 "No model can separate rows with identical inputs, so these set a ceiling on achievable scores.",
                 finding_ids=[f], evidence_ids=[e])


@rule("high_missingness", "1", thresholds={"warn_fraction": 0.2, "drop_candidate_fraction": 0.5})
def high_missingness(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    fired, uncertain, drop = [], [], []
    for c, cp in p.columns.items():
        if cp.kind is ColumnKind.CONSTANT:
            continue
        t = trigger(cp.missing_fraction, cp.missing_ci, 0.2)
        if t is Trigger.FIRED:
            fired.append(c)
        elif t is Trigger.UNCERTAIN:
            uncertain.append(c)
        if trigger(cp.missing_fraction, cp.missing_ci, 0.5) is Trigger.FIRED:
            drop.append(c)
    cols = fired + uncertain
    if not cols:
        return
    evs = [out.ev(f"col.{c}.missing", f"Missing fraction · {c}", p.columns[c].missing_fraction, WILSON,
                  ci=p.columns[c].missing_ci, columns=[c]) for c in cols]
    trig = Trigger.FIRED if fired else Trigger.UNCERTAIN
    sev = Severity.WARN
    msg = f"{len(fired)} column(s) are >= 20% missing"
    if uncertain:
        msg += f"; {len(uncertain)} more are near the threshold (interval straddles 20%)"
    f = out.finding("high_missingness", sev, trig, msg + ".", evs, cols)
    out.decision("missingness_review", "Check whether missingness is informative", DecisionClass.REVIEW,
                 "data_review",
                 "Ask how these values went missing. Missing-not-at-random values carry signal (kept via the "
                 "missing-indicator columns) and may differ in production.",
                 "Imputation assumes the missing values resemble the observed ones; that is often false.",
                 risk="If missingness is caused by the target (e.g. a test only ordered for sick patients), the "
                      "indicator itself can be a leak.",
                 finding_ids=[f], evidence_ids=evs, why=f"{len(cols)} column(s) are >= 20% missing; missingness may itself be informative.", do="Ask how the values went missing.")
    if drop:
        dev = [e for e, c in zip(evs, cols) if c in drop]
        out.decision("drop_mostly_missing", f"Drop {len(drop)} mostly-missing column(s)", DecisionClass.CONDITIONAL,
                     "preprocessing", f"Drop {drop} instead of imputing them.",
                     "Columns that are mostly imputed add noise; but their missing indicator can be informative, "
                     "so only an experiment can tell.",
                     risk="Dropping discards any signal in the missingness pattern.",
                     verify_hint="Compare CV scores with and without these columns.",
                     finding_ids=[f], evidence_ids=dev, step=Step(kind="drop_columns", params={"columns": drop}),
                     why=f"{len(drop)} column(s) are >= 50% missing; their missing indicator may still be informative.",
                     do="Test dropping them with your estimator (report.verify).",
                     trig=Trigger.FIRED)


@rule("near_constant", "1", thresholds={"dominant_fraction": 0.99})
def near_constant(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols = [c for c, cp in p.columns.items()
            if cp.kind in (ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN)
            and (cp.stats.get("dominant_fraction") or 0) >= 0.99]
    if not cols:
        return
    evs = [out.ev(f"col.{c}.dominant", f"Dominant value share · {c}", p.columns[c].stats["dominant_fraction"],
                  "share of the most frequent value among non-missing rows", columns=[c]) for c in cols]
    f = out.finding("near_constant", Severity.INFO, Trigger.FIRED,
                    f"{len(cols)} column(s) have one value in >= 99% of rows.", evs, cols)
    out.decision("near_constant", "Check near-constant columns", DecisionClass.REVIEW, "data_review",
                 "Keep them unless they are artefacts: a rare value can be a strong signal, or a leak.",
                 "With so few non-dominant rows, a model can memorize them.",
                 finding_ids=[f], evidence_ids=evs)


@rule("high_correlation", "1", thresholds={"abs_spearman": 0.95})
def high_correlation(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    pairs = p.relations.high_corr_pairs
    if not pairs:
        return
    e = out.ev("relations.high_corr", "Feature pairs with |Spearman| >= 0.95", len(pairs),
               p.relations.corr_method, columns=sorted({x for pr in pairs for x in (pr["a"], pr["b"])}))
    f = out.finding("high_correlation", Severity.INFO, Trigger.FIRED,
                    f"{len(pairs)} feature pair(s) are nearly collinear (|rho| >= 0.95).", [e])
    out.decision("high_correlation", "Expect unstable coefficients / importances", DecisionClass.REVIEW,
                 "data_review", "Do not interpret individual coefficients or importances of these pairs; "
                 "consider keeping one per pair if interpretability matters.",
                 "Collinear features split credit arbitrarily between them.", finding_ids=[f], evidence_ids=[e])


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #

@rule("constant_columns", "1")
def constant_columns(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols = p.kind_of(ColumnKind.CONSTANT)
    if not cols:
        return
    e = out.ev("columns.constant", "Constant columns", cols, "at most one distinct non-missing value", columns=cols)
    f = out.finding("constant_columns", Severity.INFO, Trigger.FIRED,
                    f"{len(cols)} column(s) have a single value and carry no information: {cols}.", [e], cols)
    out.decision("drop_constant", f"Drop {len(cols)} constant column(s)", DecisionClass.PROCEDURAL, "columns",
                 f"Exclude {cols} from the features.",
                 "A column with one value cannot inform any model; dropping it changes no prediction.",
                 finding_ids=[f], evidence_ids=[e], step=Step(kind="drop_columns", params={"columns": cols}))


@rule("unsupported_columns", "2")
def unsupported_columns(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols = [c for c, cp in p.columns.items()
            if cp.kind in (ColumnKind.DATETIME, ColumnKind.TEXT)
            or (cp.kind is ColumnKind.ID_LIKE and not cp.dtype.startswith(("int", "uint", "Int", "UInt")))]
    if not cols:
        return
    kinds = {c: p.columns[c].kind.value for c in cols}
    e = out.ev("columns.unsupported", "Columns not usable raw in v1", kinds,
               "column-kind inference (datetime, free text, string identifiers)", columns=cols)
    f = out.finding("unsupported_columns", Severity.WARN, Trigger.FIRED,
                    f"{len(cols)} column(s) are not supported raw by the v1 tabular pipeline: {kinds}. "
                    "They may carry strong signal.", [e], cols)
    out.decision("unsupported_columns", f"Engineer features from {len(cols)} datetime/text/id column(s)",
                 DecisionClass.REVIEW, "columns",
                 f"{cols} are not in the exported feature matrix because v1 has no encoder for raw datetimes, "
                 "free text or string identifiers. If they matter, derive features yourself (date parts, elapsed "
                 "time, text vectors, entity statistics) and re-run.",
                 "Unsupported in v1 is a capability limit of prefit, not a judgement that these columns are "
                 "uninformative; dates and text are often among the strongest signals.",
                 risk="Leaving them out can cost real signal. If one identifies an entity with repeated rows, pass it "
                      "as groups=; if one orders rows in time, pass it as time_col=.",
                 finding_ids=[f], evidence_ids=[e])


@rule("id_like_columns", "1", thresholds={"unique_ratio": 0.95})
def id_like_columns(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    for c, cp in p.columns.items():
        if cp.kind is not ColumnKind.ID_LIKE or not cp.dtype.startswith(("int", "uint", "Int", "UInt")):
            continue
        e = out.ev(f"col.{c}.unique_ratio", f"Unique ratio · {c}", cp.unique_ratio,
                   f"distinct / non-missing values; {cp.kind_reason}", columns=[c])
        f = out.finding(f"id_like.{c}", Severity.WARN, Trigger.FIRED,
                        f"'{c}' looks like an identifier ({cp.kind_reason}). This is a review item, not a leakage verdict.",
                        [e], [c])
        contextual(out, ctx, qid=f"id_like.{c}", question=f"Is '{c}' an identifier rather than a measurement?",
                   options=[("identifier", "Yes, an identifier", "drop it from the features"),
                            ("feature", "No, a real feature", "keep it")],
                   confirm="identifier", decision_id=f"drop_id.{c}", title=f"Drop identifier '{c}'",
                   stage="columns", action=f"Exclude '{c}' from the features.",
                   rationale="Row identifiers let models memorize rows; their order can also encode time or the target.",
                   risk="If it identifies an entity with repeated rows, pass it as groups= rather than just dropping it.",
                   finding_ids=[f], evidence_ids=[e], step=Step(kind="drop_columns", params={"columns": [c]}), why=f"'{c}' is near-unique ({cp.unique_ratio:.0%} distinct) and looks like an identifier.", do=f"Answer whether '{c}' is an identifier.")


@rule("leakage_hint", "2", thresholds={"association": 0.95, "uncertain_association": 0.90, "abs_pearson": 0.98})
def leakage_hint(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    for c, cp in p.columns.items():
        a = cp.association
        pr = p.relations.pearson_target.get(c)
        strong = (a is not None and a >= 0.95) or (pr is not None and pr >= 0.98)
        weak = a is not None and 0.90 <= a < 0.95
        if not (strong or weak):
            continue
        evs = []
        if a is not None:
            evs.append(out.ev(f"col.{c}.association", f"Single-feature 2-fold CV score · {c}", a,
                              p.relations.association_method, columns=[c, p.target.name]))
        if pr is not None and pr >= 0.98:
            evs.append(out.ev(f"col.{c}.pearson", f"|Pearson r| with target · {c}", pr,
                              "Pearson correlation on dev rows", columns=[c, p.target.name]))
        if weak and not strong:
            out.finding(f"leakage_hint.{c}", Severity.INFO, Trigger.UNCERTAIN,
                        f"'{c}' alone predicts the target unusually well (score {a:.3f}); below the leakage-hint "
                        "threshold but worth a look.", evs, [c])
            continue
        sev = Severity.CRITICAL if (a or 0) >= 0.99 or (pr or 0) >= 0.995 else Severity.WARN
        f = out.finding(f"leakage_hint.{c}", sev, Trigger.FIRED,
                        f"Single-feature predictive leakage hint — manual review required: '{c}' alone predicts "
                        f"the target (2-fold CV single-feature score {a if a is not None else float('nan'):.3f}). High predictive "
                        "power is not proof of leakage; genuinely strong features look the same.", evs, [c])
        contextual(out, ctx, qid=f"leakage.{c}",
                   question=f"Is '{c}' known at prediction time, before the target is observed?",
                   options=[("no", "No, it is only known afterwards", "drop it from the features"),
                            ("yes", "Yes, it is available in production", "keep it; the signal is genuine")],
                   confirm="no", decision_id=f"drop_leak.{c}", title=f"Drop '{c}' (possible leak)",
                   stage="columns", action=f"Exclude '{c}' from the features.",
                   rationale="A feature that is a consequence or copy of the target makes every metric meaningless.",
                   risk="Some features are legitimately this strong; dropping one throws away real signal.",
                   verify_hint="Trace how and when the column is recorded relative to the target.",
                   finding_ids=[f], evidence_ids=evs, step=Step(kind="drop_columns", params={"columns": [c]}), why=f"'{c}' alone predicts the target with a 2-fold CV score of {a if a is not None else float('nan'):.3f}.", do=f"Answer whether '{c}' is known at prediction time.")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

@rule("fit_in_fold", "2")
def fit_in_fold(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    miss = p.dataset.missing_columns
    cat = p.kind_of(ColumnKind.CATEGORICAL)
    num = p.kind_of(ColumnKind.NUMERIC) + p.kind_of(ColumnKind.BOOLEAN)
    evs = [out.ev("columns.kinds", "Column kinds", p.dataset.kind_counts, "column-kind inference")]
    if miss:
        evs.append(out.ev("columns.missing", "Columns with missing values", len(miss), "count of columns with any missing value",
                          columns=miss))
    parts = []
    if num:
        parts.append(f"{len(num)} numeric/boolean: median imputation" + (" + missing indicator" if miss else ""))
    if cat:
        parts.append(f"{len(cat)} categorical: 'missing' category + one-hot (categories under 1% grouped)")
    out.decision("fit_in_fold", "Fit all preprocessing inside each fold", DecisionClass.PROCEDURAL, "preprocessing",
                 "Put every preprocessing step in one sklearn Pipeline so it is re-fit on each training fold"
                 + (f" (defaults: {'; '.join(parts)})." if parts else "."),
                 "Statistics computed on the full data (medians, category lists, encoders) leak test-fold "
                 "information into the score.",
                 risk="The imputation and encoding defaults are generic; replace them where domain knowledge says "
                      "otherwise. Feature scaling is estimator-dependent and is not part of this decision: export "
                      "adds it with scale=True (default) and you can turn it off.",
                 evidence_ids=evs, why="Statistics fit on all rows leak test-fold information into the score.",
                 do="Keep imputation and encoding inside the exported Pipeline.")


@rule("skewed_features", "2", thresholds={"abs_skew": 2.0})
def skewed_features(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols, trig_any = [], []
    for c, cp in p.columns.items():
        if cp.kind is not ColumnKind.NUMERIC or cp.stats.get("skew") is None:
            continue
        t = trigger(abs(cp.stats["skew"]), abs_ci(cp.ci.get("skew")), 2.0)
        if t:
            cols.append(c)
            trig_any.append(t)
    if not cols:
        return
    evs = [out.ev(f"col.{c}.skew", f"Skewness · {c}", p.columns[c].stats["skew"], BOOT,
                  ci=p.columns[c].ci.get("skew"), columns=[c]) for c in cols]
    trig = Trigger.FIRED if Trigger.FIRED in trig_any else Trigger.UNCERTAIN
    n_fired = sum(t is Trigger.FIRED for t in trig_any)
    n_near = len(cols) - n_fired
    counts = f"{n_fired} skewed" + (f" + {n_near} near-threshold" if n_near else "")
    f = out.finding("skewed_features", Severity.INFO, trig,
                    f"{counts} numeric feature(s) (|skew| >= 2; near-threshold = interval straddles 2).", evs, cols)
    out.decision("power_transform", f"Power-transform {counts} feature(s)", DecisionClass.CONDITIONAL,
                 "preprocessing", f"Apply a Yeo-Johnson transform to {cols[:8]}{' ...' if len(cols) > 8 else ''}.",
                 "Skewed inputs can hurt linear and distance-based models; tree ensembles are usually far less "
                 "sensitive. The effect is model-dependent, so it is tested, not assumed.",
                 risk="Transforms make coefficients harder to interpret and can amplify noise in the tails.",
                 verify_hint="Compare CV scores with and without the transform for your model family.",
                 finding_ids=[f], evidence_ids=evs, trig=trig,
                 step=Step(kind="power_transform", params={"columns": cols}),
                 why=f"{counts} feature(s) have |skew| >= 2; whether that matters depends on the model family.",
                 do="Test a Yeo-Johnson transform with your estimator (report.verify).")


@rule("high_cardinality", "1", thresholds={"cardinality": 50})
def high_cardinality(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    cols = [c for c, cp in p.columns.items() if cp.kind is ColumnKind.CATEGORICAL and cp.cardinality > 50]
    if not cols:
        return
    evs = [out.ev(f"col.{c}.cardinality", f"Distinct categories · {c}", p.columns[c].cardinality,
                  "distinct non-missing values", columns=[c]) for c in cols]
    f = out.finding("high_cardinality", Severity.INFO, Trigger.FIRED,
                    f"{len(cols)} categorical column(s) have more than 50 categories.", evs, cols)
    out.decision("target_encode", f"Target-encode {len(cols)} high-cardinality column(s)", DecisionClass.CONDITIONAL,
                 "preprocessing", f"Use sklearn TargetEncoder (internally cross-fitted) for {cols}.",
                 "One-hot with rare categories grouped loses information on long tails; target encoding keeps it "
                 "but can overfit. Only an experiment can tell which wins here.",
                 risk="Target encoding fit outside the fold is a leak; TargetEncoder inside the Pipeline avoids that.",
                 finding_ids=[f], evidence_ids=evs, trig=Trigger.FIRED,
                 why=f"{len(cols)} categorical column(s) have more than 50 categories.",
                 do="Test target encoding against one-hot with your estimator (report.verify).",
                 step=Step(kind="target_encode", params={"columns": cols}))


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #

@rule("class_imbalance", "1", thresholds={"imbalance_ratio": 3.0, "critical_ratio": 10.0}, tasks=(CLS,))
def class_imbalance(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    s = p.target.stats
    ratio = s.get("imbalance_ratio")
    t = trigger(ratio, p.target.ci.get("imbalance_ratio"), 3.0)
    if t is None or ratio is None:
        return
    e1 = out.ev("target.imbalance", "Imbalance ratio (largest / smallest class)", ratio, BOOT,
                ci=p.target.ci.get("imbalance_ratio"), columns=[p.target.name])
    e2 = out.ev("target.majority", "Majority-class share (= accuracy of always predicting it)",
                s.get("majority_fraction"), "class counts on dev rows", columns=[p.target.name])
    sev = Severity.CRITICAL if ratio >= 10 else Severity.WARN
    f = out.finding("class_imbalance", sev, t,
                    f"Classes are imbalanced {ratio:.1f}:1; always predicting the majority class already scores "
                    f"{s.get('majority_fraction', 0):.1%} accuracy.", [e1, e2], [p.target.name])
    if ctx.has_estimator and "class_weight" not in ctx.estimator_params:
        status, note = DecisionStatus.NOT_APPLICABLE, "Your estimator has no class_weight parameter."
    else:
        status, note = None, ""
    out.decision("class_weight", "Weight classes inversely to frequency", DecisionClass.CONDITIONAL, "target",
                 "Set class_weight='balanced' on the estimator.",
                 "Re-weighting can raise minority-class recall, at the cost of more false positives and "
                 "distorted probabilities.",
                 risk="Probabilities are no longer calibrated; re-calibrate if you need them.",
                 verify_hint="Compare the chosen metric with and without class weights.",
                 finding_ids=[f], evidence_ids=[e1, e2], trig=t, status=status, note=note,
                 why=f"Classes are {ratio:.1f}:1; re-weighting may raise minority recall at the cost of precision.",
                 do="Test class_weight='balanced' with your estimator (report.verify).",
                 step=Step(kind="class_weight", params={"value": "balanced"}))


@rule("target_skew", "1", thresholds={"abs_skew": 1.0, "outlier_fraction": 0.05}, tasks=(REG,))
def target_skew(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    s, ci = p.target.stats, p.target.ci
    skew = s.get("skew")
    if skew is None:
        return
    t1 = trigger(abs(skew), abs_ci(ci.get("skew")), 1.0)
    t2 = trigger(s.get("outlier_fraction"), ci.get("outlier_fraction"), 0.05)
    if not (t1 or t2):
        return
    t = Trigger.FIRED if Trigger.FIRED in (t1, t2) else Trigger.UNCERTAIN
    e1 = out.ev("target.skew", "Target skewness", skew, BOOT, ci=ci.get("skew"), columns=[p.target.name])
    e2 = out.ev("target.outliers", "Target IQR-outlier fraction", s.get("outlier_fraction"),
                "share outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]; " + BOOT, ci=ci.get("outlier_fraction"),
                columns=[p.target.name])
    sev = Severity.WARN if abs(skew) >= 2 or (s.get("outlier_fraction") or 0) >= 0.05 else Severity.INFO
    f = out.finding("target_skew", sev, t,
                    f"Target is skewed (skew={skew:.2f}, {s.get('outlier_fraction', 0):.1%} IQR-outliers): "
                    "squared-error metrics will be driven by the largest values.", [e1, e2], [p.target.name])
    neg = s.get("negative_fraction") or 0
    if neg > 0:
        e3 = out.ev("target.negative", "Negative target share", neg, "share of target values < 0", columns=[p.target.name])
        out.decision("log_target", "No log transform: target has negative values", DecisionClass.REVIEW, "target",
                     "A log transform is undefined here; if a multiplicative scale makes sense, consider a signed "
                     "transform (e.g. arcsinh) and test it yourself.",
                     f"{neg:.1%} of target values are negative.", finding_ids=[f], evidence_ids=[e1, e3], why=f"{neg:.1%} of target values are negative, so log(1 + y) is undefined.", do="Consider a signed transform (arcsinh) and test it yourself.")
        return
    out.decision("log_target", "Model log(1 + target)", DecisionClass.CONDITIONAL, "target",
                 "Wrap the estimator in TransformedTargetRegressor(func=log1p, inverse_func=expm1).",
                 "On a skewed, non-negative target, fitting on the log scale often reduces absolute error; it also "
                 "changes what the model optimizes (closer to relative error).",
                 risk="Predictions back-transformed from the log scale are biased low for the mean; RMSE can get worse "
                      "while MAE improves.",
                 verify_hint="Compare MAE and RMSE with and without the transform.",
                 finding_ids=[f], evidence_ids=[e1, e2], trig=t, step=Step(kind="log_target"),
                 why=f"The target is skewed (skew {skew:.2f}, {s.get('outlier_fraction', 0):.1%} IQR-outliers) and "
                     "non-negative.",
                 do="Test log(1 + y) with your estimator (report.verify).")


@rule("heavy_tailed_target", "1", thresholds={"excess_kurtosis": 3.0}, tasks=(REG,))
def heavy_tailed_target(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    k = p.target.stats.get("kurtosis")
    t = trigger(k, p.target.ci.get("kurtosis"), 3.0)
    if t is None:
        return
    e = out.ev("target.kurtosis", "Target excess kurtosis", k, BOOT, ci=p.target.ci.get("kurtosis"),
               columns=[p.target.name])
    out.finding("heavy_tailed_target", Severity.WARN, t,
                f"Target is heavy-tailed (excess kurtosis={k:.1f}): a few extreme values can swing RMSE.", [e],
                [p.target.name])


@rule("zero_inflation", "1", thresholds={"zero_fraction": 0.3}, tasks=(REG,))
def zero_inflation(p: DataProfile, ctx: RuleContext, out: Emitter) -> None:
    z = p.target.stats.get("zero_fraction")
    t = trigger(z, p.target.ci.get("zero_fraction"), 0.3)
    if t is None:
        return
    e = out.ev("target.zeros", "Target zero share", z, WILSON, ci=p.target.ci.get("zero_fraction"),
               columns=[p.target.name])
    f = out.finding("zero_inflation", Severity.WARN, t,
                    f"{z:.0%} of target values are exactly zero (zero-inflated).", [e], [p.target.name])
    out.decision("zero_inflation", "Consider a two-part or Tweedie model", DecisionClass.REVIEW, "target",
                 "Evaluate zero-vs-positive and the size of positives separately, or try a Tweedie/Poisson loss. "
                 "prefit does not test model families.",
                 "A single regressor tends to predict small positive values everywhere on zero-inflated targets.",
                 finding_ids=[f], evidence_ids=[e])
