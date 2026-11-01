# Architecture

metriclens shortens the pre-training analysis of a tabular supervised-learning
project. It measures the data, derives classified decisions from the
measurements, tests the decisions that can be tested, asks the user about the
ones that cannot, and exports only what is safe, confirmed or verified.

## Pipeline

```
ml.split(df, ...) -> dev_df, test_df            (explicit; same splitter as holdout=)
dev_df, target, [estimator, groups, time_col, holdout, context, answers]
   │
   ▼  guard.py ─ validate inputs (misspelled target, single-class target, groups+time…);
   │             optional internal holdout (only with holdout=…), set aside before profiling
   ▼  fingerprint.py ─ SHA-256 digests of the input frame -> manifest (no data)
   ▼  profiling/ ─ dev rows only, seeded sample above ProfileOptions.max_rows
   │     kinds.py   column kind + reason (numeric, categorical, boolean,
   │                datetime, text, id_like, constant)
   │     stats.py   per-kind stats, bootstrap / Wilson intervals,
   │                single-feature held-out association score
   │     tabular.py → DataProfile (aggregates only, JSON-serializable)
   ▼  rules/ ─ deterministic: DataProfile + RuleContext → RuleOutput
   │     catalog.py  versioned rules with thresholds; emit Evidence → Finding → Decision (+ Question)
   │     metrics.py  metric policy: one contextual decision driven by the cost of errors
   ▼  figures.py ─ decision-bound plotly specs (aggregates + color tokens)
   ▼  report/model.py ─ Report (decisions, questions, metric plan, roadmap,
   │                    experiments, baseline) + details (findings, evidence,
   │                    profile, figures, manifest, context, limitations, llm);
   │                    deeply immutable (frozen.py)
   │
   ├─ Report.answer(...)                  re-runs the rules with new answers (no data needed)
   ├─ Report.verify(df)                   fingerprint check, then verify.py on dev rows
   ├─ Report.holdout(df)                  fingerprint check, re-derives the internal holdout
   ├─ Report.register_external_holdout()  metadata + exact-overlap count, never profiled
   ├─ Report.export()                     export.py: model recipe + preprocessor + CV + scoring
   ├─ Report.interpret()  llm/: optional, schema-constrained interpretation
   ├─ Report.to_html()    render/: single-file report
   └─ ml.review(...)      posttrain.py: assumptions vs. actual errors
```

## Contracts

- **Evidence** — `id`, `label`, `value`, optional 95% `ci`, `method`, `columns`,
  linked `figure_ids`.
- **Finding** — produced by one rule; `rule_id`, `rule_version`, `severity`,
  `trigger` (`fired` | `uncertain`), `thresholds`, and at least one evidence id
  (enforced by the model).
- **Decision** — `cls` (procedural / conditional / contextual / review),
  `status`, `stage`, `action`, `rationale`, `risk`, optional `step` (the concrete
  pipeline change), links to findings, evidence, figures, experiments, question.
- **Decision.eligible()** — the single rule for what enters the pipeline:
  procedural+applied, contextual+confirmed, conditional+verified (or
  supported_by_proxy when explicitly requested).

## Why these boundaries

1. **Thresholds are triggers.** A rule compares a statistic *and its interval*
   to a threshold. Interval straddles the threshold → `uncertain`.
2. **Data-modifying actions are never procedural.** Dropping duplicates, rows
   or columns can be a collection artefact or legitimate data; only the user or
   an experiment can decide. The one exception is constant columns, which carry
   no information. Raw datetime/text/string-id columns are left out of the
   export because v1 cannot encode them — a capability limit reported as a
   review item, not a judgement that they are uninformative.
3. **The data cannot choose the metric.** It can say which metrics mislead
   (evidence-backed `avoid`); what to optimize depends on what errors cost.
4. **Experiments are bounded, and their inference matches the split.** One
   change at a time against a fixed base, identical folds, hard budget. The
   test follows the protocol: corrected repeated k-fold t-test (repeated or
   grouped k-fold), no p-value for time-ordered folds. "Improved" needs
   direction, Holm-adjusted significance and a practical effect threshold.
   Proxy models give sensitivity analysis; only the user's estimator verifies.
5. **Split precedence is explicit.** time_col > groups > stratification;
   time_col disables stratification (stated in the decision); groups and
   time_col together raise instead of silently dropping one.
6. **The EDA must not leak into the final estimate.** The holdout is reserved
   before profiling and is never used by rules or experiments.
7. **Reports are immutable, data is fingerprinted.** answer / verify /
   interpret / register_external_holdout return a new report with an
   incremented manifest revision and a history entry; every model is frozen
   with tuples and read-only dicts. Steps that need data again check the
   frame's fingerprint and refuse a different one instead of re-splitting.
8. **The LLM is above the trust boundary.** It sees an allowlisted payload
   (inspectable with `llm_payload()`), must cite existing ids, and can only
   suggest; `answer()` is the only way an answer is applied.

## Adding a rule

See CONTRIBUTING.md. In short: write `fn(profile, ctx, out)` in
`rules/catalog.py` with `@rule(id, version, thresholds)`, use `trigger()` with
the statistic's interval, pick the decision class honestly, add benchmark cases
(must fire / must not fire / near threshold), and bump the manifest version.

## Not in v1

Other modalities, time-series forecasting, clustering, model selection,
multi-feature leakage detection, interaction effects between decisions.
