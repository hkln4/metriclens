# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0, minor versions may
change the public API.

## 0.1.0 — 2026-11-01

First release of the decision-driven design. Replaces the v0 `Roadmap` API.

### Added
- `ml.analyze()` returns a `Report`: Evidence → Finding → Decision chain with
  decision classes (procedural / conditional / contextual / review) and statuses.
- Input guard: misspelled targets and unknown keywords raise; holdout reserved
  before profiling (stratified, group-aware with `groups=`, time-ordered with
  `time_col=`).
- Column-kind inference (numeric, categorical, boolean, datetime, text,
  id_like, constant) with reasons; bootstrap / Wilson intervals on
  threshold-relevant statistics; near-threshold findings marked `uncertain`.
- Task-agnostic leakage hint via a single-feature held-out score; leakage is
  reported as "manual review required" with a question, never as a verdict.
- Rules: task confirmation, validation split, holdout policy, small data,
  temporal order, target missing, duplicates, conflicting duplicates, high
  missingness, near-constant, high correlation, constant columns, unsupported
  columns, id-like columns, leakage hint, fit-in-fold, skewed features, high
  cardinality, class imbalance, target skew, heavy tails, zero inflation;
  metric policy driven by the cost of errors.
- `Report.answer()`, `Report.verify()` (bounded experiments, corrected paired
  t-test, Holm correction, budget; user estimator verifies, proxies give
  sensitivity analysis), `Report.export()` (sklearn Pipeline + CV + scoring from
  eligible decisions only, with an explicit excluded list).
- Single-file HTML report with decision cards, linked evidence, rule versions
  and thresholds, experiments, decision-bound plotly figures with data tables,
  light/dark themes; text, markdown and JSON renders; reproducibility manifest.
- `ml.review()`: reference-based hypothesis checks, baselines, residual /
  confusion / PR / segment analysis, list of never-verified assumptions;
  standalone or embedded HTML.
- Optional LLM layer: allowlisted payload with `llm_payload()` preview, column
  aliasing, schema-constrained output, unknown-id claims dropped, suggestions
  never applied; Anthropic, Gemini and local OpenAI-compatible providers.
- Versioned benchmark (`benchmarks/`), track record cited in reports.
- CLI: `metriclens report data.csv --target y`.
- CI (ruff, mypy, pytest with coverage, HTML smoke test).

### Changed (pre-release review)
- Raw datetime / text / string-id columns are a review item ("unsupported in
  v1, may carry signal"), no longer a procedural drop.
- `groups=` together with `time_col=` raises instead of silently ignoring
  `groups`; the time split states that class stratification is off; grouped
  classification notes that class balance is approximate.
- Verification: the test now follows the split protocol (corrected repeated
  k-fold t-test; single-run grouped k-fold; no p-value for time-series folds),
  and "improved" requires a practical effect threshold
  (`Budget.min_relative_effect` / `min_effect`). Per-fold differences and the
  protocol are shown in the report.
- `export()` returns `model` (full estimator recipe, possibly a
  `TransformedTargetRegressor`) and `preprocessor` instead of `pipeline`;
  estimator-level steps without an estimator go to `needs_estimator`, and an
  estimator without `class_weight` moves that step to `excluded` — nothing
  listed as included is dropped.
- Reports record `manifest.revision` and `manifest.history` for every
  analyze / answer / verify / interpret step.
- Wording: "single-feature predictive leakage hint"; benchmark records are a
  "rule detection pass rate"; the HTML report opens with a data-partition note.

### Changed
- `GroupKFold` is no longer suggested from duplicates or id-like columns.
- Integer targets with more than 10 distinct values are inferred as regression.
- `use_llm=`, `provider=`, `api_key=` replaced by `llm=LLMConfig(...)`.

### Removed
- `Roadmap.make_sklearn()` / `validation_recipe()` (use `Report.export()`).
- Reserved non-tabular modalities from the public enums.
