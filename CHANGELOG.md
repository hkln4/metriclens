# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0, minor versions may
change the public API.

## 0.2.0 — 2026-11-03

Exact duplicate rows no longer leak across splits. This changes which rows land
in the holdout and in each CV fold whenever the data has exact duplicates, so
scores on such data move (down, toward the honest value). Data without exact
duplicates splits exactly as in 0.1.0 for the same seed.

### Changed
- `pf.split` (`"stratified"`, `"random"`) and `analyze(holdout=...)` treat
  identical rows (all columns, target included) as one unit and keep them on
  one side; class ratios and the holdout fraction are kept approximately.
  `profile.holdout` records `duplicate_groups_kept_together` and the method says
  "exact duplicates kept on one side"; `report.holdout(df)` re-derives the same
  positions. With `groups=` the user's groups take precedence and a warning
  names duplicates that span groups; `time_col=` is unchanged.
- Rule `validation_split` v3: with exact duplicates in the dev rows (and no
  groups / time_col), the CV step carries `duplicate_aware=True`; `verify` and
  `export` build a `DuplicateAwareKFold` instead of the plain k-fold. The
  decision title says "(exact duplicates kept in one fold)" and `describe()`
  adds a line. Repeated k-fold and the corrected resampled t-test still apply.
- Rule `duplicate_rows` v2: one procedural decision, "Keep exact duplicates on
  one side of every split" (applied; no row is removed), and one review item,
  "Decide whether duplicates are the same record or legitimate repeats".
- The data-partition note says when duplicates were kept together.
- The external-holdout overlap warning points to `pf.split`.

### Added
- `pf.DuplicateAwareKFold(n_splits, n_repeats, stratify, random_state)`: a
  scikit-learn-compatible repeated (stratified) k-fold over sets of identical
  rows. It hashes features + target itself, so no `groups=` is needed.
- Benchmark manifest `2026.11-3`: case `duplicates_split_overlap` checks that
  no exact duplicate lands on both sides of the holdout or of any CV fold.
- Case study 2 compares the 0.1 and 0.2 splitters on Breast Cancer with 20%
  duplicated rows.

### Limitations
- Only exact duplicates are matched; near-duplicates are not detected.
- Under `time_col=`, exact duplicates are not kept together.

## 0.1.0 — 2026-11-01

First release of **prefit** (`pip install prefit`, `import prefit as pf`), the
decision-driven design. Replaces the v0 `Roadmap` API.

### Added
- `pf.analyze()` returns a `Report`: Evidence → Finding → Decision chain with
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
- `pf.review()`: reference-based hypothesis checks, baselines, residual /
  confusion / PR / segment analysis, list of never-verified assumptions;
  standalone or embedded HTML.
- Optional LLM layer: allowlisted payload with `llm_payload()` preview, column
  aliasing, schema-constrained output, unknown-id claims dropped, suggestions
  never applied; Anthropic, Gemini and local OpenAI-compatible providers.
- Versioned benchmark (`benchmarks/`), track record cited in reports.
- CLI: `prefit report data.csv --target y`.
- CI (ruff, mypy, pytest with coverage, HTML smoke test).

### Changed (pre-release API round)
- Data partitions are explicit: `pf.split(df, target, test_size, strategy)` is the
  primary path; `analyze` defaults to `holdout=None` and only sets a holdout aside
  when asked (`holdout=0.2`). `report.split()` became `report.holdout(df)`, which
  re-derives that partition and never splits anew.
- Fingerprints: `verify`, `holdout` and `register_external_holdout(dev_df=...)`
  check the frame against SHA-256 digests (column schema, dtypes, index, target,
  row-order-sensitive content) of the frame passed to `analyze`, and explain
  any mismatch (reordered, changed, shorter, different columns).
- `report.register_external_holdout(test_df, dev_df=...)` records an evaluation
  partition (metadata only, never profiled) and counts exact row overlap with
  the development data, reported as a review item without a leakage verdict.
- Deep immutability: all report models are frozen, lists are tuples and dicts
  read-only; every change is a new revision. `pf.load_report()` reloads JSON.
- API surface: reading / flow / automation on `Report`, internals under
  `report.details` (findings, evidence, profile, figures, manifest, context,
  limitations, rule_track_record, llm). Short reprs for all report objects,
  a notebook summary card, and `report.decisions_table()`. v0 aliases removed.
- Split columns (`groups`, `time_col`) stay out of the features unless
  `split_columns_as_features=True`, which raises a warning.
- HTML report rebuilt: Summary / Actions / Roadmap / Details navigation, short
  decision cards (why with the decisive number, do, experiment summary, at most
  one figure) with evidence, experiment, figure and rule/benchmark details in
  collapsed sections, consecutive roadmap numbering, collapsed Details section.
- Verification and baselines use the export splitter (repeated for power).
- Feature scaling is an export option (`scale=True`), not part of the
  procedural fit-in-fold decision.
- Bootstrap intervals computed on subsampled resamples are rescaled to the full
  sample size (they were ~sqrt(n/m) times too wide on large data).
- Single-feature association is a 2-fold cross-validated score.
- Single-class targets are rejected up front; +/-inf values produce an
  `infinite_values` review item instead of numpy warnings and are treated as
  missing by the exported preprocessing; duplicate detection hashes rows
  directly (much faster on wide data).
- `benchmarks/run.py` keeps the committed experiment section when run without
  `--verify`.

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
