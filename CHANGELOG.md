# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/); before 1.0, minor versions may
change the public API.

## 0.4.0 — 2026-11-04

Usability, missing values, speed and an honest benchmark. Checked against Telco
Customer Churn (`benchmarks/notebooks/telco_churn.ipynb`) and Red Wine Quality;
the rules changed here are listed with their new versions under "Changed".

### Added
- **The frame stays with the report.** `analyze(..., keep_data=True)` (default) keeps a reference (no copy,
  never serialized); `verify()`, `holdout()`, `change_task(task)`, `register_external_holdout(test_df)`,
  `export().fit()` and `review(model)` need no frame. It is fingerprint-checked on every use, so an in-place
  edit after `analyze` raises. Passing the frame still works (and is required after `load_report`).
- **Keyword answers.** `report.answer(task="regression", metric="mae")`; column questions take
  `{column: True/False}` (`known_at_prediction={"x": False}`, `identifier=["user_id"]`,
  `missing_is_zero={"TotalCharges": True}`); options accept short aliases (`mae`, `rmse`, `recall`,
  `roc_auc`, ...). The dict form with question ids still works. `answer(task=...)` with a different task
  re-runs the analysis on the same rows (see Fixed).
- **Runnable next steps.** `report.next_steps()`, the notebook card, `print(report)` and the HTML summary
  give one complete line per open item, then the rest of the flow (`verify`, `export().fit()`, `review`).
- **Three states.** Every decision is *done*, *needs you* or *tested* (`Decision.state`); the card and the
  text report group by them. Statuses and details stay in `decisions_table()`.
- `ExportResult.fit(df=None)`: fits the recipe on the development rows (holdout excluded, exact feature
  columns, missing targets dropped, time order respected).
- `report.review(model, test_df=None)`: predicts with the model's own columns on the internal holdout (or
  `test_df`, checked against a registered external holdout), passes scores for the positive class, and warns
  when the same test data is reviewed twice in a session.
- `report.show()` / `report.show("d.xxx")` for notebooks: the card, a first-look table and the figures behind
  the decisions that need you (one decision with its evidence and figure). Color tokens are resolved in
  Python; without plotly each figure shows its data table. `report.first_look()`: one row per column with kind,
  missing %, distinct values, skew, signal, leak score and the decisions that mention it.
- **Missing values, measured** (`profiling/missingness.py`, `ColumnProfile.missingness`): for every column
  with at least 5 missing rows, the target in missing vs present rows (permutation test), the missing
  indicator's single-feature score, a *structural* pattern (missing exactly where another column takes one
  value), the feature that best separates missing rows, and the strongest rank correlation (what a model-based
  imputer could use).
- Rules `structural_missingness` v1 (a question: does missing mean 0 there? confirmed -> constant fill step),
  `informative_missingness` v1 (a review item with the rates and the p-value, Holm across columns; replaces
  the generic "check whether missingness is informative" item), `model_imputation` v1 (a conditional
  decision: IterativeImputer vs median, tested by `verify`).
- `Decision.alternatives`: the other ways to go and when each is right, for every conditional decision and
  the new missingness decisions; shown in the text report, the HTML card and `show("d.xxx")`.
- `ColumnProfile.signal`: a threshold-free single-feature strength (Gini = 2 AUC - 1 for classification,
  R² for regression). On Telco the old score was 0 for every categorical column because none flips the
  majority class; `signal` ranks Contract and tenure first. The leakage screen keeps its calibrated score.
- Figures: target in missing vs present rows; row distance from each duplicate to its original.
- Review: bootstrap intervals for ROC-AUC and PR-AUC (with duplicates resampled together), so the CV check
  also runs for those metrics.
- `Budget(n_jobs=None)`: automatic parallel folds in `verify` (serial until one cross-validation takes more
  than 2 s, then all cores); results do not depend on it.
- Benchmarks: `benchmarks/real_data.py` (plain flow vs prefit on five real datasets, several seeds, scored on a
  final set neither flow saw; results in `benchmarks/results/real_data.md`), `benchmarks/real_datasets.py`
  (cached loaders), `benchmarks/check_manifest.py` (CI: the committed benchmark must match the shipped rule
  versions and the package version), five new fixture cases (manifest `2026.11-4`, 26 cases).
- Tests: statistics against scipy/pandas references and by simulation (the corrected t-test rejects 3% of
  100 equal-learner comparisons where the naive paired test rejects 21%; Wilson coverage; skewness bootstrap
  coverage; permutation calibration), dirty-data kinds with Hypothesis, the suggested lines executed to the
  end, real-data golden tests (network, cached), wall-clock bounds. `pytest -n auto` runs the suite in ~20 s.

### Changed
- Numbers stored as text are numeric: blanks and placeholders (`""`, `" "`, `NA`, `?`, `null`, ...) are
  missing values, values with a leading zero stay codes. Telco's `TotalCharges` was a categorical column
  with 6,531 categories and no missing values, and prefit proposed target-encoding it.
- Rules: `task_confirmation` v4 asks only when the inference is ambiguous (an integer target with few
  values); a text, boolean, binary or continuous target is not questioned. `high_missingness` v2 (no
  generic review item), `unsupported_columns` v3 (a string column with a distinct value on every row is a
  row identifier: information, not a review item), `temporal_order` v2, `id_like_columns` v2,
  `leakage_hint` v3, `skewed_features` v4, `high_cardinality` v2, `class_imbalance` v2, `target_skew` v2,
  `metric_policy` v2 (keywords, aliases, alternatives; same triggers and thresholds).
- Speed: vectorized bootstrap (same random stream, same intervals), one Spearman matrix on at most 50,000
  rows with a fast path for missing values (equal to pandas), the single-feature score on at most
  `ProfileOptions.association_rows` (50,000) rows, vectorized category lookup. `analyze` on 200,000 x 50
  went from 12.9 s to 6.1 s, on 20,000 x 300 from 12.3 s to 7.6 s (with the new missingness and signal
  measurements included); `verify` with HistGradientBoosting on 7,000 rows from 17 s to 8 s.
- Wilson intervals use the exact normal quantile (1.95996...) instead of 1.96.
- `change_task(task, df=None)`: the task comes first; `change_task(df, task)` is still accepted.
- The CV check in the review states its limit: leakage that inflates the CV and the test score alike passes it.

### Fixed
- `cv_estimate()` (and the review's CV check) described the base setup, not the exported recipe, when `verify`
  verified a change: on Adult, with class weights verified, the CV estimate was 0.05 balanced accuracy below
  the model's real score and the check would have reported a development/test mismatch. `verify` now scores
  the recipe with every verified change on the same folds (`cv_estimate()["recipe"]` lists them). Found by
  the real-data benchmark.
- `change_task` (and so `answer(task=...)`) re-drew the internal holdout under the new task, so rows the
  first analysis had profiled could land in the holdout. The holdout now keeps its rows; `profile.holdout`
  records the task it was reserved under, so a loaded report re-derives the same rows.
- Benchmark manifest `2026.11-3` was re-run with 0.3.0 rules without a new manifest number; results now
  record the rule versions and CI refuses a mismatch.
- `llm/anthropic.py` reads text blocks with `getattr(b, "text", "")`.

### Repository
- Scratch notebooks removed; the Wine Quality guide moved to `benchmarks/notebooks/wine_quality_guide.ipynb`,
  next to the new `telco_churn.ipynb` (both tracked).

## 0.3.1 — 2026-11-03

### Fixed
- A review item closed with `report.resolve` reached the LLM only as status
  `reviewed`, without the user's note, while its action text still said "Record
  what you decide"; the model then presented the decision as open (found by the
  Red Wine Quality guide). The payload now marks such items `closed_by_user`
  and carries the note as `user_note` under the same rule as the free-text
  context (`include_context=True` and column names shared). The prompt tells
  the model not to reopen them, and the disclosure counts the notes sent.

## 0.3.0 — 2026-11-03

Feedback from the Red Wine Quality guide on Kaggle, each item verified on that
data before it was changed (one proposal, "skip transforms for tree models",
was refuted there and is not in this release: RandomForest predictions do
change under monotone transforms, only the score barely moves).

### Added
- Equivalence test for experiments: TOST against ±the practical threshold with
  the same corrected standard error, Holm-adjusted (`Experiment.p_equivalence`).
  New outcome and status `no_practical_effect`; a non-significant result without
  equivalence stays `inconclusive`, now with a note saying why.
- `pf.review(..., X=, n_bootstrap=, seed=)`: seeded 95% bootstrap intervals on the
  headline metrics (`ReviewReport.intervals`), resampling exact duplicates
  together when `X` is given, and a check that the CV estimate lies inside the
  test interval.
- `report.cv_estimate(family=None)`: the development CV estimate in natural units.
- `report.resolve(decision_id, note)`: close a review item with a note (status
  `reviewed`); shown in the report, `export().reviewed` / `describe()` and
  `pf.review`, carried through later revisions and JSON.
- `report.change_task(df, task)`: re-analyze under another task on the same
  fingerprint-checked frame, keeping valid answers, review notes, the external
  holdout and the options; `answer({"task": ...})` points to it.
- `Finding.task_dependent` / `Decision.task_dependent`: items that exist only
  under the inferred task are flagged while it is unconfirmed, in the report and
  in the LLM payload (`task_confirmed`, `task_dependent`); the prompt tells the
  model not to build advice on them.
- `Decision.standard`: standard practice (fit in fold, plain CV, holdout rule) is
  grouped apart in the text and HTML reports; `decisions_table()` gains
  `standard` and `note` columns.
- Duplicate evidence: `duplicate_groups`, `duplicate_rows_in_groups` and
  `duplicate_adjacency` (median row distance between copies vs. random order,
  permutation test) in the profile and the duplicate review item.
- `LLMInterpretation.dropped` (what was rejected and why) and `.raw_response`.

### Changed
- Rules: `task_confirmation` v2 (ordered-score hint in the question and option
  labels), `validation_split` v4, `holdout_policy` v3, `fit_in_fold` v3
  (standard flag), `duplicate_rows` v3 (extra copies vs. rows in groups,
  adjacency, resolve hint), `skewed_features` v3 (accurate near-threshold text).
- Experiment changes are shown in natural units ("MAE +0.0002 (lower = better)"
  instead of a negated score labelled "higher = better").
- Report text: "the test set (321 rows) was registered" (was "an test set"); the
  regression baseline row says "median", classification "class prior"; unknown
  answer keys list every question and point review items to `resolve`.

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
