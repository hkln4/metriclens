# prefit

**Decision-driven EDA for tabular supervised learning.**

Point `prefit` at your development data before you train. It profiles the
data, turns what it measures into a short list of *decisions* — how to split,
which columns to question, what to preprocess, which metric to trust — and shows
the evidence, threshold and uncertainty behind every one. Decisions that depend
on your data and model are tested with small, bounded experiments instead of
being assumed. At the end you get a scikit-learn model recipe that contains only
steps that are safe, that you confirmed, or that an experiment supported.

It is not AutoML: it does not choose a model or tune hyperparameters, and it
never applies an unconfirmed assumption silently.

**[Example report](https://hkln4.github.io/prefit/report.html)** ·
[Project page](https://hkln4.github.io/prefit/) · [PyPI](https://pypi.org/project/prefit/)

```bash
pip install "prefit[verify]"        # core + experiments/export (scikit-learn, scipy)
pip install "prefit[verify,report]" # + plotly, for fully offline HTML reports
pip install "prefit[llm]"           # optional LLM interpretation (Anthropic)
```

The core (`pandas`, `numpy`, `pydantic`) profiles, decides and renders reports
with no scikit-learn, no network and no API key.

---

## Quickstart

```python
import prefit as pf
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(random_state=0)

# 1. set the test data aside before anything is computed
dev_df, test_df = pf.split(df, target="price", test_size=0.2, strategy="random", random_state=42)

# 2. analyze the development data only
report = pf.analyze(dev_df, target="price", estimator=model)
report                                   # notebook: a short summary card with the next step
print(report)                            # roadmap: one line of why / do per decision
report.to_html("report.html")            # the full report

# 3. every step returns a new report revision
report = report.answer({"metric_priority": "typical_error"})
report = report.verify(dev_df)           # bounded experiments with your model
report = report.register_external_holdout(test_df, dev_df=dev_df, name="test set")

# 4. export only what is applied, confirmed or verified
built = report.export()
print(built.describe())                  # what is in, what is out, and why
fitted = built.model.fit(dev_df[built.feature_columns], dev_df["price"])

# 5. once, at the end, on the untouched test data
review = pf.review(test_df["price"], fitted.predict(test_df[built.feature_columns]), report)
report.to_html("report.html", review=review)
```

Shortcut: `pf.analyze(df, target, holdout=0.2)` sets the holdout aside itself,
before profiling; `dev_df, test_df = report.holdout(df)` gives it back. Without
`holdout=`, `analyze` never splits — the default is `holdout=None`.

From the shell:

```bash
prefit report dev.csv --target price --verify --answer metric_priority=typical_error --out report.html
```

An example report produced by `examples/quickstart.py` is live at
[hkln4.github.io/prefit/report.html](https://hkln4.github.io/prefit/report.html) (source:
[`examples/output/california_report.html`](https://github.com/hkln4/prefit/blob/main/examples/output/california_report.html)).

---

## The report object

```python
# reading
print(report); report.to_html(path); report.to_markdown()
report.roadmap; report.pending_questions(); report.decisions_table()
report.target; report.task

# flow (each returns a new revision; the old report is unchanged)
report.answer({...}); report.verify(df); report.interpret(llm)
report.register_external_holdout(test_df, dev_df=dev_df)
report.export(estimator); report.holdout(df); pf.review(...)

# automation
report.decisions; report.decision(id); report.questions; report.experiments
report.metric_plan; report.baseline; report.to_json(); report.llm_payload()

# internals: the traceability chain and reproducibility record
report.details.findings / .finding(id) / .evidence / .profile / .figures
report.details.manifest / .context / .limitations / .rule_track_record / .llm
```

`report.decisions_table()` is the best place to start in a notebook: one row per
decision with its status, a one-line why (with the decisive number), what to do,
and any experiment result.

**Reports are immutable.** Every model is frozen, lists are tuples and dicts are
read-only, all the way down. `answer`, `verify`, `interpret` and
`register_external_holdout` return a new report whose
`details.manifest.revision` is incremented and whose `history` records the step.
`pf.load_report(path)` reloads a report saved with `to_json`.

**The data is checked, never trusted.** A report stores no rows, so `verify`
and `holdout` take the DataFrame again. Its fingerprint — SHA-256 digests of the
column schema, dtypes, index, target and row-order-sensitive content — must
match the frame passed to `analyze`. A shuffled, edited, shortened or different
frame (the test set, say) raises an error that says what differs; nothing is
ever re-split.

---

## How decisions are made

Every rule turns *evidence* (a measured value, its 95% interval, the method)
into a *finding*, and the finding into one or more *decisions*. Each decision has
a **class**, which says how much the data alone can settle it:

| class | meaning | enters the model recipe when |
|---|---|---|
| **procedural** | about *how to evaluate*; correct regardless of the data (fit preprocessing inside folds, stratify, keep split columns and the holdout out). The only column it removes are constant ones. | always |
| **conditional** | may help or hurt depending on data and model (log target, power transform, target encoding, class weights, dropping mostly-missing columns) | an experiment with **your** estimator verifies it |
| **contextual** | cannot be decided from data (metric = cost of errors, is a column known at prediction time, is it an identifier, does time order matter) | you answer the question |
| **review** | needs a human look (duplicates, label conflicts, informative missingness, infinite values, collinearity, raw datetime/text columns v1 cannot encode) | never; it is a to-do |

Ground rules the catalog follows:

- **Thresholds are triggers, not verdicts.** Threshold-relevant statistics carry
  a bootstrap or Wilson interval (bootstrap intervals are computed for the full
  sample size even when resamples are subsampled); when the interval straddles
  the threshold the finding is *near threshold* and its decision *uncertain*.
- **Leakage is only ever a hint.** A feature that alone predicts the target in a
  2-fold cross-validated single-feature score raises a "single-feature
  predictive leakage hint — manual review required" and a question ("is this
  known at prediction time?"). Genuinely strong features look the same, so
  nothing is dropped until you answer.
- **Unsupported is not uninformative.** Raw datetimes, free text and string
  identifiers are left out of the exported features because v1 has no encoder
  for them; they are reported as a review item, never as "carries no signal".
- **No group splitting without real groups.** ID-like columns and duplicates
  produce review items. Group-aware splitters are used only with `groups=`;
  `TimeSeriesSplit` only with `time_col=`. Both columns stay out of the features
  unless `split_columns_as_features=True` (which adds a warning).
- **Split precedence is explicit.** `time_col` turns off class stratification
  (the report says so); `groups` with classification uses
  `StratifiedGroupKFold`, which balances classes only approximately; `groups`
  and `time_col` together raise an error in v1.
- **The data you analyze is the data the report describes.** The report opens
  with a data-partition note: which rows were analyzed, whether an internal or
  registered external holdout exists, and that it was never read.

## Experiments, and what they can claim

`verify` tests each conditional decision once per model family as "current
setup" vs. "current setup + this one change", on identical folds. The folds come
from the same splitter as the export CV decision (repeated when it is a plain
k-fold, for power); baselines use the same folds. A `Budget` caps the number of
experiments, wall-clock time and rows.

| protocol | when | test |
|---|---|---|
| `repeated_kfold` | default | corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004), Holm across the run |
| `grouped_kfold` | `groups=` | same correction on a single grouped k-fold run (low power, said in the report) |
| `time_series` | `time_col=` | none — expanding windows are not exchangeable; per-fold differences only, outcome left to you |

An experiment counts as **improved** only with the right direction, a
significant Holm-adjusted p-value, *and* a mean difference of at least the
practical threshold (`Budget(min_relative_effect=0.01)` = 1% of the baseline
score, or an absolute `min_effect`). Significant but smaller differences are
**negligible**. Per-fold differences are in the report.

- With `estimator=`, an improvement makes the decision **verified** (and
  exportable); a practical-size loss makes it **refuted**.
- Without it, a linear and a tree proxy are used. Their results are a
  **sensitivity analysis** (`supported_by_proxy`), not a verdict on your model,
  and are excluded from export unless you pass
  `export(include_proxy_supported=True)`. When one family gains and the other
  loses, the decision is marked model-family dependent.
- All of it is **development evidence**. The generalization claim comes from the
  untouched test data (or nested CV when you have none).

## Export

`report.export(estimator, scale=True)` returns:

- `model` — the full recipe for `fit` / `cross_validate`: `Pipeline(preprocess,
  estimator)`, wrapped in a `TransformedTargetRegressor` when a target transform
  was verified (`built.model_kind` says which). `None` without an estimator.
- `preprocessor` — the feature preprocessing alone. Feature scaling is an
  estimator-dependent default (`scale=False` turns it off), not a decision.
- `cv`, `scoring`, `feature_columns`, `groups`, `sort_by`.
- `included`, `excluded` (with status and reason), `needs_estimator` (eligible
  estimator-level steps that need an estimator), `unsupported` (columns left
  out), `assumptions` still open.

## The HTML report

`report.to_html(path)` writes one file with four sections:

- **Summary** — target, task, development rows; how many questions await you and
  how many decisions are critical or need review; one next step; the data
  partition note.
- **Actions** — what needs your attention (open questions, review items, and
  conditional decisions an experiment verified or refuted), the questions with
  their options (and an LLM suggestion awaiting your confirmation, if you ran
  one), and the metric plan.
- **Roadmap** — one short card per decision: status, title, one-line why with
  the decisive number, one-line do, experiment summary, and at most one figure.
  Evidence, experiment details, related figures and rule/benchmark details are
  one click away inside the card.
- **Details** — limitations, then collapsed: all diagnostics, findings,
  evidence, dataset profile, manifest and history.

No raw rows are embedded; category labels are shown only for low-cardinality
columns (`ProfileOptions(redact_values=True)` hides them all). Charts use
plotly.js: with `plotlyjs="cdn"` (default) the report is a single file but needs
internet for the charts; `plotlyjs="inline"` embeds plotly.js (~5 MB, needs the
`report` extra) for fully offline use; every chart also has a data table, which
works offline either way.

## Optional LLM interpretation

```python
report.llm_payload(pf.LLMConfig(share_column_names=False))   # inspect first
report = report.interpret(pf.LLMConfig(provider="anthropic"))
report = report.interpret(pf.LLMConfig(provider="local", model="llama3.1"))  # Ollama etc.
```

The LLM receives only the computed report — never rows, never category labels —
and must answer in a fixed JSON schema. It can summarize, prioritize existing
decisions, suggest answers to open questions (shown as awaiting your
confirmation, never applied) and raise *semantic* flags on column names, which
are marked manual-review. Claims that cite unknown evidence or decision ids are
dropped and counted. With `share_column_names=False` columns are aliased and
free-text messages are withheld; the report lists exactly what was sent. The
free-text `context=` never changes a decision by itself. Without an LLM the
report is complete.

Provider status: the interpretation logic and the local OpenAI-compatible
provider are tested (with a fake provider and a loopback server). The Anthropic
and Gemini providers are thin wrappers that have **not** been exercised against
the live APIs in this repository's tests.

## After training

`pf.review(y_true, y_pred, report, y_score=..., segments=...)` checks the
roadmap's hypotheses against real errors, each against an explicit reference
(e.g. "tails dominate" vs. what Gaussian errors would give), adds residual /
confusion / precision–recall / segment analysis, and lists every assumption
that was never verified.

---

## Benchmark

`python benchmarks/run.py` runs versioned fixture cases (synthetic data with one
injected hazard each, plus datasets bundled with scikit-learn) and checks that
each rule fires, stays silent, or reports *near threshold* as expected. Results,
including failures, are written to `benchmarks/results/<manifest>.json|md`;
`--verify` adds experiment outcomes on real datasets (a plain run keeps the
committed experiment section). Reports cite a rule's "detection pass rate on
benchmark <manifest>: k/n checks".

Current manifest `2026.11-2`: 20/20 cases pass. This is a detection pass rate on
known-hazard fixtures — not a claim about real-world accuracy, nor about how
often a decision improves a model.

## Case studies

`python examples/case_studies.py` (numbers below are from that script; fixed seeds):

| study | rule | before | after |
|---|---|---|---|
| California Housing, skewed target, log-transform (HGB) | `target_skew` | MAE 0.3125 | MAE 0.3074 (−1.6%) |
| Breast Cancer + 20% duplicated rows, de-duplicate (RF) | `duplicate_rows` | CV accuracy 0.9736 | 0.9579 (1.57 pts of inflation) |
| Diabetes + injected leaked feature, remove it (linear) | `leakage_hint` (critical) | CV R² 0.9999 | 0.4892 |

## Scope and limitations

- v1 covers tabular classification and regression. No images, text, time-series
  forecasting, clustering or model selection.
- Leakage screening sees single-feature association and numeric correlation
  only; leakage through feature combinations, time, preprocessing or repeated
  entities is not detected.
- Column kinds and the task are inferred heuristically; check the schema table.
- Experiments test one change at a time; interactions are not measured.
- `groups` and `time_col` cannot be combined in v1.
- Raw datetime, free-text and string-identifier columns are not encoded.

Design notes: [ARCHITECTURE.md](https://github.com/hkln4/prefit/blob/main/ARCHITECTURE.md) · changes: [CHANGELOG.md](https://github.com/hkln4/prefit/blob/main/CHANGELOG.md) ·
contributing: [CONTRIBUTING.md](https://github.com/hkln4/prefit/blob/main/CONTRIBUTING.md)
