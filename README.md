# metriclens

**Decision-driven EDA for tabular supervised learning.**

Point `metriclens` at a dataset before you train. It profiles the data, turns what
it measures into a short list of *decisions* — how to split, which columns to
question, what to preprocess, which metric to trust — and shows the evidence,
threshold and uncertainty behind every one. Decisions that depend on your data
and model are tested with small, bounded experiments instead of being assumed.
At the end you get a scikit-learn pipeline that contains only steps that are
safe, that you confirmed, or that an experiment supported.

It is not AutoML: it does not choose a model or tune hyperparameters, and it
never applies an unconfirmed assumption silently.

```bash
pip install -e ".[verify]"        # core + experiments/export (scikit-learn, scipy)
pip install -e ".[verify,report]" # + plotly, for fully offline HTML reports
pip install -e ".[llm]"           # optional LLM interpretation (Anthropic)
```

The core (`pandas`, `numpy`, `pydantic`) profiles, decides and renders reports
with no scikit-learn, no network and no API key.

---

## Quickstart

```python
import metriclens as ml
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(random_state=0)
report = ml.analyze(df, target="price", estimator=model)   # holdout reserved first
print(report)                       # roadmap + questions only you can answer
report.to_html("report.html")       # single-file report, figures included

report = report.answer({"metric_priority": "typical_error"})
report = report.verify(df)          # bounded experiments, with your model

built = report.export()             # Pipeline + CV splitter + scoring
dev, holdout = report.split(df)
fitted = built.pipeline.fit(dev[built.feature_columns], dev["price"])
review = ml.review(holdout["price"], fitted.predict(holdout[built.feature_columns]), report)
report.to_html("report.html", review=review)
```

Or from the shell:

```bash
metriclens report data.csv --target price --verify --answer metric_priority=typical_error --out report.html
```

An example report produced by `examples/quickstart.py` is in
[`examples/output/california_report.html`](examples/output/california_report.html).

---

## How decisions are made

Every rule turns *evidence* (a measured value, its 95% interval, the method)
into a *finding*, and the finding into one or more *decisions*. Each decision has
a **class**, which says how much the data alone can settle it:

| class | meaning | enters the pipeline when |
|---|---|---|
| **procedural** | about *how to evaluate*; correct regardless of the data (fit preprocessing inside folds, stratify, keep the holdout untouched). Never changes the data's content. | always |
| **conditional** | may help or hurt depending on data and model (log target, power transform, target encoding, class weights, dropping mostly-missing columns) | an experiment with **your** estimator verifies it |
| **contextual** | cannot be decided from data (metric = cost of errors, is a column known at prediction time, is it an identifier, does time order matter) | you answer the question |
| **review** | needs a human look (duplicates, label conflicts, informative missingness, collinearity) | never; it is a to-do |

Some ground rules the catalog follows:

- **Thresholds are triggers, not verdicts.** Threshold-relevant statistics carry
  a bootstrap or Wilson interval; when the interval straddles the threshold the
  finding is marked *near threshold* and its decision *uncertain*.
- **Leakage is only ever a hint.** A feature that alone predicts the target on
  held-out rows raises "leakage hint — manual review required" and a question
  ("is this known at prediction time?"). Nothing is dropped until you answer.
- **No group splitting without real groups.** ID-like columns and duplicates
  produce review items. `GroupKFold`-style splitters are used only when you pass
  `groups=`; `TimeSeriesSplit` and a time-ordered holdout only with `time_col=`.
- **The holdout is reserved before anything is computed.** Profiling, rules and
  experiments all run on the development rows.

## Experiments, and what they can claim

`verify` tests each conditional decision once per model family as
"current setup" vs. "current setup + this one change", on identical
(repeated) folds. Differences are judged with the Nadeau–Bengio corrected
paired t-test and Holm-corrected across the run. A `Budget` caps the number of
experiments, wall-clock time and rows.

- With `estimator=`, a significant improvement makes the decision **verified**
  (and pipeline-eligible); a significant loss makes it **refuted**.
- Without it, a linear and a tree proxy are used. Their results are a
  **sensitivity analysis** (`supported_by_proxy`), not a verdict on your model,
  and are excluded from the pipeline unless you pass
  `export(include_proxy_supported=True)`. When one family gains and the other
  loses, the decision is marked model-family dependent.
- All of it is **development evidence**. The generalization claim comes from
  the untouched holdout (or nested CV when the data is too small for one).

## The report

`report.to_html(path)` writes one self-contained file. Decision cards link to
their evidence (value, interval, method), the rule version and thresholds, any
experiment, and the figures that show them. Figures are selected by applicability
and importance, each states what it shows and why it was chosen, and each has a
data table. Raw rows are never embedded; category labels are shown only for
low-cardinality columns (`ProfileOptions(redact_values=True)` hides them all).
`plotlyjs="cdn"` (default) loads plotly.js from its CDN, `"inline"` embeds it for
offline use, `"none"` shows tables only.

`report.to_text()`, `to_markdown()` and `to_json()` are also available; the JSON
holds the full profile, evidence, decisions and manifest (versions, seed, rule
versions, budget).

## Optional LLM interpretation

```python
report.llm_payload(ml.LLMConfig(share_column_names=False))   # inspect first
report = report.interpret(ml.LLMConfig(provider="anthropic"))
report = report.interpret(ml.LLMConfig(provider="local", model="llama3.1"))  # Ollama etc.
```

The LLM receives only the computed report — never rows, never category labels —
and must answer in a fixed JSON schema. It can summarize, prioritize existing
decisions, suggest answers to open questions (shown, never applied) and raise
*semantic* flags on column names ("sounds like it is recorded after the
outcome"), which are marked manual-review. Claims that cite unknown evidence or
decision ids are dropped and counted. With `share_column_names=False` columns
are aliased and free-text messages are withheld; the report lists exactly what
was sent. Without an LLM the report is complete.

## After training

`ml.review(y_true, y_pred, report, y_score=..., segments=...)` checks the
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
reports cite a rule's record as "correct on k/n fixture cases (manifest …)".
`--verify` also records experiment outcomes on real datasets.

Current manifest `2026.11-1`: 20/20 cases pass. These are detection checks on
known hazards, not a claim about how often a decision improves a model.

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

Design notes: [ARCHITECTURE.md](ARCHITECTURE.md) · changes: [CHANGELOG.md](CHANGELOG.md) ·
contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
