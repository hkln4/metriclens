# prefit

**Decision-driven EDA for tabular supervised learning.** Point it at your data before you train; it tells you
what to decide, shows the evidence, tests what can be tested with your model, and keeps the test set honest.

```python
import pandas as pd, prefit as pf
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

report = pf.analyze(df, "quality", estimator=HistGradientBoostingRegressor(), holdout=0.2)
report = report.answer(task="regression", metric="mae").verify()
fitted = report.export().fit()
review = report.review(fitted)
```

What `print(report)` says right after `analyze` (abridged):

```text
classification (inferred) on 'quality' · 1,277 development rows, 11 features · 2 question(s) open

Status: ! 4 needs you · ⚗ 0 tested · ✓ 5 done

Data partition: ... The internal holdout (322 rows, stratified random 20%; exact duplicates kept on one side)
was set aside before profiling and was never read ... 220 set(s) of exact duplicate rows were kept whole.

Run next:
    report = report.answer(task="classification")  # or "regression"
    report = report.answer(metric="f1_macro")  # or "accuracy"
    report = report.resolve("d.duplicate_rows", "what you found and decided")
    report = report.verify()  # test the conditional decisions with your model
    fitted = report.export().fit()  # the recipe: applied, confirmed and verified steps
    review = report.review(fitted)  # once, on the internal holdout

QUESTIONS FOR YOU
- task: Is this a classification task? Ordered scores are usually regression; the data cannot tell.
```

and what `print(review)` says at the end:

```text
- MAE: 0.5271  [95% CI 0.474, 0.5833]        baseline MAE (predict the median): 0.6646
- [HELD] The development CV estimate describes the score on the untouched test data.
    CV MAE 0.5053 lies inside the test interval [0.474, 0.5833]
- never verified: Decide whether duplicates are the same record or legitimate repeats
```

```bash
pip install "prefit[verify]"          # + scikit-learn, scipy: experiments, export, review
pip install "prefit[verify,report]"   # + plotly: figures in notebooks, fully offline HTML
```

**[Example report](https://hkln4.github.io/prefit/report.html)** · [Project page](https://hkln4.github.io/prefit/) ·
[PyPI](https://pypi.org/project/prefit/) · [Changelog](CHANGELOG.md) · [Architecture](ARCHITECTURE.md)

---

## How it thinks

prefit profiles the development rows and turns measurements into **decisions**. Each decision has a class,
which says how much the data alone can settle it:

| class | what it is | enters the model recipe when |
|---|---|---|
| **procedural** | how to evaluate, right for any data: fit preprocessing inside folds, stratify, keep exact duplicates and the holdout on one side | always |
| **conditional** | may help or hurt depending on data and model: log target, power transform, target encoding, class weights, model-based imputation, dropping mostly-missing columns | an experiment with **your** estimator verifies it |
| **contextual** | cannot be decided from data: the cost of errors, whether a column exists at prediction time, what a missing value means | you answer the question |
| **review** | needs a human look: duplicates, missingness that relates to the target, infinite values, raw dates and text | never; it is a to-do |

You see three states: **done** (applied, confirmed, closed), **needs you** (a question, a review item, an
experiment you have not run) and **tested** (an experiment settled it, in or out of the recipe). Statuses and
details are in `report.decisions_table()`.

Three principles, used to decide what goes in:

- **A statistic enters the roadmap only if it changes a decision.** Otherwise it stays in the profile (the
  first-look table, the HTML details). Missing values completely at random raise nothing; missing values that
  sit exactly where `tenure == 0` become a question.
- **Recommendations are hypotheses.** Every conditional decision lists its alternatives and when each is right,
  and how to find out cheaply (usually: `verify`). Thresholds are triggers, not verdicts: a statistic whose
  interval straddles its threshold is *near threshold* and its decision *uncertain*.
- **prefit does not choose models.** It measures whether a suggestion helps *your* model. Leakage is only ever a
  hint ("is this known at prediction time?"); strong genuine features look the same.

## The flow

```python
report = pf.analyze(df, target, estimator=model, holdout=0.2)   # holdout set aside before anything is computed
report                                    # notebook card: three states + the lines to run next
report.show()                             # + first-look table and the figures behind what needs you
report.show("d.duplicate_rows")           # one decision: why, do, alternatives, evidence, figure
print(report)                             # the same as text, with the roadmap

report = report.answer(task="regression", metric="mae")         # keywords; see the table below
report = report.resolve("d.duplicate_rows", "re-entered records; kept, the split keeps copies together")
report = report.verify()                  # bounded experiments for the conditional decisions
built = report.export(); print(built.describe())                 # what is in, what is out, and why
fitted = built.fit()                      # fit on the development rows (holdout excluded)
review = report.review(fitted)            # once, on the holdout; a second look warns
report.to_html("report.html", review=review)
```

The report keeps a reference to the frame you analyzed (no copy, never serialized) and checks its fingerprint
every time it uses it, so an edit after `analyze` raises instead of silently changing results.
`analyze(..., keep_data=False)` keeps nothing; then pass the frame (`verify(df)`, `holdout(df)`, ...), as you
must after `pf.load_report(path)`. If you split yourself, `pf.split(df, target)` keeps exact duplicates on one
side; analyze the development part and pass the test part to `report.review(fitted, test_df)`.

**Answers.** Every open question names its keyword, and `print(report)` writes the line to run.

| keyword | question | answers |
|---|---|---|
| `task` | classification or regression? (asked only when ambiguous, e.g. a 0–10 score) | `"classification"`, `"regression"`; a different task re-runs the analysis on the same rows and the same holdout |
| `metric` | which error costs more? | regression `"mae"`, `"rmse"`, `"mape"`; binary `"recall"`, `"precision"`, `"balanced_accuracy"`, `"roc_auc"`; multiclass `"f1_macro"`, `"accuracy"` (or the option names) |
| `known_at_prediction` | is this column known before the target? | `{"col": True/False}` (False drops it) |
| `identifier` | is this integer column an id? | `{"col": True/False}` or `["col", ...]` |
| `missing_is_zero` | does a structural blank mean 0? | `{"col": True/False}` |
| `time_order` | must the model predict the future from the past? | `True/False` |

The dict form with question ids (`answer({"metric_priority": "typical_error"})`) works too.

## Missing values

For every column with at least five missing rows, prefit measures how the values go missing:

- **the target where values are missing vs present** (minority-class share or mean, a seeded permutation test,
  Holm across columns) and the single-feature score of the missing indicator;
- **structural gaps**: missing exactly where another column takes one value (Telco: `TotalCharges` is blank in
  exactly the 11 rows with `tenure == 0`);
- **related features**: which feature best separates missing from observed rows (missing at random given it),
  and how well other features predict the column.

These become three kinds of decision: a **question** for a structural gap ("does missing mean 0 there?"; yes
fills the constant inside the Pipeline), a **review item** when missingness relates to the target, with the
rates ("where 'lab' is missing, class '1' is 85% of rows; where it is present, 42% (permutation p = 0.002)",
from the `informative_missing` benchmark case), and a
**conditional** decision to test model-based imputation (IterativeImputer) against the median when other
features predict the column. Numbers stored as text are read as numbers, with blanks and placeholders (`""`,
`"NA"`, `"?"`, ...) as missing values.

## Experiments, and what they can claim

`verify` tests each conditional decision once per model family as "current setup" vs "current setup + this one
change", on identical folds from the same splitter as the exported CV (repeated for power). `Budget` caps
experiments, time and rows; folds run in parallel automatically once one cross-validation takes over 2 s.

| protocol | when | test |
|---|---|---|
| `repeated_kfold` | default | corrected repeated k-fold t-test (Nadeau & Bengio 2003), Holm across the run |
| `grouped_kfold` | `groups=` | same correction on one grouped k-fold run (low power, said in the report) |
| `time_series` | `time_col=` | none: expanding windows are not exchangeable; per-fold differences only |

**Improved** needs the right direction, a significant Holm-adjusted p-value *and* a change of at least the
practical threshold (`Budget(min_relative_effect=0.01)`). **No practical effect** needs a passed equivalence test
(TOST); otherwise a non-significant result is **inconclusive**. With your estimator an improvement *verifies*
the decision; without one, a linear and a tree proxy give a sensitivity analysis that is not exported by
default. All of it is development evidence: the generalization claim comes from the holdout.

## After training

`report.review(fitted)` (or `pf.review(y_true, y_pred, report, y_score=..., X=...)`) checks the roadmap's
hypotheses against the real errors, each against an explicit reference, with seeded 95% bootstrap intervals
on the headline metrics (ROC-AUC and PR-AUC included; exact duplicates are resampled together), residual /
confusion / precision–recall / segment views, and the list of assumptions that were never verified.

It also checks that the development CV estimate lies inside the test interval. **This check has a limit:**
leakage that inflates the CV and the test score alike (a leaked feature present in both, near-duplicates on
both sides of the split) moves both together and passes it. It detects a mismatch between development and test
data, not leakage. On Red Wine Quality a plain flow passes it too, while about a fifth of its test rows have an
exact copy in its training rows.

## Does it work?

**On real data** (`python benchmarks/real_data.py`, results in
[`benchmarks/results/real_data.md`](benchmarks/results/real_data.md)): a plain flow (`train_test_split`, KFold)
and the prefit flow run on the same data with the same estimator; a final set E that neither flow sees, with no
exact copy of its rows elsewhere, scores both models. *Optimism* is how much better a flow's estimate looks than
its score on E (positive = optimistic).

| dataset (seeds) | metric | CV optimism: plain → prefit | own-test optimism: plain → prefit | score on E: plain / prefit |
|---|---|---|---|---|
| wine_red (10) | MAE | +0.047 → -0.012 | +0.067 → -0.001 | 0.505 / 0.511 |
| breast_cancer (10) | bal. acc. | -0.000 → +0.009 | +0.021 → +0.000 | 0.952 / 0.947 |
| telco (5) | ROC-AUC | -0.010 → -0.013 | -0.010 → -0.004 | 0.838 / 0.839 |
| california (3) | MAE | -0.002 → -0.000 | +0.006 → +0.001 | 0.316 / 0.313 |
| adult (3) | bal. acc. | -0.003 → -0.007 | -0.001 → -0.001 | 0.796 / 0.847 |

Wine is the dataset with a hazard: 22% of the plain flow's test rows have an exact copy in its training rows, and
both its CV and its own test score look about 0.05–0.07 MAE better than the model is. prefit's estimates are
within noise of the truth. The price is visible too: keeping copies on one side leaves fewer distinct training
rows (818 vs 848), and prefit's model scores 0.006 MAE worse on E. On Breast Cancer, Telco and California there is
nothing to fix, and both flows report honestly and score alike. On Adult, `verify` found that class weights help
the user's model (3/3 seeds), so prefit's model scores higher on E; its CV estimate is for that recipe (an early
0.4 build reported the base setup's CV here, 0.05 too low; the benchmark caught it). The run takes about 15
minutes; standard deviations are in the results file.

**Detection** (`python benchmarks/run.py`): versioned fixture cases with one injected hazard each, plus datasets
bundled with scikit-learn. Manifest `2026.11-4`: 26/26 cases pass, including a check that no exact duplicate
lands on both sides of the holdout or any CV fold. This is a detection pass rate on known hazards, not a claim
about real-world accuracy. CI fails when the committed results come from other rule versions.

**Case studies** (`python examples/case_studies.py`; fixed seeds; the Wine row is from the real-data benchmark):

| study | rule | before | after |
|---|---|---|---|
| Red Wine Quality, real data: duplicates nobody injected | `duplicate_rows` | plain flow: CV MAE 0.047 and test MAE 0.067 better than on unseen data (10 seeds) | prefit: −0.012 and −0.001 (honest); MAE on unseen data 0.511 vs 0.505 |
| California Housing, skewed target, log-transform (HGB) | `target_skew` | MAE 0.3125 | MAE 0.3074 (−1.6%) |
| Breast Cancer + 20% duplicated rows (RF): old vs new CV | `duplicate_rows` | copies straddle folds: CV accuracy 0.9702 | duplicate-aware CV: 0.9604 (0.98 pts of inflation removed) |
| Diabetes + injected leaked feature, remove it (linear) | `leakage_hint` | CV R² 0.9999 | 0.4892 |

**The statistics are tested against references and by simulation** (`tests/test_stats_reference.py`): the
corrected t-test reduces to scipy's paired t-test, TOST to two one-sided tests, Wilson intervals match scipy,
the fast Spearman matches pandas, the weighted ROC/PR-AUC match scikit-learn. Comparing two copies of one
learner that differ only in their seed (100 simulated data sets), the corrected test rejects 3% of the time
where the naive paired test rejects 21%; the missingness permutation test rejects at about α when values are missing at
random. Percentile bootstrap intervals of the skewness cover the true value about 92% of the time at n = 500,
a little under the nominal 95%.

## Speed

`analyze` on 200,000 rows × 50 columns: about 6 s (0.3: 12.9 s); 20,000 × 300: about 8 s (0.3: 12.3 s) (one laptop core). The
bootstrap is vectorized, rank correlations and single-feature scores use at most 50,000 rows
(`ProfileOptions.association_rows`), and `verify` parallelizes folds once they are slow. The test suite runs in
about 20 s with `pytest -n auto`.

## The HTML report

`report.to_html(path)` writes one file: **Summary** (three states, the next lines, the data partition),
**Actions** (what needs you, the questions with their answer lines, the metric plan), **Roadmap** (one short card
per decision: why with the decisive number, do, experiment, alternatives, one figure; evidence and rule details
one click away) and **Details** (limitations, all diagnostics, findings, evidence, profile, manifest, history).
No raw rows are embedded; category labels only for low-cardinality columns (`ProfileOptions(redact_values=True)`
hides them). Charts use plotly.js from a CDN (`plotlyjs="inline"` for offline); every chart also has its data
table.

## Optional LLM interpretation

```python
report.llm_payload(pf.LLMConfig(share_column_names=False))     # inspect what would be sent
report = report.interpret(pf.LLMConfig(provider="anthropic"))  # or "gemini", or "local" (Ollama etc.)
```

The LLM receives only the computed report, never rows or category labels, and answers in a fixed JSON schema:
a summary, priorities among existing decisions, suggested answers (shown, never applied) and semantic flags on
column names. Claims that cite unknown ids are dropped (`details.llm.dropped`). Review items you closed are sent
as closed, with your note. Without an LLM the report is complete. The Anthropic and Gemini providers are thin
wrappers not exercised against the live APIs in this repository's tests.

## Scope and limitations

- Tabular classification and regression. No images, text, forecasting, clustering or model selection.
- Leakage screening sees single-feature association and numeric correlation only; leakage through feature
  combinations, time, preprocessing or repeated entities is not detected.
- **Relations between columns are not checked**: a column that is a product of two others (Telco
  `TotalCharges` ≈ `tenure` × `MonthlyCharges`), or categorical columns that repeat each other ("No internet
  service" in six columns). Both are reported by human EDAs of Telco and missed here.
- Duplicates are matched exactly; near-duplicates are not detected. Pass `groups=` when rows share an entity.
  Keeping copies on one side costs distinct training rows (the real-data table shows how many).
- Column kinds and the task are inferred heuristically; check the first-look table.
- Experiments test one change at a time; interactions are not measured. `groups` and `time_col` cannot be
  combined. With `time_col`, exact duplicates are not kept together (time order wins).
- Raw datetime, free-text and string-identifier columns are not encoded.

## Repository

```text
prefit/            the library
  profiling/         kinds, stats, missingness, tabular profile (measures, never concludes)
  rules/             the versioned rule catalog and the metric policy (decides)
  report/            the immutable report contract
  render/            text, HTML, notebook view, tables
  verify.py export.py posttrain.py splitters.py guard.py fingerprint.py figures.py api.py
tests/             unit, golden, statistical reference, dirty data, real data (network), timing
benchmarks/        detection benchmark, real-data benchmark, cached datasets, results, notebooks
examples/          quickstart.py, case_studies.py, output/ (the example report)
site/              the project page
```

Design notes: [ARCHITECTURE.md](ARCHITECTURE.md) · contributing: [CONTRIBUTING.md](CONTRIBUTING.md) ·
benchmarks: [benchmarks/README.md](benchmarks/README.md)
