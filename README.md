# metriclens

**Data-aware evaluation roadmaps: which metrics matter for your data, and why.**

Point `metriclens` at a dataset *before* you train. It profiles the data
deterministically and hands back a reasoned roadmap: which metrics to report,
which ones will mislead you and why, and how to set up validation — so you don't
have to stare at the data and figure it out by hand every time.

The design intent and rationale live in [ARCHITECTURE.md](ARCHITECTURE.md). This
file is the practical guide to what the library does *today*.

---

## Install

```bash
pip install -e .                 # core: pandas, numpy, pydantic
pip install -e ".[llm]"          # optional: Anthropic-written narrative
pip install google-genai         # optional: Gemini-written narrative
pip install scikit-learn         # optional: make_sklearn() executable output
```

The core works offline with no API key. LLM narrative and `make_sklearn()` are
strictly optional and imported lazily — `import metriclens` needs none of them.

---

## Quickstart

```python
import pandas as pd
import metriclens as ml

roadmap = ml.analyze(df, target="price", context="house-price estimation")
print(roadmap)
```

`print(roadmap)` renders three sections, one per pipeline stage — always in this
order, each shown only if it produced something:

```
EVALUATION ROADMAP

1. COMPUTED METRICS · DETERMINISTIC PROFILER
- Modality: tabular · Task: regression (inferred — confirm)
- Samples: 20640 · Features: 8
- Target: skew=0.98 · IQR-outliers=5.2% · std=1.15

2. RULE-BASED ROADMAP · RULE ENGINE
Recommended metrics:
- primary: MAE, median absolute error
- secondary: RMSE (report alongside, not alone)
Avoid:
- RMSE — Heavily weights a few large/outlier errors under a skewed target.
- R² — Can look healthy while large-error behavior is poor.
Preprocessing:
- Consider a log/Box-Cox transform of the target ...
Findings:
- [warn] target_skew: Target is skewed (skew=0.98) with 5.2% IQR-outliers ...

3. INTERPRETATION · DETERMINISTIC
Detected 1 point(s) worth attention: ...
```

Programmatic access:

```python
roadmap.recommended_metrics   # {'primary': [...], 'secondary': [...]}
roadmap.avoid                 # {metric: why it misleads}
roadmap.validation_strategy   # ['Use stratified k-fold ...']
roadmap.preprocessing_notes
roadmap.findings              # raw Findings, each bound to the number that fired it
roadmap.profile_summary       # the computed metrics, numbers only
roadmap.to_markdown()         # markdown instead of plain text
```

### The trust boundary

Everything in sections 1–2 is deterministic and reproducible: the same data
always yields the same findings. A finding **cannot** be constructed without the
`evidence` number that triggered it, so recommendations are always bound to a
concrete statistic — no platitudes. The LLM (section 3) only ever sees the
computed numbers, never the raw data, and only turns the *why* into prose. With
no key it falls back to a deterministic narrative; the roadmap is already
complete without it.

---

## What it computes (deterministic profiler)

Tabular, today. Pure measurement — no interpretation.

- **Core:** n_samples, n_features; for classification: n_classes, class counts,
  imbalance ratio, minority fraction.
- **Regression target:** mean/median/std/min/max, **skew**, **kurtosis**,
  **IQR-outlier fraction**, zero fraction, negative fraction, missing count.
- **Features:** per-column dtype / cardinality / missingness, `max_missing_fraction`,
  constant columns, near-unique (id-like) columns, numeric leakage hints
  (`|r|≥0.98` with the target), **duplicate-row fraction**.
- **Task inference:** classification vs regression from the target (always
  flagged `inferred` for you to confirm).

## What it concludes (rule engine)

Every rule is *evidence-gated*: it can only fire if the number it depends on was
computed, so a rule structurally cannot suggest an irrelevant metric.

| Rule | Fires when | Recommends / avoids |
|---|---|---|
| `class_imbalance` | imbalance ≥ 3:1 (≥10 critical) | PR-AUC, balanced acc, macro F1, MCC / **avoid** accuracy, ROC-AUC |
| `target_skew` | \|skew\|≥1 or outliers≥5% | MAE, median abs err / **avoid** RMSE-alone, R² |
| `heavy_tailed_target` | excess kurtosis ≥ 3 | MAE, Huber/quantile / **avoid** RMSE, MSE |
| `zero_inflation` | ≥30% of target is zero | MAE, Tweedie, two-part / **avoid** MAPE, MSLE |
| `negative_target_values` | any negative target | **avoid** MSLE, MAPE |
| `target_leakage` | feature \|r\|≥0.98 with target | **critical**: remove feature, every metric is inflated |
| `high_missingness` | column >20% missing (>50% critical) | situational imputation advice; fit imputation inside CV folds |
| `duplicate_rows` | any exact duplicate rows | de-duplicate / group-split before evaluating |
| `constant_features` | zero-variance column | drop it |
| `id_like_features` | near-unique identifier column | drop/hash — memorization/leakage risk |
| `small_data` | n < 1000 (n < 200 critical) | cross-validation + confidence intervals |

## Executable output

Turn the recommendation into something you can run, instead of copying metrics by
hand:

```python
recipe = roadmap.validation_recipe()      # dependency-free plan (splitter + metrics)

built = roadmap.make_sklearn()            # real scikit-learn objects
from sklearn.model_selection import cross_validate
cross_validate(model, X, y, cv=built["cv"], scoring=built["scoring"])
```

`make_sklearn()` picks the splitter from the findings — `StratifiedKFold` for
class imbalance, `GroupKFold` when duplicates / id-like columns are present,
`KFold` otherwise — and maps the recommended metrics to built-in sklearn scorers.

## Second pass: did the roadmap hold?

`analyze` is a *hypothesis* formed before training. After training, `review`
checks it against the actual errors:

```python
report = ml.review(y_true, y_pred, roadmap=roadmap)
print(report)
```

- **Regression:** measures RMSE/MAE ratio and how much of the total error the
  worst 5% carries → confirms or refutes "a few large errors dominate".
- **Classification:** measures accuracy vs minority-class recall → confirms or
  refutes "accuracy is hiding the failure".

It is deterministic and honest — on well-behaved errors it will tell you the
warning **did not** hold, rather than rubber-stamping the roadmap.

---

## Status

**Working today:** tabular profiling, the 11 rules above, three-section render
(plain text + markdown), optional LLM narrative (Anthropic / Gemini),
`validation_recipe()` / `make_sklearn()`, `review()` for regression &
classification, a passing pytest suite (`pytest -q`).

**Not yet (see [ARCHITECTURE.md](ARCHITECTURE.md)):**

- **One-call evaluation** — a convenience like
  `ml.evaluate(model, X, y, roadmap=...)` that runs the recommended CV and the
  review in a single call, removing the remaining train/split/predict boilerplate.
- **Other modalities** (image / text) — the abstraction is in place; no profiler
  shipped yet.
- **Sample-size-aware thresholds / confidence on findings.**
- **More rules** (classification leakage, temporal/time-based split,
  high-cardinality categoricals, label-noise estimate).

## License

MIT.
