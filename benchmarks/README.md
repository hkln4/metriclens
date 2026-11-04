# benchmarks

Two questions, two benchmarks, and the notebooks that motivated them.

| file | question | run |
|---|---|---|
| `fixtures.py`, `run.py` | Does each rule fire where it should, and stay silent where it should not? Synthetic data with one injected hazard per case, plus datasets bundled with scikit-learn. | `python benchmarks/run.py` (seconds; `--verify` adds experiments on real data) |
| `real_data.py` | Does the score a workflow *reports* match the score the model *gets*? Plain flow vs prefit on five real datasets, several seeds, both scored on a final set neither saw. | `python benchmarks/real_data.py [names]` (~15 min, downloads once) |
| `real_datasets.py` | Cached loaders (Wine Quality red, Telco churn, Adult, California Housing, Breast Cancer). Cache: `$PREFIT_DATA` or `~/.cache/prefit`. | imported by the above and by `tests/test_real_data.py` |
| `check_manifest.py` | Do the committed results describe the rules that ship? (CI) | `python benchmarks/check_manifest.py` |
| `results/` | Every run's output, failures included: `<manifest>.json/.md` for the detection benchmark, `real_data.json/.md`. | |
| `notebooks/wine_quality_guide.ipynb` | The Red Wine Quality guide (prefit 0.3): duplicates in a "clean" dataset, an ordered score read as classes. Its feedback became 0.3.0. | Jupyter / Kaggle |
| `notebooks/telco_churn.ipynb` | Telco churn (prefit 0.4): findings from human EDAs that change a decision, checked one by one against prefit (caught / missed / noise); the TotalCharges blanks; the four-line flow. | Jupyter / Kaggle |

The detection benchmark's manifest version (`MANIFEST_VERSION` in `fixtures.py`)
changes whenever a rule, a case or an expectation changes; reports cite it as
"rule detection pass rate on benchmark <manifest>". It measures detection on
known hazards, not real-world accuracy: that is what `real_data.py` is for.
