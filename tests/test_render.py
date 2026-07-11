"""Render contracts: plain-text print has no markdown symbols, markdown does, and
both expose the three pipeline sections. Plus the executable recipe."""
import numpy as np
import pandas as pd

import metriclens as ml


def _imbalanced():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"f1": rng.normal(size=n),
                       "label": (rng.random(n) < 0.05).astype(int)})
    return ml.analyze(df, target="label", use_llm=False)


def test_plain_text_has_no_markdown_symbols():
    text = str(_imbalanced())          # __str__ -> to_text
    assert "#" not in text
    assert "**" not in text
    assert "EVALUATION ROADMAP" in text
    # three sections always present
    assert "1. COMPUTED METRICS" in text
    assert "2. RULE-BASED ROADMAP" in text
    assert "3. INTERPRETATION" in text


def test_markdown_keeps_markdown():
    md = _imbalanced().to_markdown()
    assert md.startswith("# Evaluation Roadmap")
    assert "**" in md
    assert "## 1. Computed metrics" in md


def test_profile_summary_is_populated():
    r = _imbalanced()
    assert r.profile_summary.get("task") == "classification"
    assert r.profile_summary.get("n_samples") == 2000


def test_validation_recipe_and_sklearn_objects():
    r = _imbalanced()
    recipe = r.validation_recipe()
    assert recipe["splitter"]["name"] == "StratifiedKFold"
    assert recipe["primary_metrics"]

    built = r.make_sklearn()
    from sklearn.model_selection import StratifiedKFold
    assert isinstance(built["cv"], StratifiedKFold)
    # imbalance recommends PR-AUC / balanced accuracy / macro F1 -> real scorers
    assert "average_precision" in built["scoring"]
    assert "balanced_accuracy" in built["scoring"]
