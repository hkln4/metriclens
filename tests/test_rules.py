"""Rule-engine behaviour: the right findings fire on the right evidence, and the
known crash bug (Finding without validation/preprocessing fields) stays fixed."""
import numpy as np
import pandas as pd
import pytest

import metriclens as ml
from metriclens.enums import Severity


def _ids(roadmap):
    return {f.id for f in roadmap.findings}


def test_imbalance_fires_critical():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"f1": rng.normal(size=n),
                       "label": (rng.random(n) < 0.05).astype(int)})
    r = ml.analyze(df, target="label", use_llm=False)
    assert "class_imbalance" in _ids(r)
    imb = next(f for f in r.findings if f.id == "class_imbalance")
    assert imb.severity is Severity.CRITICAL
    assert "accuracy" in r.avoid
    # validation note must survive (this is the field that used to crash)
    assert r.validation_strategy


def test_skew_and_heavy_tails_fire_on_regression():
    rng = np.random.default_rng(1)
    y = rng.exponential(2.0, size=3000)
    df = pd.DataFrame({"f1": rng.normal(size=3000), "target": y})
    r = ml.analyze(df, target="target", use_llm=False)
    ids = _ids(r)
    assert "target_skew" in ids
    assert "heavy_tailed_target" in ids
    assert "RMSE" in r.avoid
    assert r.preprocessing_notes  # log/Box-Cox note present


def test_missingness_duplicates_constant_leakage_smalldata():
    rng = np.random.default_rng(2)
    n = 150
    target = rng.exponential(2.0, size=n)
    df = pd.DataFrame({
        "leaky": target + rng.normal(0, 1e-3, n),  # ~perfect correlation
        "noise": rng.normal(size=n),
        "const": 1,
        "target": target,
    })
    df.loc[:50, "noise"] = np.nan                  # >20% missing
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)  # duplicate rows
    r = ml.analyze(df, target="target", use_llm=False)
    ids = _ids(r)
    assert {"high_missingness", "duplicate_rows", "constant_features",
            "target_leakage", "small_data"} <= ids
    # findings are ordered critical-first
    sev = [f.severity for f in r.findings]
    assert sev == sorted(sev, key=lambda s: {Severity.CRITICAL: 0,
                                             Severity.WARN: 1, Severity.INFO: 2}[s])


def test_clean_data_yields_no_findings():
    rng = np.random.default_rng(3)
    n = 5000
    df = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "target": rng.normal(50, 5, size=n)})
    r = ml.analyze(df, target="target", use_llm=False)
    assert r.findings == []
    assert r.narrative_source == "deterministic"


def test_non_dataframe_rejected():
    with pytest.raises(TypeError):
        ml.analyze([1, 2, 3], target="x", use_llm=False)
