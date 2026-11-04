"""prefit is a pre-training step and must stay fast. Generous wall-clock bounds
(CI machines vary) that still catch a return of the 0.3 hot spots: a Python
loop in the bootstrap, Spearman computed twice, the single-feature score on
every row."""
import time

import numpy as np
import pandas as pd
import pytest

import prefit as pf

pytestmark = pytest.mark.slow


def _frame(n, p_num, p_cat, seed=0):
    rng = np.random.default_rng(seed)
    d = {f"n{i}": rng.normal(size=n) for i in range(p_num)}
    for i in range(p_cat):
        d[f"c{i}"] = rng.choice([f"v{j}" for j in range(8 if i % 2 else 200)], n)
    d["n0"][rng.random(n) < 0.1] = np.nan
    d["y"] = (d["n1"] + rng.normal(size=n) > 0.8).astype(int)
    return pd.DataFrame(d)


@pytest.mark.parametrize("n,p_num,p_cat,limit", [(200_000, 40, 10, 40.0), (20_000, 300, 0, 40.0)])
def test_analyze_stays_fast(n, p_num, p_cat, limit):
    df = _frame(n, p_num, p_cat)
    t = time.perf_counter()
    rep = pf.analyze(df, "y")
    assert time.perf_counter() - t < limit
    assert rep.details.profile.dataset.n_rows == n
    assert "seeded sample" in rep.details.profile.relations.corr_method or n <= 50_000
