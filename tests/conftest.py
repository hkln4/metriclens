import os
import sys
from pathlib import Path

# under pytest-xdist every worker gets one thread for OpenMP/BLAS, so N workers do not
# start N x cores threads (HistGradientBoosting uses every core by default)
if os.environ.get("PYTEST_XDIST_WORKER"):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))


@pytest.fixture
def reg_df():
    from fixtures import clean_regression

    return clean_regression()


@pytest.fixture
def imb_df():
    from fixtures import imbalanced

    return imbalanced()


@pytest.fixture
def messy_df():
    """Several hazards at once: id, leak, missing, datetime, text, high-cardinality, string labels."""
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "age": rng.integers(18, 90, n),
        "x": rng.lognormal(size=n),
        "city": rng.choice(list("abcde"), n),
        "zip": [f"z{i % 300}" for i in range(n)],
        "ts": pd.date_range("2020", periods=n, freq="h"),
        "note": [f"free text comment number {i} written by an agent" for i in range(n)],
    })
    df.loc[: n // 3, "x"] = np.nan
    df["y"] = np.where(rng.random(n) < 0.15, "yes", "no")
    df["leak"] = (df.y == "yes").astype(float) + rng.normal(0, 0.01, n)
    return df
