"""Real datasets for the benchmarks and the real-data tests, downloaded once and cached.

Cache: $PREFIT_DATA, else ~/.cache/prefit. Nothing here is bundled with the
package; each loader names its source and licence.

    load("wine_red")       UCI Wine Quality, red (Cortez et al., 2009), CC BY 4.0
    load("telco")          IBM Telco Customer Churn sample data (Apache 2.0)
    load("adult")          UCI Adult / Census Income via OpenML (CC BY 4.0)
    load("california")     California Housing via scikit-learn
    load("breast_cancer")  Breast Cancer Wisconsin via scikit-learn
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CACHE = Path(os.environ.get("PREFIT_DATA", Path.home() / ".cache" / "prefit"))


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    target: str
    task: str
    metric: str          # the answer to the metric question used in the benchmark
    source: str


def _download(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.rename(path)
    return path


def load(name: str) -> Dataset:
    if name == "wine_red":
        p = _download("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                      "winequality-red.csv")
        return Dataset(name, pd.read_csv(p, sep=";"), "quality", "regression", "typical_error",
                       "UCI Wine Quality (red), Cortez et al. 2009")
    if name == "telco":
        p = _download("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
                      "Telco-Customer-Churn.csv", "telco-customer-churn.csv")
        return Dataset(name, pd.read_csv(p), "Churn", "classification", "ranking", "IBM Telco Customer Churn")
    if name == "adult":
        path = CACHE / "adult.csv"
        if not path.exists():
            from sklearn.datasets import fetch_openml

            CACHE.mkdir(parents=True, exist_ok=True)
            fetch_openml("adult", version=2, as_frame=True, parser="auto", data_home=str(CACHE / "openml")) \
                .frame.to_csv(path, index=False)
        return Dataset(name, pd.read_csv(path), "class", "classification", "symmetric", "UCI Adult via OpenML")
    if name == "california":
        from sklearn.datasets import fetch_california_housing

        df = fetch_california_housing(as_frame=True, data_home=str(CACHE / "sklearn")).frame
        return Dataset(name, df, "MedHouseVal", "regression", "typical_error", "California Housing (scikit-learn)")
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        return Dataset(name, load_breast_cancer(as_frame=True).frame, "target", "classification", "symmetric",
                       "Breast Cancer Wisconsin (scikit-learn)")
    raise KeyError(name)


NAMES = ("wine_red", "telco", "adult", "california", "breast_cancer")
