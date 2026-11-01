"""End-to-end metriclens flow on California Housing.

    python examples/quickstart.py

1. split:    set the test data aside before anything is computed
2. analyze:  profile the development data, derive decisions
3. answer:   give the context only you know (which errors matter)
4. verify:   bounded experiments for the conditional decisions, with YOUR model
5. register: record the test set in the report (metadata only)
6. export:   a model recipe holding only applied / confirmed / verified steps
7. review:   score once on the untouched test set and check the assumptions
Writes examples/output/california_report.html.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor

import metriclens as ml

OUT = Path(__file__).parent / "output"
TARGET = "MedHouseVal"


def main() -> None:
    df = fetch_california_housing(as_frame=True).frame
    model = HistGradientBoostingRegressor(random_state=0)

    dev_df, test_df = ml.split(df, TARGET, test_size=0.2, strategy="random", random_state=0)

    report = ml.analyze(dev_df, target=TARGET, estimator=model,
                        context="house-price estimates used to prioritise field appraisals")
    print(report)                                            # roadmap + open questions

    report = report.answer({"task": "regression", "metric_priority": "typical_error"})
    report = report.verify(dev_df, budget=ml.Budget(max_experiments=4))
    report = report.register_external_holdout(test_df, dev_df=dev_df, name="test set")

    built = report.export()
    print(built.describe())

    fitted = built.model.fit(dev_df[built.feature_columns], dev_df[TARGET])
    pred = fitted.predict(test_df[built.feature_columns])
    age = test_df["HouseAge"].gt(30).map({True: "older than 30y", False: "30y or newer"}).rename("house_age")
    rev = ml.review(test_df[TARGET], pred, report, segments=age)
    print(rev)

    OUT.mkdir(exist_ok=True)
    report.to_html(str(OUT / "california_report.html"), review=rev)
    print(f"wrote {OUT / 'california_report.html'}")


if __name__ == "__main__":
    main()
