"""End-to-end metriclens flow on California Housing.

    python examples/quickstart.py

1. analyze: profile the dev rows, derive decisions (holdout reserved first)
2. answer:  give the context only you know (here: which errors matter)
3. verify:  bounded experiments for the conditional decisions, with YOUR model
4. export:  a sklearn Pipeline holding only applied/confirmed/verified steps
5. review:  score once on the untouched holdout and check the assumptions
Writes examples/output/california_report.html.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor

import metriclens as ml

OUT = Path(__file__).parent / "output"


def main() -> None:
    df = fetch_california_housing(as_frame=True).frame
    model = HistGradientBoostingRegressor(random_state=0)

    report = ml.analyze(df, target="MedHouseVal", estimator=model,
                        context="house-price estimates used to prioritise field appraisals")
    print(report)                                            # roadmap + open questions

    report = report.answer({"task": "regression", "metric_priority": "typical_error"})
    report = report.verify(df, budget=ml.Budget(max_experiments=4))

    built = report.export()
    print(built.describe())

    dev, holdout = report.split(df)
    fitted = built.model.fit(dev[built.feature_columns], dev["MedHouseVal"])
    pred = fitted.predict(holdout[built.feature_columns])
    rev = ml.review(holdout["MedHouseVal"], pred, report, segments=holdout["HouseAge"].gt(30).map(
        {True: "older than 30y", False: "30y or newer"}).rename("house_age"))
    print(rev)

    OUT.mkdir(exist_ok=True)
    report.to_html(str(OUT / "california_report.html"), review=rev)
    print(f"wrote {OUT / 'california_report.html'}")


if __name__ == "__main__":
    main()
