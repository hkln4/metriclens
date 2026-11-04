"""End-to-end prefit flow on California Housing.

    python examples/quickstart.py

1. analyze:  set a holdout aside first, profile the rest, derive decisions
2. answer:   the context only you know (which errors matter), by keyword
3. verify:   bounded experiments for the conditional decisions, with YOUR model
4. export:   a model recipe holding only applied / confirmed / verified steps; fit it
5. review:   score once on the untouched holdout and check the assumptions

The report keeps a reference to the frame (fingerprint-checked on every use), so
no step needs the DataFrame again. Writes examples/output/california_report.html.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor

import prefit as pf

OUT = Path(__file__).parent / "output"


def main() -> None:
    df = fetch_california_housing(as_frame=True).frame
    model = HistGradientBoostingRegressor(random_state=0)

    report = pf.analyze(df, "MedHouseVal", estimator=model, holdout=0.2,
                        context="house-price estimates used to prioritise field appraisals")
    print(report)                                         # status, the lines to run next, the roadmap

    report = report.answer(task="regression", metric="mae").verify(budget=pf.Budget(max_experiments=4))
    print(report.export().describe())                     # what is in the recipe, what is out, and why

    fitted = report.export().fit()
    _, test = report.holdout()
    age = test["HouseAge"].gt(30).map({True: "older than 30y", False: "30y or newer"}).rename("house_age")
    review = report.review(fitted, segments=age.reset_index(drop=True))
    print(review)

    OUT.mkdir(exist_ok=True)
    report.to_html(str(OUT / "california_report.html"), review=review)
    print(f"wrote {OUT / 'california_report.html'}")


if __name__ == "__main__":
    main()
