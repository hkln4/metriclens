"""Command line: `prefit report data.csv --target y --out report.html`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def _read(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if path.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prefit", description="Decision-driven EDA for tabular supervised learning.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="analyze a CSV/Parquet file and write an HTML report")
    rp.add_argument("data")
    rp.add_argument("--target", required=True)
    rp.add_argument("--task", choices=["classification", "regression"])
    rp.add_argument("--groups", help="column identifying entities that must not be split across folds")
    rp.add_argument("--time-col", help="column defining time order")
    rp.add_argument("--holdout", type=float, default=None,
                    help="ask prefit to set aside this fraction before profiling (default: none)")
    rp.add_argument("--answer", action="append", default=[], metavar="QUESTION=OPTION",
                    help="answer a report question, e.g. metric_priority=miss_costly (repeatable)")
    rp.add_argument("--verify", action="store_true", help="run the bounded experiments (proxy models)")
    rp.add_argument("--max-experiments", type=int, default=6)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--out", default="prefit_report.html")
    rp.add_argument("--json", help="also write the report as JSON")
    rp.add_argument("--plotlyjs", choices=["cdn", "inline", "none"], default="cdn")
    a = ap.parse_args(argv)

    import prefit as pf

    answers = {}
    for item in a.answer:
        if "=" not in item:
            ap.error(f"--answer expects QUESTION=OPTION, got {item!r}")
        k, v = item.split("=", 1)
        answers[k] = v
    df = _read(a.data)
    report = pf.analyze(df, a.target, task=a.task, groups=a.groups, time_col=a.time_col,
                        holdout=a.holdout, answers=answers, verify=a.verify,
                        budget=pf.Budget(max_experiments=a.max_experiments), seed=a.seed)
    report.to_html(a.out, plotlyjs=a.plotlyjs)
    if a.json:
        report.to_json(a.json)
    print(report.to_text())
    print(f"HTML report written to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
