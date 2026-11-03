"""Run the benchmark and publish every result, including failures.

    python benchmarks/run.py              # detection benchmark (fast)
    python benchmarks/run.py --verify     # + experiment outcomes on real datasets (slow)

Writes benchmarks/results/<manifest>.json and .md, and refreshes
prefit/data/track_record.json, which reports cite as "rule detection pass
rate on benchmark <version>: k/n checks". Nothing is filtered: a failing case stays
in the results and lowers the rule's track record.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fixtures import CASES, MANIFEST_VERSION  # noqa: E402

import prefit as pf  # noqa: E402


def run_case(case) -> dict:
    df = case.make()
    rep = pf.analyze(df, case.target, task=case.task, **case.kwargs)
    by_rule: dict[str, set[str]] = {}
    for f in rep.details.findings:
        by_rule.setdefault(f.rule_id, set()).add(f.trigger.value)
    checks = []
    for rid in case.must_fire:
        checks.append({"rule": rid, "expect": "fired", "ok": "fired" in by_rule.get(rid, set()),
                       "got": sorted(by_rule.get(rid, set())) or ["none"]})
    for rid in case.must_not_fire:
        checks.append({"rule": rid, "expect": "none", "ok": rid not in by_rule,
                       "got": sorted(by_rule.get(rid, set())) or ["none"]})
    for rid in case.uncertain:
        checks.append({"rule": rid, "expect": "uncertain", "ok": by_rule.get(rid) == {"uncertain"},
                       "got": sorted(by_rule.get(rid, set())) or ["none"]})
    if case.no_split_overlap:
        for name, got in split_overlaps(rep, df).items():
            checks.append({"rule": name, "expect": "overlap 0", "ok": got == 0, "got": [f"overlap {got}"]})
    return {"case": case.name, "source": case.source, "rows": len(df), "task": rep.task.value,
            "ok": all(c["ok"] for c in checks), "checks": checks}


def split_overlaps(rep, df) -> dict[str, int]:
    """Row hashes found on both sides of the holdout split, and of each exported CV fold (summed)."""
    from prefit.export import build_cv
    from prefit.fingerprint import row_hashes

    dev, hold = rep.holdout(df)
    out = {"holdout_split": len(set(row_hashes(dev)) & set(row_hashes(hold)))}
    cv = build_cv(rep.decision("d.cv").step)
    h = row_hashes(dev)
    X, y = dev.drop(columns=rep.target), dev[rep.target]
    out["cv_split"] = sum(len(set(h[tr]) & set(h[te])) for tr, te in cv.split(X, y))
    return out


def verify_real() -> list[dict]:
    from sklearn import datasets

    out = []
    for name, task in (("load_breast_cancer", "classification"), ("load_diabetes", "regression"),
                       ("load_wine", "classification")):
        df = getattr(datasets, name)(as_frame=True).frame
        t0 = time.perf_counter()
        rep = pf.analyze(df, "target", task=task, verify=True, budget=pf.Budget(max_experiments=6))
        out.append({"dataset": f"sklearn:{name}", "seconds": round(time.perf_counter() - t0, 1),
                    "dev_metric": rep.metric_plan.dev_metric,
                    "decisions": {d.id: d.status.value for d in rep.decisions if d.cls.value == "conditional"},
                    "experiments": [{"decision": e.decision_id, "family": e.family, "mean_diff": e.mean_diff,
                                     "ci": e.ci, "p_holm": e.p_adjusted, "outcome": e.outcome}
                                    for e in rep.experiments]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    results = [run_case(c) for c in CASES]
    rules: dict[str, dict[str, int]] = {}
    for r in results:
        for c in r["checks"]:
            s = rules.setdefault(c["rule"], {"checks": 0, "passed": 0})
            s["checks"] += 1
            s["passed"] += int(c["ok"])
    payload = {
        "manifest_version": MANIFEST_VERSION, "prefit_version": pf.__version__,
        "python": platform.python_version(),
        "protocol": ("Each case runs pf.analyze with default options (plus the case's kwargs) and seed 0. A check "
                     "passes when the rule's finding matches the expectation: fired / none / uncertain (near "
                     "threshold). Split checks (holdout_split, cv_split) count exact row hashes found on both "
                     "sides of the internal holdout and of every exported CV fold; they pass at 0."),
        "cases": results, "rules": rules,
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    result_path = out_dir / f"{MANIFEST_VERSION}.json"
    if a.verify:
        payload["verification"] = verify_real()
    elif result_path.exists():
        # keep the committed experiment section when only the detection benchmark is re-run
        previous = json.loads(result_path.read_text())
        if "verification" in previous:
            payload["verification"] = previous["verification"]
            payload["verification_from"] = previous.get("verification_from", previous.get("prefit_version"))
    result_path.write_text(json.dumps(payload, indent=2, default=str))
    md = [f"# Benchmark {MANIFEST_VERSION} (prefit {pf.__version__})", "", payload["protocol"], "",
          "Detection pass rate on known-hazard fixtures. This measures whether each rule fires (or stays silent) "
          "where the fixture says it should; it is not a claim about accuracy on real-world data.", "",
          "| rule | passed / checks |", "|---|---|"]
    md += [f"| `{k}` | {v['passed']}/{v['checks']} |" for k, v in sorted(rules.items())]
    md += ["", "| case | source | rows | ok | failed checks |", "|---|---|---|---|---|"]
    for r in results:
        failed = "; ".join(f"{c['rule']}: expected {c['expect']}, got {','.join(c['got'])}"
                           for c in r["checks"] if not c["ok"])
        md.append(f"| {r['case']} | {r['source']} | {r['rows']} | {'yes' if r['ok'] else 'NO'} | {failed} |")
    if "verification" in payload:
        md += ["", "## Experiment outcomes on real data (proxy models, development evidence)", ""]
        for v in payload["verification"]:
            md.append(f"**{v['dataset']}** ({v['dev_metric']}, {v['seconds']} s)")
            md += [f"- {e['decision']} · {e['family']}: diff {e['mean_diff']:+.4g}, Holm p={e['p_holm']:.3g} -> "
                   f"{e['outcome']}" for e in v["experiments"]] or ["- no conditional decisions fired"]
    (out_dir / f"{MANIFEST_VERSION}.md").write_text("\n".join(md) + "\n")
    tr = {"manifest_version": MANIFEST_VERSION, "rules": rules}
    (ROOT / "prefit" / "data").mkdir(exist_ok=True)
    (ROOT / "prefit" / "data" / "track_record.json").write_text(json.dumps(tr, indent=2))
    n_ok = sum(r["ok"] for r in results)
    print(f"{n_ok}/{len(results)} cases passed")
    for r in results:
        if not r["ok"]:
            print("FAILED", r["case"], [c for c in r["checks"] if not c["ok"]])
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
