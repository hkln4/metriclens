"""Run the benchmark and publish every result, including failures.

    python benchmarks/run.py              # detection benchmark (fast)
    python benchmarks/run.py --verify     # + experiment outcomes on real datasets (slow)

Writes benchmarks/results/<manifest>.json and .md, and refreshes
metriclens/data/track_record.json, which reports cite as "rule detection pass
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

import metriclens as ml  # noqa: E402


def run_case(case) -> dict:
    df = case.make()
    rep = ml.analyze(df, case.target, task=case.task, **case.kwargs)
    by_rule: dict[str, set[str]] = {}
    for f in rep.findings:
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
    return {"case": case.name, "source": case.source, "rows": len(df), "task": rep.task.value,
            "ok": all(c["ok"] for c in checks), "checks": checks}


def verify_real() -> list[dict]:
    from sklearn import datasets

    out = []
    for name, task in (("load_breast_cancer", "classification"), ("load_diabetes", "regression"),
                       ("load_wine", "classification")):
        df = getattr(datasets, name)(as_frame=True).frame
        t0 = time.perf_counter()
        rep = ml.analyze(df, "target", task=task, verify=True, budget=ml.Budget(max_experiments=6))
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
        "manifest_version": MANIFEST_VERSION, "metriclens_version": ml.__version__,
        "python": platform.python_version(),
        "protocol": ("Each case runs ml.analyze with default options and seed 0. A check passes when the rule's "
                     "finding matches the expectation: fired / none / uncertain (near threshold)."),
        "cases": results, "rules": rules,
    }
    if a.verify:
        payload["verification"] = verify_real()
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{MANIFEST_VERSION}.json").write_text(json.dumps(payload, indent=2, default=str))
    md = [f"# Benchmark {MANIFEST_VERSION} (metriclens {ml.__version__})", "", payload["protocol"], "",
          "Detection pass rate on known-hazard fixtures. This measures whether each rule fires (or stays silent) "
          "where the fixture says it should; it is not a claim about accuracy on real-world data.", "",
          "| rule | passed / checks |", "|---|---|"]
    md += [f"| `{k}` | {v['passed']}/{v['checks']} |" for k, v in sorted(rules.items())]
    md += ["", "| case | source | rows | ok | failed checks |", "|---|---|---|---|---|"]
    for r in results:
        failed = "; ".join(f"{c['rule']}: expected {c['expect']}, got {','.join(c['got'])}"
                           for c in r["checks"] if not c["ok"])
        md.append(f"| {r['case']} | {r['source']} | {r['rows']} | {'yes' if r['ok'] else 'NO'} | {failed} |")
    if a.verify:
        md += ["", "## Experiment outcomes on real data (proxy models, development evidence)", ""]
        for v in payload["verification"]:
            md.append(f"**{v['dataset']}** ({v['dev_metric']}, {v['seconds']} s)")
            md += [f"- {e['decision']} · {e['family']}: diff {e['mean_diff']:+.4g}, Holm p={e['p_holm']:.3g} -> "
                   f"{e['outcome']}" for e in v["experiments"]] or ["- no conditional decisions fired"]
    (out_dir / f"{MANIFEST_VERSION}.md").write_text("\n".join(md) + "\n")
    tr = {"manifest_version": MANIFEST_VERSION, "rules": rules}
    (ROOT / "metriclens" / "data").mkdir(exist_ok=True)
    (ROOT / "metriclens" / "data" / "track_record.json").write_text(json.dumps(tr, indent=2))
    n_ok = sum(r["ok"] for r in results)
    print(f"{n_ok}/{len(results)} cases passed")
    for r in results:
        if not r["ok"]:
            print("FAILED", r["case"], [c for c in r["checks"] if not c["ok"]])
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
