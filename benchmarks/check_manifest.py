"""CI guard: the committed benchmark must describe the rules that ship.

    python benchmarks/check_manifest.py

Fails when
- pyproject.toml and prefit/_version.py disagree on the version,
- benchmarks/results/<MANIFEST_VERSION>.json is missing, or was produced by rules
  whose versions differ from the current ones (a rule changed but the benchmark
  was not re-run under a new MANIFEST_VERSION),
- prefit/data/track_record.json cites another manifest.
The fix is always: bump MANIFEST_VERSION in benchmarks/fixtures.py and run
python benchmarks/run.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fixtures import MANIFEST_VERSION  # noqa: E402

import prefit as pf  # noqa: E402
from prefit.rules.engine import rule_versions  # noqa: E402


def problems() -> list[str]:
    out = []
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if version != pf.__version__:
        out.append(f"pyproject.toml says {version}, prefit/_version.py says {pf.__version__}.")
    path = ROOT / "benchmarks" / "results" / f"{MANIFEST_VERSION}.json"
    if not path.exists():
        out.append(f"{path.relative_to(ROOT)} is missing: run python benchmarks/run.py.")
        return out
    res = json.loads(path.read_text())
    recorded = res.get("rule_versions")
    if recorded is None:
        out.append(f"{path.name} records no rule versions (produced before 0.4): bump MANIFEST_VERSION and re-run.")
    else:
        now = rule_versions()
        changed = sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
        if changed:
            out.append(f"rules changed since benchmark {MANIFEST_VERSION} was produced: "
                       + ", ".join(f"{k} {recorded.get(k, '-')} -> {now.get(k, '-')}" for k in changed)
                       + ". Bump MANIFEST_VERSION in benchmarks/fixtures.py and run python benchmarks/run.py.")
    tr = json.loads((ROOT / "prefit" / "data" / "track_record.json").read_text())
    if tr.get("manifest_version") != MANIFEST_VERSION:
        out.append(f"track_record.json cites {tr.get('manifest_version')}, fixtures say {MANIFEST_VERSION}.")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print("✕", p)
    if not found:
        print(f"ok: benchmark {MANIFEST_VERSION} matches the rules of prefit {pf.__version__}")
    sys.exit(1 if found else 0)
