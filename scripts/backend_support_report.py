"""Execute real backend contracts and record environment-specific support evidence."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUPS = {
    "engines": (["polars", "duckdb"], ["tests/unit/test_engines.py", "tests/unit/test_duckdb_connection_lifecycle.py", "tests/unit/test_engine_persistence_contracts.py"]),
    "forecasting": (["statsmodels"], ["tests/unit/test_forecasting_r3_depth.py"]),
    "resampling": (["imblearn"], ["tests/unit/test_imbalance_harden.py"]),
    "rl": (["gymnasium"], ["tests/unit/test_rl_tabular.py"]),
    "tda": (["ripser", "persim"], ["tests/unit/test_tda_slice.py", "tests/unit/test_tda_m2_depth.py"]),
    "graph": (["networkx"], ["tests/unit/test_graph_catalog.py", "tests/unit/test_graph_slice.py",
                              "tests/unit/test_graph_m2_depth.py"]),
    "mapie": (["mapie"], ["tests/unit/test_mapie_real_contract.py", "tests/unit/test_mapie_alpha_contract.py",
                         "tests/unit/test_executor_domain_contracts.py::test_agent_probabilistic_honors_requested_real_mapie_backend"]),
    "ann": (["hnswlib"], ["tests/unit/test_isolated_ann.py", "tests/unit/test_cbr_industry_depth.py"]),
    "serving": (["fastapi", "httpx"], ["tests/unit/test_serving_production.py", "tests/unit/test_schema_contract.py",
                                      "tests/integration/test_real_http_serving.py", "tests/unit/test_trust_boundary_contracts.py"]),
}


def run_group(name: str, directory: Path) -> dict:
    """Require dependencies, isolate faults, and preserve every skipped-test reason."""
    modules, tests = GROUPS[name]
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        return {"group": name, "status": "missing_dependency", "missing": missing}
    # Import in a child: a discoverable wheel can still be broken or lack its
    # transitive dependencies, and a native import can crash the process.
    probe_code = "import importlib,sys;[importlib.import_module(name) for name in sys.argv[1:]]"
    try:
        probe = subprocess.run([sys.executable, "-c", probe_code, *modules], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"group": name, "status": "dependency_import_timeout"}
    (directory / f"{name}-dependency-import.log").write_text(probe.stdout + probe.stderr, encoding="utf-8")
    if probe.returncode:
        return {"group": name, "status": "dependency_import_failed", "returncode": probe.returncode,
                "reason": probe.stderr[-2000:]}
    xml_path = directory / f"{name}.xml"
    command = [sys.executable, "-m", "pytest", *tests, "-q", "-rs", "-o", "addopts=",
               "--basetemp", str(directory / f"temp-{name}"), "--junitxml", str(xml_path)]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        (directory / f"{name}.log").write_text(output, encoding="utf-8")
        status = "passed" if result.returncode == 0 else "failed"
        if result.returncode < 0 or "access violation" in output.lower():
            status = "crashed"
        skipped = []
        passed = failed = 0
        if xml_path.exists():
            for case in ET.parse(xml_path).iter("testcase"):
                skip = case.find("skipped")
                if skip is not None:
                    skipped.append({"test": case.get("name"), "reason": skip.get("message", "")})
                elif case.find("failure") is not None or case.find("error") is not None:
                    failed += 1
                else:
                    passed += 1
        if status == "passed" and passed == 0:
            status = "no_executed_tests"
        return {"group": name, "status": status, "passed": passed, "failed": failed,
                "skipped": skipped, "seconds": time.perf_counter() - started,
                "command": command, "log": str(directory / f"{name}.log")}
    except subprocess.TimeoutExpired:
        return {"group": name, "status": "timeout", "seconds": time.perf_counter() - started}


def main() -> int:
    """Produce evidence without equating missing dependencies with passing support."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    parser.add_argument("--out", type=Path, default=Path("artifacts/backend-support"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    from buildml import __version__

    checked_at = datetime.now(timezone.utc).isoformat()
    results = []
    for name in args.groups:
        row = run_group(name, args.out)
        results.append(row)
        print(f"{name}: {row['status']} ({row.get('passed', 0)} passed)", flush=True)
    payload = {"checked_at": checked_at, "buildml_source_version": __version__,
               "source_root": str(ROOT), "python": sys.version, "platform": platform.platform(),
               "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
               "results": results, "limitations": "Local tested configurations only; skips are listed, not counted as support."}
    (args.out / "support.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Backend support evidence", "", f"Platform: {platform.platform()}; Python {platform.python_version()}",
             "", "| Group | Status | Passed | Skipped |", "|---|---|---:|---:|"]
    lines += [f"| {r['group']} | {r['status']} | {r.get('passed', 0)} | {len(r.get('skipped', []))} |" for r in results]
    lines += ["", "The JSON contains exact versions, commands and skip reasons. Missing/failed/crashed groups are unverified support."]
    (args.out / "support.md").write_text("\n".join(lines), encoding="utf-8")
    return int(any(r["status"] != "passed" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
