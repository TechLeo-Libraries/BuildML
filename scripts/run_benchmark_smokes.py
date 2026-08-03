#!/usr/bin/env python3
"""Run benchmark smoke scripts under ``benchmarks/`` for CI.

Each script is invoked as ``python benchmarks/<domain>/<script>.py``. Scripts
are written to skip optional industry backends when extras are missing; this
runner only fails on non-zero exit codes (hard errors or metric floors on
core paths such as RAG hashing).

Discovery uses ``benchmarks/*/*.py`` (26 scripts). Domains:

  activelearning, anomaly, automl, causal, cbr, federated, graph, kg,
  metalearning, multitask, nlp, online, optimize, probabilistic, rag, ranking,
  recommenders, rl, semisupervised, ssl, symbolic, synthetic, tda, timeseries,
  unsupervised

Linux-only in CI — Torch and several industry wheels are flaky on Windows
runners; see ``.github/workflows/ci.yml`` ``benchmarks`` job comments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_scripts() -> list[Path]:
    scripts = sorted(
        p
        for p in ROOT.glob("benchmarks/*/*.py")
        if p.is_file() and not p.name.startswith("_")
    )
    if not scripts:
        raise SystemExit(f"No benchmark scripts found under {ROOT / 'benchmarks'}")
    return scripts


def run_script(script: Path, *, verbose: bool) -> int:
    rel = script.relative_to(ROOT)
    print(f"=== {rel} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
    )
    if verbose and proc.returncode != 0:
        print(f"FAILED: {rel} (exit {proc.returncode})", file=sys.stderr)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BuildML benchmark smoke scripts")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stderr hints when a script fails",
    )
    args = parser.parse_args(argv)

    scripts = discover_scripts()
    failures: list[tuple[Path, int]] = []
    for script in scripts:
        code = run_script(script, verbose=args.verbose)
        if code != 0:
            failures.append((script, code))

    print(f"\nRan {len(scripts)} benchmark smoke script(s).", flush=True)
    if failures:
        print("Failures:", file=sys.stderr)
        for path, code in failures:
            print(f"  {path.relative_to(ROOT)} -> exit {code}", file=sys.stderr)
        return 1
    print("All benchmark smokes passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
