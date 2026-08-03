"""Run the full pytest suite with coverage and record the measured TOTAL.

Mirrors the CI ``test`` job coverage intent (``pytest tests --cov=buildml``).
On Windows, optional native extensions (hnswlib, Torch DLLs, etc.) can
access-violate the parent process. This runner isolates each test *module*
in its own subprocess so one SIGSEGV cannot erase the rest of the suite,
then ``coverage combine``s into a single TOTAL.

Usage (repo root)::

    python scripts/run_full_coverage.py
    python scripts/run_full_coverage.py --update-ratchet
    python scripts/run_full_coverage.py --mode monolith   # CI-shaped single process
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATCHET = ROOT / "scripts" / "coverage_ratchet.json"
REPORT_JSON = ROOT / "coverage-full.json"
REPORT_TXT = ROOT / "coverage-full.log"
CRASH_JSON = ROOT / "coverage-full-crashes.json"
COV_DIR = ROOT / ".coverage_full_parts"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _discover_test_modules() -> list[Path]:
    tests_root = ROOT / "tests"
    modules = sorted(
        p
        for p in tests_root.rglob("test_*.py")
        if p.is_file() and ".pytest_tmp" not in p.parts
    )
    return modules


def _part_name(module: Path) -> str:
    rel = module.relative_to(ROOT).as_posix()
    return _SAFE_NAME.sub("_", rel)


def _clean_coverage_artifacts() -> None:
    for path in ROOT.glob(".coverage*"):
        if path.is_file():
            path.unlink()
    if COV_DIR.exists():
        shutil.rmtree(COV_DIR)
    COV_DIR.mkdir(parents=True, exist_ok=True)


def _run_module(module: Path, *, env: dict[str, str], fail_under: float) -> dict:
    rel = module.relative_to(ROOT).as_posix()
    part = COV_DIR / f".coverage.{_part_name(module)}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        rel,
        "-q",
        "--tb=line",
        "--cov=buildml",
        "--cov-report=",
        f"--cov-fail-under={fail_under}",
    ]
    run_env = env.copy()
    run_env["COVERAGE_FILE"] = str(part)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=run_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        rc = int(completed.returncode)
        out = (completed.stdout or "") + (completed.stderr or "")
        crashed = rc < 0 or (
            "Windows fatal exception" in out or "access violation" in out.lower()
        )
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = f"TIMEOUT after 600s\n{(exc.stdout or '')}{(exc.stderr or '')}"
        crashed = True
    except OSError as exc:
        rc = 1
        out = f"OSError: {exc}"
        crashed = True
    elapsed = time.perf_counter() - started
    # Keep a short tail for crash diagnosis without ballooning the log.
    tail = "\n".join(out.splitlines()[-40:])
    return {
        "module": rel,
        "returncode": rc,
        "crashed": crashed,
        "elapsed_s": round(elapsed, 2),
        "coverage_part": str(part) if part.is_file() else None,
        "tail": tail,
    }


def _run_monolith(*, env: dict[str, str], fail_under: float | None) -> int:
    """Single-process suite (Linux CI shape)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=line",
        "--cov=buildml",
        "--cov-report=term",
        f"--cov-report=json:{REPORT_JSON}",
    ]
    if fail_under is not None:
        cmd.append(f"--cov-fail-under={fail_under}")
    print("+", " ".join(cmd), flush=True)
    with REPORT_TXT.open("w", encoding="utf-8") as log:
        print("+", " ".join(cmd), file=log, flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return int(proc.wait())


def _combine_and_report(*, fail_under: float) -> tuple[float, int, int]:
    """Combine part files and write JSON + terminal report."""
    # Point combine at the parts directory via COVERAGE_FILE prefix convention:
    # coverage combine accepts explicit data files.
    parts = sorted(COV_DIR.glob(".coverage.*"))
    if not parts:
        raise RuntimeError("no per-module coverage data files were produced")

    combined = ROOT / ".coverage"
    if combined.exists():
        combined.unlink()

    combine_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "combine",
        "--keep",
        *[str(p) for p in parts],
    ]
    print("+", "coverage combine", len(parts), "parts", flush=True)
    subprocess.run(combine_cmd, cwd=ROOT, check=True)

    report_cmd = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "-o",
        str(REPORT_JSON),
    ]
    subprocess.run(report_cmd, cwd=ROOT, check=True)

    # Also emit a term report into the log for humans.
    term = subprocess.run(
        [sys.executable, "-m", "coverage", "report", f"--fail-under={fail_under}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    with REPORT_TXT.open("a", encoding="utf-8") as log:
        log.write("\n===== COMBINED COVERAGE REPORT =====\n")
        log.write(term.stdout or "")
        log.write(term.stderr or "")
    sys.stdout.write(term.stdout or "")
    if term.stderr:
        sys.stderr.write(term.stderr)

    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    percent = float(payload["totals"]["percent_covered"])
    covered = int(payload["totals"]["covered_lines"])
    total = int(payload["totals"]["num_statements"])
    return percent, covered, total


def _update_ratchet(
    *,
    percent: float,
    covered: int,
    total: int,
    mode: str,
    crashed: list[dict],
    command: str,
) -> None:
    ratchet = json.loads(RATCHET.read_text(encoding="utf-8"))
    ratchet["measured"] = {
        "percent_covered": round(percent, 2),
        "covered_lines": covered,
        "num_statements": total,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "mode": mode,
        "command": command,
        "crashed_modules": [c["module"] for c in crashed],
        "crashed_count": len(crashed),
    }
    ratchet["last_full_suite_percent"] = round(percent, 2)
    # Suggested floor: integer percent minus 1pp headroom, never below current
    # one-way ratchet without an explicit human decision to lower.
    suggested = max(0, int(percent) - 1)
    ratchet["suggested_fail_under"] = suggested
    RATCHET.write_text(json.dumps(ratchet, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {RATCHET} with measured={percent:.2f}% suggested_fail_under={suggested}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-ratchet",
        action="store_true",
        help="Write measured TOTAL into scripts/coverage_ratchet.json",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="Coverage fail_under while measuring (default 0 so TOTAL is always written)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "isolated", "monolith"),
        default="auto",
        help="auto=isolated on Windows else monolith; isolated=per-module; monolith=single process",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "isolated" if sys.platform.startswith("win") else "monolith"

    _clean_coverage_artifacts()
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(ROOT / ".coverage")

    with REPORT_TXT.open("w", encoding="utf-8") as log:
        log.write(f"mode={mode} platform={sys.platform} python={sys.version}\n")

    crashed: list[dict] = []
    results: list[dict] = []

    if mode == "monolith":
        rc = _run_monolith(env=env, fail_under=args.fail_under)
        if not REPORT_JSON.is_file():
            print(
                "coverage JSON missing; monolith suite likely hard-crashed. "
                "Re-run with --mode isolated.",
                file=sys.stderr,
            )
            return rc or 1
        payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        percent = float(payload["totals"]["percent_covered"])
        covered = int(payload["totals"]["covered_lines"])
        total = int(payload["totals"]["num_statements"])
        command = "pytest tests --cov=buildml (monolith)"
    else:
        modules = _discover_test_modules()
        print(f"Isolated full-suite: {len(modules)} test modules", flush=True)
        with REPORT_TXT.open("a", encoding="utf-8") as log:
            log.write(f"modules={len(modules)}\n")

        for idx, module in enumerate(modules, start=1):
            rel = module.relative_to(ROOT).as_posix()
            print(f"[{idx}/{len(modules)}] {rel}", flush=True)
            result = _run_module(module, env=env, fail_under=args.fail_under)
            results.append(result)
            status = "CRASH" if result["crashed"] else ("FAIL" if result["returncode"] else "ok")
            print(
                f"  -> {status} rc={result['returncode']} "
                f"{result['elapsed_s']}s cov={bool(result['coverage_part'])}",
                flush=True,
            )
            with REPORT_TXT.open("a", encoding="utf-8") as log:
                log.write(
                    f"{rel}\t{status}\trc={result['returncode']}\t"
                    f"{result['elapsed_s']}s\n"
                )
                if result["crashed"] or result["returncode"]:
                    log.write(result["tail"] + "\n")
            if result["crashed"]:
                crashed.append(result)

        CRASH_JSON.write_text(
            json.dumps(
                {
                    "platform": sys.platform,
                    "python": sys.version.split()[0],
                    "modules": len(modules),
                    "crashed": crashed,
                    "failed_non_crash": [
                        r
                        for r in results
                        if (not r["crashed"]) and r["returncode"] not in (0, 5)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote crash manifest: {CRASH_JSON} ({len(crashed)} hard crashes)")

        percent, covered, total = _combine_and_report(fail_under=args.fail_under)
        command = f"pytest per-module isolated ({len(modules)} modules) + coverage combine"
        # Non-zero if any non-crash test failures (pytest rc 1) — still emit TOTAL.
        failed = [
            r for r in results if (not r["crashed"]) and r["returncode"] not in (0, 5)
        ]
        rc = 1 if failed else 0

    print(f"\nFULL-SUITE COVERAGE: {percent:.1f}% ({covered}/{total} lines)")
    print(f"Wrote {REPORT_JSON} and {REPORT_TXT}")
    if crashed:
        print(f"Hard-crashed modules excluded from their own data only ({len(crashed)}):")
        for item in crashed:
            print(f"  - {item['module']}")

    if args.update_ratchet:
        _update_ratchet(
            percent=percent,
            covered=covered,
            total=total,
            mode=mode,
            crashed=crashed,
            command=command,
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
