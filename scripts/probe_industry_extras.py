#!/usr/bin/env python3
"""Probe which industry / optional extras import cleanly on this platform.

Prints a table of extra name → status (ok / unavailable). Always exits ``0`` —
this is an honesty / CI artifact helper, not a hard gate. Use after
``pip install -e ".[dev]"`` (and optionally individual ``*-industry`` extras).

Environment markers in ``pyproject.toml`` already skip known-broken wheels
(for example LightFM on Windows, giotto-tda on Py3.13). This probe reports what
actually imports *here* so install docs never overclaim.

Examples
--------
::

    python scripts/probe_industry_extras.py
    python scripts/probe_industry_extras.py --json
    python scripts/probe_industry_extras.py --artifact industry-probe.json
    python scripts/probe_industry_extras.py --markdown industry-probe.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Modules whose import can hard-crash the parent process on some Windows installs.
_SUBPROCESS_MODULES = frozenset({"torch", "sentence_transformers", "pykeen"})

# Extra marker → modules that prove the adapter surface imported.
# Keep aligned with pyproject optional-dependencies *-industry groups.
PROBES: dict[str, tuple[str, ...]] = {
    "automl-industry": ("flaml", "autogluon.tabular"),
    "anomaly-industry": ("pyod",),
    "ranking-industry": ("lightgbm", "xgboost", "catboost"),
    "recommenders-industry": ("implicit",),
    "recommenders-lightfm": ("lightfm",),
    "tda-industry": ("giotto_tda", "ripser"),
    "causal-industry": ("dowhy", "econml"),
    "online-industry": ("river",),
    "metalearning-industry": ("learn2learn",),
    "rl-industry": ("stable_baselines3",),
    "nlp-industry": ("spacy", "sentence_transformers"),
    "synthetic-industry": ("sdv",),
    "symbolic-industry": ("skope_rules", "imodels"),
    "probabilistic-industry": ("mapie",),
    "federated-industry": ("flwr",),
    "kg-industry": ("pykeen",),
    "activelearning-industry": ("skactiveml",),
    "semisupervised-industry": ("lightgbm", "xgboost"),
    "multitask-industry": ("lightgbm", "xgboost", "catboost"),
    "cbr-industry": ("hnswlib",),
    "optimize-industry": ("optuna", "pulp", "ortools"),
    "timeseries-ml": ("neuralforecast",),
    "timeseries-prophet": ("prophet",),
    "torch": ("torch",),
    "rag": ("sentence_transformers",),
}

# Documented platform / version fragility (mirrors pyproject + industry_markers).
KNOWN_FRAGILE: dict[str, str] = {
    "lightfm": "marker: python_version < '3.13' and sys_platform != 'win32'",
    "giotto_tda": "marker: python_version < '3.13'",
    "learn2learn": "marker: python_version < '3.13'",
    "skope_rules": "marker: python_version < '3.13'",
    "neuralforecast": "marker: python_version < '3.13'",
    "autosklearn": "often Linux-only; not pinned in BuildML extras",
    "autogluon.tabular": "heavy; may fail on constrained CI images",
}


@dataclass(frozen=True)
class ProbeRow:
    extra: str
    module: str
    status: str
    detail: str
    known_fragile: str
    marker_allows_install: bool | None
    skipped_by_marker: bool


def _marker_flags(module: str) -> tuple[bool | None, bool]:
    try:
        from buildml.core.industry_markers import marker_allows, marker_reason
    except Exception:  # noqa: BLE001 — probe must never fail hard
        return None, False
    reason = marker_reason(module)
    if reason is None:
        return None, False
    allows = marker_allows(module)
    return allows, (not allows)


def _probe_via_subprocess(module: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            timeout=60,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "unavailable", f"{type(exc).__name__}: {exc}"
    if completed.returncode == 0:
        return "ok", ""
    # Windows access-violation / hard kill often surfaces as negative exit codes.
    if completed.returncode < 0 or completed.returncode > 0xC0000000:
        return "unavailable", f"hard-exit {completed.returncode} (likely native crash)"
    err = (completed.stderr or completed.stdout or "").strip().replace("\n", " ")
    if len(err) > 160:
        err = err[:157] + "..."
    return "unavailable", err or f"exit {completed.returncode}"


def _probe_module(module: str) -> tuple[str, str]:
    # On Windows, probe every module out-of-process: broken native wheels
    # (Torch / CUDA / AV) can kill the parent with STATUS_ACCESS_VIOLATION.
    if sys.platform == "win32" or module in _SUBPROCESS_MODULES:
        return _probe_via_subprocess(module)
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 — honesty probe, never gate
        name = type(exc).__name__
        msg = str(exc).strip().replace("\n", " ")
        if len(msg) > 160:
            msg = msg[:157] + "..."
        return "unavailable", f"{name}: {msg}"
    return "ok", ""


def run_probes() -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    for extra, modules in sorted(PROBES.items()):
        for module in modules:
            status, detail = _probe_module(module)
            allows, skipped = _marker_flags(module)
            rows.append(
                ProbeRow(
                    extra=extra,
                    module=module,
                    status=status,
                    detail=detail,
                    known_fragile=KNOWN_FRAGILE.get(module, ""),
                    marker_allows_install=allows,
                    skipped_by_marker=skipped,
                )
            )
    return rows


def _write_markdown(path: Path, meta: dict, rows: list[ProbeRow], summary: dict) -> None:
    lines = [
        "# BuildML industry extras probe",
        "",
        f"- Python: `{meta['python']}`",
        f"- Platform: `{meta['platform']}`",
        f"- System: `{meta['system']}` / `{meta['machine']}`",
        f"- Implementation: `{meta['implementation']}`",
        "",
        f"Summary: **{summary['ok']}** ok / **{summary['unavailable']}** "
        f"unavailable / **{summary['total']}** total "
        f"(marker-skipped reported: **{summary['marker_skipped']}**).",
        "",
        "Informational only — missing upstream wheels must not fail CI.",
        "",
        "| Extra | Module | Status | Marker skip | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        detail = row.detail.replace("|", "\\|") if row.detail else ""
        fragile = row.known_fragile.replace("|", "\\|")
        marker = "yes" if row.skipped_by_marker else ("no" if row.marker_allows_install else "n/a")
        note = detail or fragile
        lines.append(
            f"| `{row.extra}` | `{row.module}` | {row.status} | {marker} | {note} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    parser.add_argument(
        "--artifact",
        type=str,
        default="",
        help="Write the same JSON payload to this path (CI upload helper).",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default="",
        help="Write a Markdown summary table to this path (CI artifact helper).",
    )
    args = parser.parse_args(argv)

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform_tags": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_full": sys.version.split()[0],
        },
    }
    rows = run_probes()
    summary = {
        "ok": sum(1 for r in rows if r.status == "ok"),
        "unavailable": sum(1 for r in rows if r.status == "unavailable"),
        "total": len(rows),
        "marker_skipped": sum(1 for r in rows if r.skipped_by_marker),
    }
    payload = {
        "meta": meta,
        "probes": [asdict(r) for r in rows],
        "summary": summary,
        "honesty": (
            "Informational only. Missing wheels must not fail CI. "
            "Capability matrices report the same availability at runtime via "
            "platform_markers / skipped_by_marker fields."
        ),
    }

    if args.artifact:
        Path(args.artifact).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.artifact}")

    if args.markdown:
        _write_markdown(Path(args.markdown), meta, rows, summary)
        print(f"wrote {args.markdown}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if (args.artifact or args.markdown) and not args.json:
        return 0

    print(
        f"BuildML industry extra probe — Python {meta['python']} on "
        f"{meta['system']} / {meta['platform']}"
    )
    print("Status is informational; missing wheels do not fail this script.\n")
    width_extra = max(len(r.extra) for r in rows)
    width_mod = max(len(r.module) for r in rows)
    for row in rows:
        fragile = f"  [{row.known_fragile}]" if row.known_fragile else ""
        marker = "  [marker-skip]" if row.skipped_by_marker else ""
        line = (
            f"{row.extra:<{width_extra}}  {row.module:<{width_mod}}  "
            f"{row.status:<12}  {row.detail}{fragile}{marker}"
        )
        print(line.rstrip())
    print(
        f"\n{summary['ok']}/{summary['total']} "
        "probed modules importable in this environment "
        f"({summary['marker_skipped']} marker-skipped)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
