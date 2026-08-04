"""Re-run Tier A / Tier B / Tier C proofs and print a status report.

Usage (from repo root)::

    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all
    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all --tier B
    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all --tier C --skip-existing
    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PROOFS = REPO / "proofs"

TIER_A = [
    # Baseline cohort
    "loan-approval-classical",
    "churn-automl-search",
    "network-intrusion-anomaly",
    "store-sales-forecast",
    "support-kb-rag",
    "movie-recs-collaborative",
    "search-relevance-ltr",
    "kg-biomed-linkpred",
    "credit-tda-shape",
    "semi-label-efficiency",
    "active-labeling-budget",
    "stream-fraud-online",
    "multi-target-underwriting",
    "few-shot-domain-adapt",
    "policy-rules-neuro-symbolic",
    "case-memory-claims",
    "ticket-routing-nlp",
    "cost-sensitive-collections",
    "synthetic-privacy-utility",
    "cluster-customer-segments",
    "ssl-representation-probe",
    "causal-treatment-effect",
    "federated-hospital-sim",
    "graph-fraud-rings",
    "prob-interval-risk",
    "imitation-cartpole-control",
    "tabular-q-frozenlake",
    # Expansion cohort (30)
    "mortgage-default-classical",
    "claim-severity-regression",
    "voting-ensemble-attrition",
    "stacking-credit-risk",
    "blending-payment-risk",
    "torch-tabular-underwrite",
    "torch-text-intent",
    "payment-rail-anomaly",
    "iot-sensor-anomaly",
    "energy-load-forecast",
    "weather-prob-intervals",
    "policy-handbook-rag",
    "catalog-recs-implicit",
    "sponsored-ad-ltr",
    "logistics-kg-linkpred",
    "process-tda-shape",
    "radiology-semi-labels",
    "defect-active-budget",
    "clickstream-online",
    "sku-multitask-retail",
    "coldstart-meta-adapt",
    "compliance-neuro-symbolic",
    "warranty-cbr-memory",
    "campaign-budget-optimize",
    "tabular-synth-utility",
    "sku-embedding-clusters",
    "tabular-ssl-probe",
    "uplift-marketing-causal",
    "edge-fleet-federated",
    "peer-lending-graph",
    # Observational fairness (analysis-only domain)
    "loan-fairness-observational",
    # Real public-dataset cohort (sklearn / OpenML provenance)
    "breast-cancer-classical",
    "diabetes-progression-regression",
    "wine-cluster-segments",
    "adult-fairness-observational",
]

TIER_B = [
    # Baseline cohort
    "aegis-fraud-platform",
    "harbor-demand-desk",
    "atlas-label-studio",
    "pulse-support-copilot",
    "ledger-underwriting-studio",
    "nexus-federated-clinical",
    # Expansion cohort (30)
    "meridian-recs-commerce",
    "helix-knowledge-mesh",
    "prism-shape-monitor",
    "orbit-multitask-hub",
    "quasar-meta-adapt",
    "forge-synth-lab",
    "canyon-segment-studio",
    "vector-control-deck",
    "citadel-ensemble-desk",
    "nova-torch-bench",
    "sentinel-iot-watch",
    "ballast-energy-desk",
    "parchment-policy-copilot",
    "lattice-supply-graph",
    "beacon-label-factory",
    "rivulet-stream-risk",
    "cornerstone-mortgage-suite",
    "apex-uplift-studio",
    "relay-edge-federated",
    "mosaic-warranty-desk",
    "kiln-process-tda",
    "aurora-ad-ranker",
    "compass-catalog-recs",
    "folio-claims-nlp",
    "dynamo-click-lab",
    "scaffold-compliance-ai",
    "terrace-retail-mesh",
    "volt-sensor-fusion",
    "keystone-underwrite-ml",
    "zenith-support-os",
]

TIER_C = list(TIER_A)  # each Tier A may have baseline_industry.py

# Broader core-viable Tier A subset for CI (sklearn/native paths; no torch required).
# Scripts that need optional extras must complete on core or fail the smoke gate
# unless --allow-skip is set (CI never sets that flag).
CI_SMOKE_TIER_A: tuple[str, ...] = (
    "loan-approval-classical",
    "mortgage-default-classical",
    "claim-severity-regression",
    "cluster-customer-segments",
    "network-intrusion-anomaly",
    "payment-rail-anomaly",
    "voting-ensemble-attrition",
    "stacking-credit-risk",
    "blending-payment-risk",
    "stream-fraud-online",
    "clickstream-online",
    "prob-interval-risk",
    "weather-prob-intervals",
    "store-sales-forecast",
    "support-kb-rag",
    "ticket-routing-nlp",
    "policy-rules-neuro-symbolic",
    "compliance-neuro-symbolic",
    "loan-fairness-observational",
    "synthetic-privacy-utility",
    "sku-embedding-clusters",
    # Real public datasets (sklearn offline; Adult falls back if OpenML missing)
    "breast-cancer-classical",
    "diabetes-progression-regression",
    "wine-cluster-segments",
    "adult-fairness-observational",
)

_SKIP_RESULT_STATUSES = frozenset(
    {
        "skipped_missing_extra",
        "skipped",
        "partial",
        "skipped_partial",
    }
)


def _run(script: Path, timeout: int = 600) -> tuple[str, float, str]:
    if not script.is_file():
        return "missing", 0.0, f"no script at {script}"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode == 0:
            return (
                "ok",
                elapsed,
                (proc.stdout or "").strip().splitlines()[-1:][0] if proc.stdout else "",
            )
        err = (proc.stderr or proc.stdout or "").strip()
        return "error", elapsed, err[-500:]
    except subprocess.TimeoutExpired:
        return "timeout", time.perf_counter() - t0, f">{timeout}s"
    except Exception as exc:  # noqa: BLE001
        return "error", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def _read_result_status(marker: Path) -> str | None:
    """Return JSON ``status`` from a proof results marker when present."""
    if not marker.is_file():
        return None
    try:
        payload: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str) and status:
            return status
    return None


def _classify_process_ok(
    *,
    process_status: str,
    result_status: str | None,
    allow_skip: bool,
) -> str:
    """Map process exit + results.json status into harness status.

    ``ok`` means the proof completed. ``skipped`` / ``partial`` are distinct
    from success so CI can fail unexpected soft-skips.
    """
    if process_status != "ok":
        return process_status
    if result_status is None:
        # Older scripts may omit status; treat process-ok as completed.
        return "ok"
    normalized = result_status.strip().lower()
    if normalized in {"completed", "ok", "success"}:
        return "ok"
    if normalized in _SKIP_RESULT_STATUSES:
        if allow_skip:
            return "skipped" if "partial" not in normalized else "partial"
        return "unexpected_skip"
    if normalized == "partial":
        return "partial" if allow_skip else "unexpected_skip"
    # Unknown non-completed status is not success under smoke discipline.
    return "unexpected_skip" if not allow_skip else "skipped"


def main() -> int:
    parser = argparse.ArgumentParser(description="BuildML proofs runner")
    parser.add_argument(
        "--tier",
        choices=("A", "B", "C", "AB", "all"),
        default="AB",
        help="Which tier(s) to run (default: AB)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip projects that already have results/summary.json or results/comparison.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "CI smoke subset of Tier A (ignores --tier for slug selection). "
            "Never skips existing results. Fails on skipped_missing_extra / "
            "partial unless --allow-skip is also set."
        ),
    )
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help=(
            "Treat results.json status skipped_missing_extra/partial as "
            "non-fatal (harness status skipped/partial). Default for --smoke "
            "is to fail unexpected skips."
        ),
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="Optional explicit slug filter (still respects --tier script kind)",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.smoke:
        # CI gate: always re-run; never trust stale results/.
        args.skip_existing = False
        args.tier = "A"
        slug_filter = set(CI_SMOKE_TIER_A)
        # Smoke denies soft-skips unless --allow-skip is explicit.
        allow_skip = bool(args.allow_skip)
    elif args.slugs:
        slug_filter = set(args.slugs)
        # Local / explicit slug runs tolerate soft-skips by default.
        allow_skip = True
    else:
        slug_filter = None
        # Local full runs historically tolerate missing-extra soft skips.
        allow_skip = True

    jobs: list[tuple[str, str, Path, Path]] = []
    # (tier, slug, script, result_marker)
    if args.tier in ("A", "AB", "all"):
        for slug in TIER_A:
            if slug_filter is not None and slug not in slug_filter:
                continue
            jobs.append(
                (
                    "A",
                    slug,
                    PROOFS / slug / "script.py",
                    PROOFS / slug / "results" / "results.json",
                )
            )
    if args.tier in ("B", "AB", "all"):
        for slug in TIER_B:
            if slug_filter is not None and slug not in slug_filter:
                continue
            jobs.append(
                (
                    "B",
                    slug,
                    PROOFS / slug / "script.py",
                    PROOFS / slug / "results" / "summary.json",
                )
            )
    if args.tier in ("C", "all"):
        for slug in TIER_C:
            if slug_filter is not None and slug not in slug_filter:
                continue
            jobs.append(
                (
                    "C",
                    slug,
                    PROOFS / slug / "baseline_industry.py",
                    PROOFS / slug / "results" / "comparison.json",
                )
            )

    if not jobs:
        print("No proof jobs matched the requested filters.", file=sys.stderr)
        return 2

    rows = []
    for tier, slug, script, marker in jobs:
        if args.skip_existing and marker.is_file():
            rows.append(
                {"tier": tier, "slug": slug, "status": "skipped_existing", "s": 0.0}
            )
            print(f"[{tier}] SKIP existing {slug}")
            continue
        if tier == "C" and not script.is_file():
            # Some Tier A projects embed the twin inside script.py (e.g. loan-approval).
            if marker.is_file():
                rows.append(
                    {
                        "tier": tier,
                        "slug": slug,
                        "status": "ok_embedded",
                        "s": 0.0,
                        "note": "comparison.json present; no separate baseline_industry.py",
                    }
                )
                print(f"[{tier}] OK embedded {slug}")
            else:
                rows.append(
                    {"tier": tier, "slug": slug, "status": "no_baseline", "s": 0.0}
                )
                print(f"[{tier}] NO baseline {slug}")
            continue
        print(f"[{tier}] RUN {slug} ...", flush=True)
        process_status, elapsed, note = _run(script, timeout=args.timeout)
        result_status = _read_result_status(marker) if process_status == "ok" else None
        status = _classify_process_ok(
            process_status=process_status,
            result_status=result_status,
            allow_skip=allow_skip,
        )
        row: dict[str, Any] = {
            "tier": tier,
            "slug": slug,
            "status": status,
            "s": round(elapsed, 2),
            "note": note,
        }
        if result_status is not None:
            row["result_status"] = result_status
        rows.append(row)
        print(
            f"[{tier}] {status.upper()} {slug} ({elapsed:.1f}s) {note[:120]}"
            + (f" result={result_status}" if result_status else "")
        )

    # Summary counts
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    out = {
        "tier": args.tier,
        "smoke": bool(args.smoke),
        "allow_skip": allow_skip,
        "n_jobs": len(rows),
        "counts": by_status,
        "rows": rows,
    }
    report_path = PROOFS / "_lib" / "last_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n=== Proofs run report ===")
    print(json.dumps(by_status, indent=2))
    print(f"Wrote {report_path}")

    fatal = (
        by_status.get("error", 0)
        + by_status.get("timeout", 0)
        + by_status.get("unexpected_skip", 0)
        + by_status.get("missing", 0)
    )
    return 0 if fatal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
