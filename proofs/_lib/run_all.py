"""Re-run Tier A / Tier B / Tier C proofs and print a status report.

Usage (from repo root)::

    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all
    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all --tier B
    .\\.venv\\Scripts\\python.exe -m proofs._lib.run_all --tier C --skip-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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
            return "ok", elapsed, (proc.stdout or "").strip().splitlines()[-1:][0] if proc.stdout else ""
        err = (proc.stderr or proc.stdout or "").strip()
        return "error", elapsed, err[-500:]
    except subprocess.TimeoutExpired:
        return "timeout", time.perf_counter() - t0, f">{timeout}s"
    except Exception as exc:  # noqa: BLE001
        return "error", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


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
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    jobs: list[tuple[str, str, Path, Path]] = []
    # (tier, slug, script, result_marker)
    if args.tier in ("A", "AB", "all"):
        for slug in TIER_A:
            jobs.append(
                ("A", slug, PROOFS / slug / "script.py", PROOFS / slug / "results" / "results.json")
            )
    if args.tier in ("B", "AB", "all"):
        for slug in TIER_B:
            jobs.append(
                ("B", slug, PROOFS / slug / "script.py", PROOFS / slug / "results" / "summary.json")
            )
    if args.tier in ("C", "all"):
        for slug in TIER_C:
            jobs.append(
                (
                    "C",
                    slug,
                    PROOFS / slug / "baseline_industry.py",
                    PROOFS / slug / "results" / "comparison.json",
                )
            )

    rows = []
    for tier, slug, script, marker in jobs:
        if args.skip_existing and marker.is_file():
            rows.append({"tier": tier, "slug": slug, "status": "skipped_existing", "s": 0.0})
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
                rows.append({"tier": tier, "slug": slug, "status": "no_baseline", "s": 0.0})
                print(f"[{tier}] NO baseline {slug}")
            continue
        print(f"[{tier}] RUN {slug} ...", flush=True)
        status, elapsed, note = _run(script, timeout=args.timeout)
        rows.append({"tier": tier, "slug": slug, "status": status, "s": round(elapsed, 2), "note": note})
        print(f"[{tier}] {status.upper()} {slug} ({elapsed:.1f}s) {note[:120]}")

    # Summary counts
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    out = {
        "tier": args.tier,
        "n_jobs": len(rows),
        "counts": by_status,
        "rows": rows,
    }
    report_path = PROOFS / "results_run_all.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep under proofs/ root as a harness artifact (gitignored results dirs are per-project).
    report_path = PROOFS / "_lib" / "last_run_report.json"
    report_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n=== Proofs run report ===")
    print(json.dumps(by_status, indent=2))
    print(f"Wrote {report_path}")
    return 0 if by_status.get("error", 0) == 0 and by_status.get("timeout", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
