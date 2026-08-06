"""Tier A proof: Industry EDA Static + Dashboard across diverse datasets.

Evidences BUILDML STATIC EDA (research HTML) and the Industry EDA App sheet /
API payloads on ≥10 frames: sklearn real-world tables and synthetic stress
cases (classification, regression, unsupervised-ish, missingness, cardinality,
width, imbalance).

Regenerate from the repo root::

    python proofs/eda-industry-adaptability/script.py

Convenience smoke (same cases, artifacts under ``.buildml-artifacts/gauntlet/``)::

    python scripts/eda_adaptability_gauntlet.py

Requires ``buildml[dashboard]`` (FastAPI TestClient) for App evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from buildml.core.errors import MissingExtraError
from proofs._lib import new_proof_context, write_results
from proofs._lib.env import extra_available, skip_reason


SLUG = "eda-industry-adaptability"


def main() -> None:
    ctx = new_proof_context(SLUG, seed=0)

    if not extra_available("fastapi"):
        write_results(
            ctx,
            {
                "status": "skipped_missing_extra",
                "domain": "eda",
                "reason": skip_reason(
                    "fastapi",
                    feature="Industry EDA App sheet / API evidence",
                ),
                "surfaces": ["static_research_html", "dashboard_app"],
            },
        )
        raise SystemExit(0)

    try:
        from eda_adaptability_gauntlet import run_gauntlet
    except ImportError as exc:
        raise MissingExtraError("dashboard", "EDA adaptability gauntlet import") from exc

    artifacts = ctx.results_dir / "cases"
    results, summary = run_gauntlet(artifacts_dir=artifacts, quiet=False)

    cases = []
    for row in summary.get("cases") or []:
        cases.append(
            {
                "name": row.get("name"),
                "kind": row.get("kind"),
                "task": row.get("task"),
                "n_rows": row.get("n_rows"),
                "n_cols": row.get("n_cols"),
                "static_ok": row.get("static_ok"),
                "app_ok": row.get("app_ok"),
                "findings": row.get("findings"),
                "assumptions": row.get("assumptions"),
                "ledger_groups": row.get("ledger_groups"),
                "gates_items": row.get("gates_items"),
                "academy_concepts": row.get("academy_concepts"),
                "ok": bool(row.get("static_ok") and row.get("app_ok") and not row.get("errors")),
                "static_html": f"cases/{row.get('name')}_static.html",
                "app_snapshot": f"cases/{row.get('name')}_app.json",
                "errors": row.get("errors") or [],
                "notes": [n for n in (row.get("notes") or []) if n != "pass"][:8],
            }
        )

    n_ok = int(summary.get("n_passed") or 0)
    n_cases = int(summary.get("n_cases") or 0)
    all_ok = n_ok == n_cases and n_cases >= 10

    write_results(
        ctx,
        {
            "status": "completed" if all_ok else "failed",
            "domain": "eda",
            "surfaces": {
                "static": "BUILDML STATIC EDA research HTML (Offline HTML primary)",
                "app": "Industry EDA App cockpit sheet + gates + academy API payloads",
            },
            "completeness": {
                "n_datasets": n_cases,
                "n_passed": n_ok,
                "min_required": 10,
                "static_markers": [
                    "Findings register",
                    "Ledger",
                    "Recommended sequence",
                    "What each finding assumes",
                    "Offline HTML",
                ],
                "app_checks": [
                    "sheet.kpis / register / ledger / assumptions",
                    "adapt guidance bound to live report",
                    "gates + academy payloads non-empty",
                    "Offline HTML primary; no PDF briefing / CSV header button",
                ],
            },
            "datasets": cases,
            "artifacts": {
                "summary_md": "cases/summary.md",
                "summary_json": "cases/summary.json",
                "regenerate": [
                    f"python proofs/{SLUG}/script.py",
                    "python scripts/eda_adaptability_gauntlet.py",
                ],
            },
            "data": {
                "name": "eda-adaptability-cohort",
                "source": "sklearn bundled tables + in-repo synthetics (see datasets[].kind)",
                "task": "eda_screening",
                "n_rows": sum(int(c.get("n_rows") or 0) for c in cases),
                "n_features": None,
                "evidence_tier": "MIXED_PUBLIC_AND_SYNTHETIC",
                "license": "sklearn BSD-3 / in-repo synthetic license-clear",
            },
            "metrics": {
                "n_datasets": n_cases,
                "n_passed": n_ok,
                "pass_rate": round(n_ok / n_cases, 4) if n_cases else 0.0,
            },
        },
    )

    if not all_ok:
        failed = [c["name"] for c in cases if not c["ok"]]
        raise SystemExit(
            f"{SLUG} failed: {n_ok}/{n_cases} passed; failed={failed}"
        )

    print(f"{SLUG}: {n_ok}/{n_cases} datasets evidenced (Static + App).")


if __name__ == "__main__":
    main()
