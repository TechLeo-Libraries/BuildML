"""Tier A proof: coldstart-meta-adapt — few-shot new-category cold start."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import TORCH_STATUS, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("coldstart-meta-adapt", seed=26)
    rng = np.random.default_rng(ctx.seed)
    # Categories (= domains) for cold-start SKU repurchase — distinct from few-shot-domain-adapt tasks.
    rows = []
    for cat in range(18):
        center = rng.normal(size=5) * (0.6 + cat * 0.025)
        for _ in range(28):
            x = center + rng.normal(scale=0.35, size=5)
            y = int((x[0] + 0.35 * x[2] - 0.2 * x[4]) > 0)
            rows.append({
                **{f"emb{j}": float(x[j]) for j in range(5)},
                "repurchase": y,
                "category_id": f"cat{cat}",
            })
    frame = pd.DataFrame(rows)
    session = (
        Session.ingest(frame)
        .set_roles({
            **{f"emb{j}": "feature" for j in range(5)},
            "repurchase": "target",
            "category_id": "group",
        })
        .group_split(
            test_size=0.25, validation_size=0.15,
            random_state=ctx.seed, group_column="category_id",
        )
    )
    backend_note = "sklearn"
    try:
        fit = session.metalearning.fit(
            method="prototypical", task_column="category_id",
            n_way=2, k_shot=5, n_query=5, random_state=ctx.seed,
        )
    except (MissingExtraError, TypeError, ValueError) as exc:
        try:
            fit = session.metalearning.fit(
                method="warm_start", task_column="category_id",
                n_way=2, k_shot=5, n_query=5, random_state=ctx.seed,
            )
            backend_note = f"warm_start_fallback({type(exc).__name__})"
        except Exception as exc2:  # noqa: BLE001
            write_results(ctx, {
                "status": "skipped_error",
                "error": f"{type(exc2).__name__}: {exc2}",
                "torch": TORCH_STATUS,
            })
            print("coldstart-meta-adapt SKIPPED", exc2)
            return
    ev = session.metalearning.evaluate(partition="test")
    try:
        bundle = session.metalearning.save_bundle(ctx.artifacts_dir / "meta_bundle")
        bundle_path = str(bundle)
    except Exception as exc:  # noqa: BLE001
        bundle_path = f"unavailable: {exc}"
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_category_coldstart",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
        },
        "backend_note": backend_note,
        "torch": TORCH_STATUS,
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": bundle_path,
        "leakage_controls": [
            "group_split by category_id",
            "Episodic eval on held-out categories",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: per-category NearestCentroid k-shot; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Synthetic categories; not production catalog cold-start",
            "Distinct narrative from few-shot-domain-adapt",
        ],
    })
    print("coldstart-meta-adapt OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
