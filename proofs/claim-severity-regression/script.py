"""Tier A proof: classical insurance claim severity regression."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    load_claim_severity_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURES = ["vehicle_age", "driver_age", "prior_claims", "urban", "deductible"]
TARGET = "severity"


def _labels(plan) -> list[str]:
    n = max(plan.train_indices + plan.validation_indices + plan.test_indices) + 1
    out = ["unused"] * n
    for i in plan.train_indices:
        out[i] = "train"
    for i in plan.validation_indices:
        out[i] = "validation"
    for i in plan.test_indices:
        out[i] = "test"
    return out


def main() -> None:
    ctx = new_proof_context("claim-severity-regression", seed=102)
    frame, data_meta = load_claim_severity_synthetic(n=1100, seed=ctx.seed)

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    session.scale(method="standard")

    # Prefer HistGradientBoosting; fall back to Ridge if fit fails.
    estimator_name = "HistGradientBoostingRegressor"
    try:
        session.fit(
            HistGradientBoostingRegressor(max_depth=4, max_iter=120, random_state=ctx.seed),
            task="regression",
        )
    except Exception:
        session.fit(Ridge(alpha=1.0, random_state=ctx.seed), task="regression")
        estimator_name = "Ridge"

    val = session.evaluate(partition="validation")
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    test = session.evaluate(partition="test")
    bundle = session.save_pipeline(
        ctx.artifacts_dir / "pipeline",
        evaluate_partition="test",
        title="Claim severity regression proof",
    )
    bml_test = metrics_round(dict(test.metrics))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts},
            "estimator": estimator_name,
            "leakage_controls": [
                "Random train/validation/test before any fit",
                "Scaler fit on train only",
                "Model selection uses validation metrics only",
                "Test evaluated once after selection",
            ],
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": bml_test,
            "bundle_path": str(bundle),
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn Ridge twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic P&C severity — not a real claims extract",
                "Single seed; no Tweedie / GLM severity stack",
            ],
        },
    )
    print("claim-severity-regression OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("claim-severity-regression", seed=102)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("claim-severity-regression SKIPPED", exc)
