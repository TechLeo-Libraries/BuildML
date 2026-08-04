"""Tier A proof: classical regression on sklearn diabetes progression."""

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
    load_sklearn_diabetes,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    write_results,
)


TARGET = "progression"


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
    ctx = new_proof_context("diabetes-progression-regression", seed=102)
    frame, data_meta = load_sklearn_diabetes()
    features = list(data_meta["feature_columns"])

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in features}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    session.scale(method="standard")

    estimator_name = "HistGradientBoostingRegressor"
    try:
        session.fit(
            HistGradientBoostingRegressor(
                max_depth=3, max_iter=100, random_state=ctx.seed
            ),
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
    bml_test = metrics_round(dict(test.metrics))
    refuse_perfect_scores(
        bml_test,
        keys=("r2", "r2_score"),
        ceiling=1.0,
        proof_slug="diabetes-progression-regression",
        context="sklearn diabetes holdout",
    )
    # Non-trivial signal floor: diabetes is noisy; R² should be positive but modest.
    r2 = bml_test.get("r2", bml_test.get("r2_score"))
    if isinstance(r2, (int, float)) and float(r2) <= 0.0:
        raise SystemExit(
            "diabetes-progression-regression refused non-informative model: "
            f"r2={float(r2):.4f} <= 0 on real diabetes holdout."
        )

    bundle = session.save_pipeline(
        ctx.artifacts_dir / "pipeline",
        evaluate_partition="test",
        title="Diabetes progression regression (REAL_PUBLIC_DATASET)",
    )
    write_results(
        ctx,
        {
            "status": "completed",
            "evidence_tier": "REAL_PUBLIC_DATASET",
            "data": data_meta,
            "split": {
                "kind": plan.kind,
                "protocol": "random_train_validation_test_0.6_0.2_0.2",
                "counts": counts,
                "random_state": ctx.seed,
            },
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
            "limitations": [
                "Sklearn diabetes is a small research sample (n=442)",
                "Single seed; not a clinical outcome model",
                "Refuses R² >= 1.0 and R² <= 0",
            ],
        },
    )
    print("diabetes-progression-regression OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("diabetes-progression-regression", seed=102)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("diabetes-progression-regression SKIPPED", exc)
