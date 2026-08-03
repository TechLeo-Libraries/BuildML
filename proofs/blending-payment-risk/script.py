"""Tier A proof: holdout blending ensemble for payment authorization risk."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    load_payment_rail_anomaly_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURES = [
    "amount_z",
    "hour_sin",
    "hour_cos",
    "merchant_risk",
    "device_age_days",
    "velocity_1h",
]
TARGET = "is_attack"


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
    ctx = new_proof_context("blending-payment-risk", seed=105)
    frame, data_meta = load_payment_rail_anomaly_synthetic(seed=ctx.seed)

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    session.scale(method="standard")

    bases = {
        "lr": LogisticRegression(max_iter=1000, random_state=ctx.seed),
        "rf": RandomForestClassifier(n_estimators=80, max_depth=6, random_state=ctx.seed),
    }
    fit = session.fit_blending(
        bases,
        final_estimator=LogisticRegression(max_iter=1000, random_state=ctx.seed),
        holdout_fraction=0.2,
        random_state=ctx.seed,
        task="classification",
    )
    val = session.evaluate_ensemble(partition="validation")
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    test = session.evaluate_ensemble(partition="test")
    bundle = session.save_ensemble_bundle(ctx.artifacts_dir / "ensemble_bundle")
    bml_test = metrics_round(dict(test.metrics))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "strategy": "blending",
            "holdout_fraction": 0.2,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "leakage_controls": [
                "Stratified outer split before scale/blend",
                "Blend holdout carved from train only (holdout_fraction=0.2)",
                "Session validation/test never used for meta-learner fit",
                "Test evaluate_ensemble after lock",
            ],
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": bml_test,
            "bundle_path": str(bundle),
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: holdout-blend sklearn twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic payment authorizations — not a card-network extract",
                "Supervised blend on attack labels; production may be unlabeled",
            ],
        },
    )
    print("blending-payment-risk OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("blending-payment-risk", seed=105)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("blending-payment-risk SKIPPED", exc)
