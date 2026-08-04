"""Tier A proof: stacking ensemble for credit / mortgage default risk."""

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
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
TARGET = "defaulted"


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
    ctx = new_proof_context("stacking-credit-risk", seed=104)
    frame, data_meta = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURE_NUM + FEATURE_CAT}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    session.impute(strategy="median")
    session.encode(method="onehot")
    session.scale(method="standard")

    bases = {
        "lr": LogisticRegression(max_iter=1000, random_state=ctx.seed),
        "rf": RandomForestClassifier(n_estimators=60, max_depth=5, random_state=ctx.seed),
    }
    fit = session.ensemble.fit_stacking(
        bases,
        final_estimator=LogisticRegression(max_iter=1000, random_state=ctx.seed),
        cv=3,
        task="classification",
    )
    val = session.ensemble.evaluate(partition="validation")
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    test = session.ensemble.evaluate(partition="test")
    bundle = session.ensemble.save_bundle(ctx.artifacts_dir / "ensemble_bundle")
    bml_test = metrics_round(dict(test.metrics))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "strategy": "stacking",
            "cv": 3,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "leakage_controls": [
                "Stratified split before impute/encode/scale/stack",
                "OOF meta features from train CV folds only (cv=3)",
                "Test session.ensemble.evaluate after lock",
            ],
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": bml_test,
            "bundle_path": str(bundle),
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn StackingClassifier twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic mortgage default labels",
                "Two-base stack with logistic meta-learner only",
            ],
        },
    )
    print("stacking-credit-risk OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("stacking-credit-risk", seed=104)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("stacking-credit-risk SKIPPED", exc)
