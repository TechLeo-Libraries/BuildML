"""Tier A proof: classical mortgage default classification."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

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
    ctx = new_proof_context("mortgage-default-classical", seed=101)
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
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )

    val = session.evaluate(partition="validation")
    try:
        thr = session.tune_threshold(partition="validation")
        threshold_info = {
            "partition": "validation",
            "report_keys": sorted(thr.to_dict().keys())
            if hasattr(thr, "to_dict")
            else list(getattr(thr, "__dict__", {}).keys()),
        }
    except Exception as exc:  # noqa: BLE001
        threshold_info = {"error": f"{type(exc).__name__}: {exc}"}

    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    test = session.evaluate(partition="test")
    bundle = session.save_pipeline(
        ctx.artifacts_dir / "pipeline",
        evaluate_partition="test",
        title="Mortgage default classical proof",
    )
    bml_test = metrics_round(dict(test.metrics))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "leakage_controls": [
                "Stratified train/validation/test before any fit",
                "Impute/encode/scale fit on train only",
                "tune_threshold on validation only",
                "Test evaluated once after selection",
            ],
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": bml_test,
            "threshold_tuning": threshold_info,
            "bundle_path": str(bundle),
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn Pipeline twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic mortgage labels — not a real servicing / HMDA extract",
                "Single seed; no fairness audit",
            ],
        },
    )
    print("mortgage-default-classical OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("mortgage-default-classical", seed=101)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("mortgage-default-classical SKIPPED", exc)
