"""Tier A proof: classical credit-risk holdout on public German Credit when cached."""

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
    load_classical_credit_table,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    supervised_roles,
    write_results,
)


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
    frame, data_meta = load_classical_credit_table(seed=ctx.seed)
    roles = supervised_roles(data_meta)

    session = (
        Session.ingest(frame)
        .set_roles(roles)
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
    if data_meta.get("real_public_dataset"):
        refuse_perfect_scores(
            bml_test,
            keys=("accuracy", "f1", "f1_weighted", "f1_macro", "roc_auc"),
            ceiling=1.0,
            proof_slug="mortgage-default-classical",
            context="credit-g / public credit holdout",
        )
    public = bool(data_meta.get("real_public_dataset"))
    write_results(
        ctx,
        {
            "status": "completed",
            "evidence_tier": data_meta.get(
                "evidence_tier",
                "REAL_PUBLIC_DATASET" if public else "SYNTHETIC_FALLBACK",
            ),
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
                (
                    "Same public German Credit table as loan-approval-classical "
                    "(credit-g when cached). Slug kept; this proof is the shorter "
                    "holdout plus a separate industry twin file. Not HMDA / servicing."
                ),
                "Single seed; no fairness audit",
            ],
        },
    )
    print(
        "mortgage-default-classical OK",
        data_meta.get("loader_selected"),
        bml_test,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("mortgage-default-classical", seed=101)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("mortgage-default-classical SKIPPED", exc)
