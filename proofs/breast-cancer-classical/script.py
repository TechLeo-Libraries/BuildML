"""Tier A proof: classical binary classification on sklearn breast_cancer."""

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
from buildml.preprocess import PreprocessRecipe
from proofs._lib import (
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    load_sklearn_breast_cancer,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    write_results,
)


TARGET = "malignant"


def _membership_labels(plan) -> list[str]:
    n = max(plan.train_indices + plan.validation_indices + plan.test_indices) + 1
    labels = ["unused"] * n
    for i in plan.train_indices:
        labels[i] = "train"
    for i in plan.validation_indices:
        labels[i] = "validation"
    for i in plan.test_indices:
        labels[i] = "test"
    return labels


def main() -> None:
    ctx = new_proof_context("breast-cancer-classical", seed=42)
    frame, data_meta = load_sklearn_breast_cancer()
    features = list(data_meta["feature_columns"])

    session_cv = Session.ingest(frame.copy())
    session_cv.set_roles(
        {**{c: "feature" for c in features}, TARGET: "target"}
    )
    session_cv.split(
        test_size=0.2,
        validation_size=0.2,
        stratify=True,
        random_state=ctx.seed,
    )
    plan = session_cv.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_membership_labels(plan))
    assert_no_test_in_selection(
        selection_partition="train_cv",
        evaluation_partition="test",
    )

    recipe = PreprocessRecipe(impute="median", scale="standard")
    cv = session_cv.cv_score(
        LogisticRegression(max_iter=2000, random_state=ctx.seed),
        task="classification",
        cv=5,
        preprocess=recipe,
    )

    session = Session.ingest(frame.copy())
    session.set_roles({**{c: "feature" for c in features}, TARGET: "target"})
    session.inject_split(
        train_indices=list(plan.train_indices),
        validation_indices=list(plan.validation_indices),
        test_indices=list(plan.test_indices),
    )
    session.impute(strategy="median")
    session.scale(method="standard")
    session.fit(
        LogisticRegression(max_iter=2000, random_state=ctx.seed),
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

    test = session.evaluate(partition="test")
    bml_test = metrics_round(dict(test.metrics))
    refuse_perfect_scores(
        bml_test,
        keys=("accuracy", "f1", "f1_weighted", "f1_macro", "roc_auc"),
        ceiling=1.0,
        proof_slug="breast-cancer-classical",
        context="sklearn breast_cancer holdout",
    )
    bundle = session.save_pipeline(
        ctx.artifacts_dir / "pipeline",
        evaluate_partition="test",
        title="Breast cancer classical proof (REAL_PUBLIC_DATASET)",
    )

    write_results(
        ctx,
        {
            "status": "completed",
            "evidence_tier": "REAL_PUBLIC_DATASET",
            "data": data_meta,
            "split": {
                "kind": plan.kind,
                "protocol": "stratified_train_validation_test_0.6_0.2_0.2",
                "counts": counts,
                "stratify": True,
                "random_state": ctx.seed,
            },
            "leakage_controls": [
                "Stratified train/validation/test before any fit",
                "cv_score used PreprocessRecipe on train folds only",
                "Session-global impute/scale fit on train",
                "tune_threshold on validation only",
                "Test evaluated once after selection",
            ],
            "cv": {
                "mean_metrics": metrics_round(dict(cv.mean_metrics)),
                "std_metrics": metrics_round(dict(cv.std_metrics)),
            },
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": bml_test,
            "threshold_tuning": threshold_info,
            "bundle_path": str(bundle),
            "limitations": [
                "Single seed; small clinical tabular sample (n=569)",
                "Not a clinical device certification",
                "Refuses perfect holdout accuracy/F1/ROC-AUC == 1.0",
            ],
        },
    )
    print("breast-cancer-classical OK", bml_test)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("breast-cancer-classical", seed=42)
        write_results(
            ctx,
            {"status": "skipped_missing_extra", "error": str(exc)},
        )
        raise
