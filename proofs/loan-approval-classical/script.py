"""Tier A proof: classical credit approval on public German Credit when cached."""

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
    load_classical_credit_table,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    sklearn_logreg_twin,
    supervised_roles,
    write_results,
)


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
    ctx = new_proof_context("loan-approval-classical", seed=42)
    frame, data_meta = load_classical_credit_table(seed=ctx.seed)
    roles = supervised_roles(data_meta)
    target = str(data_meta["target"])
    features = list(data_meta["feature_columns"])

    session_cv = Session.ingest(frame.copy())
    session_cv.set_roles(roles)
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

    recipe = PreprocessRecipe(impute="median", encode="onehot", scale="standard")
    cv = session_cv.cv_score(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
        cv=5,
        preprocess=recipe,
    )

    session = Session.ingest(frame.copy())
    session.set_roles(roles)
    session.inject_split(
        train_indices=list(plan.train_indices),
        validation_indices=list(plan.validation_indices),
        test_indices=list(plan.test_indices),
    )
    session.impute(strategy="median")
    session.encode(method="onehot")
    session.scale(method="standard")
    session.handle_outliers(method="iqr", action="cap")
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

    test = session.evaluate(partition="test")
    bundle = session.save_pipeline(
        ctx.artifacts_dir / "pipeline",
        evaluate_partition="test",
        title="Loan approval classical proof",
    )

    industry = sklearn_logreg_twin(
        frame,
        plan,
        feature_columns=features,
        target=target,
        seed=ctx.seed,
    )
    bml_test = metrics_round(dict(test.metrics))
    if data_meta.get("real_public_dataset"):
        refuse_perfect_scores(
            bml_test,
            keys=("accuracy", "f1", "f1_weighted", "f1_macro", "roc_auc"),
            ceiling=1.0,
            proof_slug="loan-approval-classical",
            context="credit-g / public credit holdout",
        )
    comparison = {
        "same_split": True,
        "split_counts": counts,
        "buildml": {
            "backend": "buildml.Session",
            "estimator": "LogisticRegression",
            "test_metrics": bml_test,
        },
        "industry": industry,
        "deltas": {},
        "disclosure": (
            "Deltas are descriptive on one draw; not a claim of "
            "universal superiority. Workflow parity matters more than tiny metric gaps."
        ),
        "loader_selected": data_meta.get("loader_selected"),
    }
    for key in ("accuracy", "f1", "roc_auc"):
        if key in bml_test and key in industry["test_metrics"]:
            comparison["deltas"][key] = round(
                float(bml_test[key]) - float(industry["test_metrics"][key]),
                6,
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
            "split": {
                "kind": plan.kind,
                "counts": counts,
                "stratify": True,
            },
            "leakage_controls": [
                "Stratified train/validation/test before any fit",
                "cv_score used PreprocessRecipe on train folds only",
                "Session-global impute/encode/scale/outliers fit on train",
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
            "industry_comparison": comparison,
            "limitations": [
                (
                    "OpenML German Credit (credit-g) when cached; otherwise the "
                    "in-repo credit draw. Not a regulated credit bureau extract."
                ),
                "Single seed; no nested outer CV reported as primary claim",
                "No fairness / disparate-impact audit in this proof",
            ],
        },
    )
    write_results(ctx, comparison, filename="comparison.json")
    print(
        "loan-approval-classical OK",
        data_meta.get("loader_selected"),
        bml_test,
    )


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("loan-approval-classical", seed=42)
        write_results(
            ctx,
            {"status": "skipped_missing_extra", "error": str(exc)},
        )
        raise
