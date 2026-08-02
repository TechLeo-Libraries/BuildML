"""Tier A proof: classical credit/loan approval with industry sklearn twin."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.preprocess import PreprocessRecipe
from proofs._lib import (
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    load_credit_approval_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURE_NUM = ["age", "income", "debt_ratio", "employment_years"]
FEATURE_CAT = ["region", "product"]
TARGET = "approved"


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


def _sklearn_twin(frame, plan) -> dict:
    """Same split indices; sklearn ColumnTransformer pipeline (Tier C)."""
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    x_train = frame.loc[train_idx, FEATURE_NUM + FEATURE_CAT]
    y_train = frame.loc[train_idx, TARGET]
    x_test = frame.loc[test_idx, FEATURE_NUM + FEATURE_CAT]
    y_test = frame.loc[test_idx, TARGET]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                FEATURE_CAT,
            ),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "backend": "sklearn.Pipeline",
        "estimator": "LogisticRegression",
        "test_metrics": metrics_round(
            {
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            }
        ),
        "leakage_controls": [
            "Fitted ColumnTransformer + estimator on train indices only",
            "Test indices used once for final metrics",
            "Same SplitPlan indices as BuildML Session",
        ],
    }


def main() -> None:
    ctx = new_proof_context("loan-approval-classical", seed=42)
    frame, data_meta = load_credit_approval_synthetic(n=1200, seed=ctx.seed)

    # Unpoisoned Session for fold-local CV (no Session-global preprocess yet).
    session_cv = Session.ingest(frame.copy())
    session_cv.set_roles(
        {
            **{c: "feature" for c in FEATURE_NUM + FEATURE_CAT},
            TARGET: "target",
        }
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

    recipe = PreprocessRecipe(impute="median", encode="onehot", scale="standard")
    cv = session_cv.cv_score(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
        cv=5,
        preprocess=recipe,
    )

    # Production-style Session-global prep → fit → val tune → test once.
    session = Session.ingest(frame.copy())
    session.set_roles(
        {
            **{c: "feature" for c in FEATURE_NUM + FEATURE_CAT},
            TARGET: "target",
        }
    )
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
    # Threshold policy selected on validation only (not test).
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

    industry = _sklearn_twin(frame, plan)
    bml_test = metrics_round(dict(test.metrics))
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
            "Deltas are descriptive on one synthetic draw; not a claim of "
            "universal superiority. Workflow parity matters more than tiny metric gaps."
        ),
    }
    for key in ("accuracy", "f1", "roc_auc"):
        if key in bml_test and key in industry["test_metrics"]:
            comparison["deltas"][key] = round(
                float(bml_test[key]) - float(industry["test_metrics"][key]),
                6,
            )

    write_results(
        ctx,
        {
            "status": "completed",
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
                "Synthetic labels — not a regulated credit bureau dataset",
                "Single seed; no nested outer CV reported as primary claim",
                "No fairness / disparate-impact audit in this proof",
            ],
        },
    )
    write_results(ctx, comparison, filename="comparison.json")
    print("loan-approval-classical OK", bml_test)


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
