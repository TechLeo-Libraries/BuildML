"""Tier C: sklearn Pipeline twin for breast-cancer-classical."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_sklearn_breast_cancer,
    metrics_round,
    new_proof_context,
    write_comparison,
)

TARGET = "malignant"


def main() -> None:
    ctx = new_proof_context("breast-cancer-classical", seed=42)
    frame, data_meta = load_sklearn_breast_cancer()
    features = list(data_meta["feature_columns"])
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in features}, TARGET: "target"})
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    x_train = frame.loc[train_idx, features]
    y_train = frame.loc[train_idx, TARGET]
    x_test = frame.loc[test_idx, features]
    y_test = frame.loc[test_idx, TARGET]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=ctx.seed)),
        ]
    )
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "roc_auc", "f1_weighted"),
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session",
            "estimator": "LogisticRegression",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.Pipeline",
            "estimator": "LogisticRegression",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Imputer + scaler + estimator fit on train indices only",
                "Test indices used once for final metrics",
                "Same SplitPlan indices as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
        extra={"evidence_tier": "REAL_PUBLIC_DATASET"},
    )
    print("breast-cancer-classical Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
