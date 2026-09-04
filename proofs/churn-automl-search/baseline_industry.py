"""Tier C: sklearn RandomizedSearchCV twin for churn-automl-search."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    infer_feature_kinds,
    load_buildml_results,
    load_sklearn_breast_cancer,
    metrics_round,
    new_proof_context,
    supervised_roles,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("churn-automl-search", seed=7)
    frame, data_meta = load_sklearn_breast_cancer()
    features = list(data_meta["feature_columns"])
    target = str(data_meta["target"])
    numeric, categorical = infer_feature_kinds(frame, features)
    session = (
        Session.ingest(frame.copy())
        .set_roles(supervised_roles(data_meta))
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)

    x_train = frame.loc[train_idx, features]
    y_train = frame.loc[train_idx, target]
    x_test = frame.loc[test_idx, features]
    y_test = frame.loc[test_idx, target]

    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    pipe = Pipeline(
        [
            ("pre", ColumnTransformer(transformers)),
            ("clf", LogisticRegression(max_iter=1000, random_state=ctx.seed)),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        param_distributions=[
            {
                "clf": [LogisticRegression(max_iter=1000, random_state=ctx.seed)],
                "clf__C": [0.1, 1.0, 3.0],
            },
            {
                "clf": [
                    RandomForestClassifier(
                        n_estimators=120, random_state=ctx.seed, n_jobs=1
                    )
                ],
                "clf__max_depth": [3, 6, None],
            },
            {
                "clf": [GradientBoostingClassifier(random_state=ctx.seed)],
                "clf__learning_rate": [0.05, 0.1],
                "clf__n_estimators": [80, 120],
            },
        ],
        n_iter=12,
        cv=3,
        scoring="roc_auc",
        random_state=ctx.seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    proba = search.predict_proba(x_test)[:, 1]
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
        keys=("accuracy", "f1", "roc_auc", "f1_weighted", "average_precision"),
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]

    write_comparison(
        ctx,
        buildml={
            "backend": f"buildml.automl/{bml_raw.get('backend', 'native')}",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.RandomizedSearchCV",
            "best_estimator": type(search.best_estimator_.named_steps["clf"]).__name__,
            "best_cv_score": float(search.best_score_),
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "CV search on train indices only",
                "Validation held out from sklearn search (parity with BuildML test isolation)",
                "Test evaluated once after refit",
                "Same SplitPlan as BuildML Session",
            ],
            "validation_rows_reserved": len(val_idx),
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
        extra={"loader_selected": data_meta.get("name")},
    )
    print("churn-automl-search Tier C OK", data_meta.get("name"), industry_metrics)


if __name__ == "__main__":
    main()
