"""Tier C: sklearn MLPClassifier twin for torch-tabular-underwrite."""

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)

FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
TARGET = "defaulted"


def main() -> None:
    ctx = new_proof_context("torch-tabular-underwrite", seed=106)
    frame, _ = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURE_NUM + FEATURE_CAT}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
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
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=80,
                    random_state=ctx.seed,
                ),
            ),
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
    if bml_raw.get("status") == "skipped_missing_extra":
        write_comparison(
            ctx,
            buildml={"backend": "buildml.Session.fit_torch", "status": "skipped_missing_extra", "test_metrics": {}},
            industry={
                "backend": "sklearn.MLPClassifier",
                "test_metrics": industry_metrics,
                "note": "BuildML torch path skipped; industry twin still ran",
            },
            split_counts={
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
            },
            delta_keys=("accuracy", "f1", "roc_auc"),
        )
        print("torch-tabular-underwrite Tier C OK (BuildML skipped)", industry_metrics)
        return

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
            "backend": "buildml.Session.fit_torch",
            "epochs": 3,
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.MLPClassifier",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Preprocessor + MLP fit on train only",
                "Test evaluated once after lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
    )
    print("torch-tabular-underwrite Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
