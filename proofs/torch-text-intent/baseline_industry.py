"""Tier C: Tfidf + LogisticRegression twin for torch-text-intent."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("torch-text-intent", seed=107)
    frame, _ = load_support_tickets_synthetic(n=900, seed=ctx.seed)
    # Same integer label encoding as Tier A (sort=True factorize) for split parity.
    queue_codes, _ = pd.factorize(frame["queue"], sort=True)
    frame = frame.copy()
    frame["queue"] = queue_codes.astype(int)
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "ticket_id": "id",
                "body": "feature",
                "channel": "feature",
                "queue": "target",
            }
        )
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

    x_train = frame.loc[train_idx, "body"]
    y_train = frame.loc[train_idx, "queue"]
    x_test = frame.loc[test_idx, "body"]
    y_test = frame.loc[test_idx, "queue"]

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=ctx.seed),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    if bml_raw.get("status") == "skipped_missing_extra":
        write_comparison(
            ctx,
            buildml={
                "backend": "buildml.session.dl.fit/text",
                "status": "skipped_missing_extra",
                "test_metrics": {},
            },
            industry={
                "backend": "sklearn.TfidfVectorizer+LogisticRegression",
                "test_metrics": industry_metrics,
                "note": "BuildML torch path skipped; industry twin still ran",
            },
            split_counts={
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
            },
            delta_keys=("accuracy", "f1_weighted"),
        )
        print("torch-text-intent Tier C OK (BuildML skipped)", industry_metrics)
        return

    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "f1_weighted"),
    )
    if "f1_weighted" not in bml_metrics and "f1" in bml_metrics:
        bml_metrics["f1_weighted"] = bml_metrics["f1"]

    write_comparison(
        ctx,
        buildml={
            "backend": f"buildml/{bml_raw.get('path', 'torch-text')}",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.TfidfVectorizer+LogisticRegression",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Tfidf vocabulary fit on train only",
                "Test evaluated once after lock",
                "Same SplitPlan as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("accuracy", "f1_weighted"),
    )
    print("torch-text-intent Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
