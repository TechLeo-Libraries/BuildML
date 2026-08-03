"""Tier C: hand-built scikit-learn text pipeline twin for ticket-routing-nlp.

The twin reproduces what a practitioner would write by hand — TfidfVectorizer
plus LogisticRegression inside a Pipeline, fitted on the same train indices —
so the comparison isolates workflow discipline rather than model choice.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("ticket-routing-nlp", seed=11)
    frame, _ = load_support_tickets_synthetic(n=900, seed=ctx.seed)

    # Borrow BuildML only for the split so both sides score the same rows.
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
    train, validation, test = (
        list(plan.train_indices),
        list(plan.validation_indices),
        list(plan.test_indices),
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=ctx.seed,
                ),
            ),
        ]
    )
    pipeline.fit(frame.loc[train, "body"], frame.loc[train, "queue"])

    def score(indices: list[int]) -> dict[str, float]:
        y_true = frame.loc[indices, "queue"]
        y_pred = pipeline.predict(frame.loc[indices, "body"])
        proba = pipeline.predict_proba(frame.loc[indices, "body"])
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "log_loss": float(log_loss(y_true, proba, labels=list(pipeline.classes_))),
        }

    industry_validation = metrics_round(score(validation))
    industry_test = metrics_round(score(test))

    results = load_buildml_results(ctx.project_dir)
    buildml_test = metrics_round(dict(results.get("test_metrics", {})))

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.nlp/fit_text_classifier(tfidf + logistic)",
            "validation_metrics": metrics_round(
                dict(results.get("validation_metrics", {}))
            ),
            "test_metrics": buildml_test,
            "extras_beyond_the_twin": [
                "profile_text_corpus contamination screen with a stated threshold",
                "interpret_text_prediction exact token attributions",
                "train-fitted NMF topics with NPMI coherence, assigned to holdout",
                "keyphrases, extractive summaries, rule entities, lexicon sentiment",
                "buildml.nlp_bundle.v1 carrying the normalization plan with the vectorizer",
                "history records and a walkthrough disclosure for every step",
            ],
        },
        industry={
            "backend": "sklearn.Pipeline(TfidfVectorizer(1,2, min_df=2, sublinear) + LogisticRegression(balanced))",
            "validation_metrics": industry_validation,
            "test_metrics": industry_test,
            "leakage_controls": [
                "Same stratified split indices (seed=11)",
                "Vectorizer and head fitted inside a Pipeline on train rows only",
                "Validation scored before test; test scored once",
            ],
            "not_provided_by_the_twin": [
                "No duplicate / near-duplicate contamination screen",
                "No stored normalization plan — preprocessing has to be re-described by hand at inference",
                "No token attribution, topic coherence, or audit history without extra code",
            ],
        },
        split_counts={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        delta_keys=(
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "log_loss",
        ),
    )
    print("ticket-routing-nlp Tier C OK", industry_test)


if __name__ == "__main__":
    main()
