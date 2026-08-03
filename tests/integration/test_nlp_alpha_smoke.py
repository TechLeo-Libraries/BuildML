"""End-to-end smoke for the NLP Session path.

Walks the full route a user takes — profile, fit, select on validation, score
test, interpret, topic, describe, bundle, reload — and asserts the properties
that make the numbers trustworthy rather than just present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session

_POOLS: dict[str, tuple[str, ...]] = {
    "billing": (
        "Invoice INV-{ref} charged the annual fee twice on the same card.",
        "The renewal quote said one figure but the invoice came to almost double.",
        "A proration credit never appeared against invoice INV-{ref}.",
    ),
    "shipping": (
        "Order ORD-{ref} was promised for the 3rd and arrived nine days late.",
        "Two of the four cartons in shipment ORD-{ref} were crushed in transit.",
        "Tracking for ORD-{ref} has not updated since it left the depot.",
    ),
    "account": (
        "Single sign-on stopped working for the whole workspace this morning.",
        "The onboarding portal rejects the invite link for every new hire.",
        "Password resets arrive but the link has already expired on arrival.",
    ),
    "hardware": (
        "The hinge on unit HW-{ref} snapped within a week of light use.",
        "Unit HW-{ref} overheats and shuts down under a normal workload.",
        "The display on unit HW-{ref} flickers whenever the lid is moved.",
    ),
}

_TAILS = (
    "Finance has already flagged the discrepancy in their reconciliation.",
    "The team is blocked and needs a written answer before Friday.",
    "A second identical case was opened by the same site last month.",
)


def _tickets(n: int = 320, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    queues = list(_POOLS)
    rows: list[dict[str, object]] = []
    for index in range(n):
        queue = queues[index % len(queues)]
        pool = _POOLS[queue]
        ref = int(rng.integers(10_000, 99_999))
        opening = str(rng.choice(pool)).format(ref=ref)
        tail = str(rng.choice(_TAILS))
        rows.append({"body": f"{opening} {tail}", "queue": queue})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _session(frame: pd.DataFrame) -> Session:
    return (
        Session.ingest(frame.copy())
        .set_roles({"body": "feature", "queue": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=4, stratify=True)
    )


def test_nlp_alpha_smoke(tmp_path: Path) -> None:
    frame = _tickets()
    session = _session(frame)

    profile = session.profile_text_corpus(near_duplicate_threshold=0.9)
    assert profile.text_column == "body"
    assert profile.n_documents == len(frame)
    assert profile.near_duplicate_threshold == 0.9
    assert profile.language_counts.get("en", 0) > 0

    fit = session.fit_text_classifier(
        estimator="logistic",
        ngram_range=(1, 2),
        min_df=2,
        class_weight="balanced",
        random_state=0,
    )
    assert fit.n_train_rows == len(session.split_plan.train_indices)
    assert set(fit.classes) == set(_POOLS)

    validation = session.evaluate_text_classifier(partition="validation")
    test = session.evaluate_text_classifier(partition="test")
    assert validation.partition == "validation"
    assert test.partition == "test"
    # Structured synthetic tickets are separable; a broken pipeline would not be.
    assert test.metrics["accuracy"] > 0.8
    assert test.metrics["balanced_accuracy"] > 0.8
    assert set(test.per_class) == set(_POOLS)
    assert len(test.confusion) == 4

    predicted = session.predict_text(partition="test")
    assert predicted.n_rows == len(session.split_plan.test_indices)
    assert set(predicted.predictions) <= set(_POOLS)

    interpret = session.interpret_text_prediction(
        partition="test", top_k=6, max_documents=3
    )
    assert interpret.n_documents == 3
    assert set(interpret.global_top_tokens) == set(_POOLS)
    for row in interpret.document_attributions:
        for item in row:
            assert item.contribution == float(np.float64(item.weight * item.value))

    topics = session.fit_topics(
        method="nmf", n_topics=4, min_df=3, max_df=0.9, random_state=0
    )
    assigned = session.assign_topics(partition="test")
    assert topics.mean_coherence is not None
    assert -1.0 <= topics.mean_coherence <= 1.0
    assert assigned.n_rows == predicted.n_rows
    assert set(assigned.dominant_topics) <= set(range(4))

    keyphrases = session.extract_keyphrases(partition="train", method="tfidf", top_n=10)
    summaries = session.summarize_text(partition="test", method="textrank", n_sentences=1)
    entities = session.extract_entities(
        partition="test", backend="rules", gazetteers={"TERM": ["invoice", "portal"]}
    )
    sentiment = session.analyze_sentiment(partition="test", backend="lexicon")
    languages = session.detect_language(partition="test")
    assert keyphrases.corpus_keyphrases
    assert summaries.summaries
    assert entities.n_entities > 0
    assert sentiment.n_rows == predicted.n_rows
    assert languages.dominant_language == "en"

    bundle = session.save_nlp_bundle(tmp_path / "nlp_alpha")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "nlp_text_plan.joblib").is_file()
    assert (bundle / "nlp_topic_plan.joblib").is_file()

    # A reloaded bundle must reproduce the holdout score exactly; anything less
    # means the normalization plan did not travel with the vectorizer.
    reloaded = _session(frame)
    reloaded.load_nlp_bundle(bundle)
    assert reloaded.nlp_text_plan is not None
    assert reloaded.nlp_topic_plan is not None
    again = reloaded.evaluate_text_classifier(partition="test")
    assert again.metrics["accuracy"] == test.metrics["accuracy"]
    reassigned = reloaded.assign_topics(partition="test")
    assert reassigned.dominant_topics == assigned.dominant_topics

    report = session.walkthrough()
    status = report.to_dict()["nlp_status"]
    assert status["enabled"] is True
    assert status["has_text_plan"] is True
    assert status["has_topic_plan"] is True
    assert status["has_profile_result"] is True

    summary = session.summarize_history()
    counts = summary.to_dict()["operation_counts"]
    for expected in (
        "profile_text_corpus",
        "fit_text_classifier",
        "evaluate_text_classifier",
        "interpret_text_prediction",
        "fit_topics",
        "assign_topics",
        "save_nlp_bundle",
    ):
        assert counts.get(expected, 0) >= 1, expected

    preview = session.dry_run("interpret_text_prediction")
    step = next(
        item
        for item in preview.steps
        if item.operation == "interpret_text_prediction"
    )
    assert step.available is True
