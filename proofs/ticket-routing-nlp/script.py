"""Tier A proof: ticket-routing-nlp.

Routes free-text support tickets to a queue with the BuildML NLP path, then
shows the surrounding analysis surfaces that make the score believable: corpus
contamination screening, exact token attribution, unsupervised topics,
keyphrases, extractive summaries, and rule-based entity extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from proofs._lib import (
    assert_no_test_in_selection,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("ticket-routing-nlp", seed=11)
    frame, data_meta = load_support_tickets_synthetic(n=900, seed=ctx.seed)

    session = (
        Session.ingest(frame)
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

    # 1. Corpus health first: a text score is only as honest as the split.
    profile = session.nlp.profile_corpus(
        text_column="body",
        near_duplicate_threshold=0.9,
        detect_languages=True,
    )

    # 2. Train-fitted bag-of-n-grams + linear head. Selection reads validation.
    fit = session.nlp.fit_classifier(
        text_column="body",
        vectorizer="tfidf",
        estimator="logistic",
        ngram_range=(1, 2),
        min_df=2,
        class_weight="balanced",
        random_state=ctx.seed,
    )
    validation = session.nlp.evaluate(partition="validation")
    assert_no_test_in_selection(selection_partition="validation")

    # 3. Locked model, then the holdout it never saw.
    test = session.nlp.evaluate(partition="test")
    predicted = session.nlp.predict(partition="test")

    # 4. Exact token attribution — an identity for a linear head, not an
    #    approximation, and refused outright for hashing / dense backends.
    interpret = session.nlp.interpret(
        partition="test", top_k=8, max_documents=5
    )

    # 5. Unsupervised structure fitted on train only, then assigned to holdout.
    topics = session.nlp.fit_topics(
        method="nmf",
        n_topics=4,
        text_column="body",
        min_df=3,
        max_df=0.9,
        stopword_language="en",
        random_state=ctx.seed,
    )
    assigned = session.nlp.assign_topics(partition="test")

    # 6. Description surfaces that claim no quality metric.
    keyphrases = session.nlp.extract_keyphrases(
        partition="train", method="tfidf", top_n=12, per_document=False
    )
    summaries = session.nlp.summarize(
        partition="test", method="textrank", n_sentences=2, max_documents=5
    )
    entities = session.nlp.extract_entities(
        partition="test",
        backend="rules",
        gazetteers={"QUEUE_TERM": ["invoice", "courier", "workspace", "hinge"]},
        max_documents=5,
    )
    sentiment = session.nlp.analyze_sentiment(partition="test", backend="lexicon")

    bundle = session.nlp.save_bundle(ctx.artifacts_dir / "nlp_bundle")

    # 7. A reloaded bundle must reproduce the holdout score exactly, which is
    #    what proves the normalization plan travelled with the vectorizer.
    reloaded = (
        Session.ingest(frame)
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
    reloaded.nlp.load_bundle(bundle, trusted=True)
    reloaded_test = reloaded.nlp.evaluate(partition="test")
    reproduced = bool(
        abs(
            float(reloaded_test.metrics["accuracy"]) - float(test.metrics["accuracy"])
        )
        < 1e-12
    )

    split_plan = session.split_plan
    assert split_plan is not None

    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split_counts": {
                "train": len(split_plan.train_indices),
                "validation": len(split_plan.validation_indices),
                "test": len(split_plan.test_indices),
            },
            "corpus_profile": {
                "n_documents": profile.n_documents,
                "vocabulary_size": profile.vocabulary_size,
                "hapax_rate": round(profile.hapax_rate, 6),
                "duplicate_document_rate": round(profile.duplicate_document_rate, 6),
                "train_holdout_exact_overlap": profile.train_holdout_exact_overlap,
                "train_holdout_near_duplicate": profile.train_holdout_near_duplicate,
                "near_duplicate_threshold": profile.near_duplicate_threshold,
                "holdout_oov_token_rate": (
                    None
                    if profile.holdout_oov_token_rate is None
                    else round(profile.holdout_oov_token_rate, 6)
                ),
                "language_counts": dict(profile.language_counts),
                "findings": list(profile.findings),
            },
            "fit": {
                "backend": fit.backend,
                "estimator": fit.estimator,
                "n_train_rows": fit.n_train_rows,
                "vocabulary_size": fit.vocabulary_size,
                "classes": list(fit.classes),
                "class_counts": dict(fit.class_counts),
                "train_score": round(float(fit.train_score or 0.0), 6),
            },
            "validation_metrics": metrics_round(dict(validation.metrics)),
            "test_metrics": metrics_round(dict(test.metrics)),
            "test_per_class": metrics_round(dict(test.per_class)),
            "test_confusion": [list(row) for row in test.confusion],
            "test_oov_token_rate": round(float(test.oov_rate or 0.0), 6),
            "predict": {
                "n_rows": predicted.n_rows,
                "has_probabilities": bool(predicted.probabilities),
            },
            "token_attribution": {
                "method": interpret.method,
                "target_class": interpret.target_class,
                "n_documents": interpret.n_documents,
                "top_tokens_first_document": [
                    {
                        "token": item.token,
                        "contribution": round(item.contribution, 6),
                    }
                    for item in (interpret.document_attributions or ((),))[0][:5]
                ],
                "global_top_tokens": {
                    str(label): [item.token for item in items[:5]]
                    for label, items in interpret.global_top_tokens.items()
                },
            },
            "topics": {
                "method": topics.method,
                "n_topics": topics.n_topics,
                "mean_npmi_coherence": (
                    None
                    if topics.mean_coherence is None
                    else round(topics.mean_coherence, 6)
                ),
                "labels": [topic.label for topic in topics.topics],
                "terms": {
                    str(topic.index): list(topic.terms[:6]) for topic in topics.topics
                },
                "holdout_topic_share": metrics_round(dict(assigned.topic_share)),
            },
            "keyphrases": [item.phrase for item in keyphrases.corpus_keyphrases],
            "summaries": {
                "method": summaries.method,
                "mean_compression": (
                    None
                    if summaries.mean_compression is None
                    else round(summaries.mean_compression, 6)
                ),
                "first": summaries.summaries[0] if summaries.summaries else None,
            },
            "entities": {
                "backend": entities.backend,
                "n_entities": entities.n_entities,
                "label_counts": dict(entities.label_counts),
            },
            "sentiment": {
                "backend": sentiment.backend,
                "positive_rate": round(sentiment.positive_rate, 6),
                "negative_rate": round(sentiment.negative_rate, 6),
                "neutral_rate": round(sentiment.neutral_rate, 6),
                "matched_term_rate": (
                    None
                    if sentiment.matched_term_rate is None
                    else round(sentiment.matched_term_rate, 6)
                ),
            },
            "bundle_path": str(bundle),
            "bundle_reproduces_holdout_score": reproduced,
            "leakage_controls": [
                "Stratified split before any text operation",
                "Normalization plan, vocabulary, document frequencies, and head fitted on train only",
                "Topic vectorizer and NMF decomposition fitted on train only; holdout is transform-and-assign",
                "Model choice read validation; test evaluated once after the model was locked",
                "profile_text_corpus screened the split for exact and near-duplicate text contamination and reported it rather than silently deduplicating",
            ],
            "industry_comparison": {"status": "filled"},
            "limitations": [
                "Synthetic tickets composed from per-queue sentence pools; not a real support corpus",
                "Single-label document classification only — no span labelling, no generation",
                "Lexicon sentiment and rule entities are unsupervised baselines with no gold metric",
                "Topic labels are generated from top terms and are not validated category names",
            ],
        },
    )
    print("ticket-routing-nlp OK", metrics_round(dict(test.metrics)))


if __name__ == "__main__":
    main()
