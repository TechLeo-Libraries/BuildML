"""Natural-language processing Session loop (mirrors quickstart-nlp)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session

QUEUE_SENTENCES: dict[str, tuple[str, ...]] = {
    "billing": (
        "Invoice INV-44821 charged the annual fee twice on the same card.",
        "The renewal quote said one figure but the invoice came to almost double.",
        "A proration credit never appeared against the corrected invoice.",
    ),
    "shipping": (
        "The order was promised for the 3rd and arrived nine days late.",
        "Two of the four cartons in the shipment were crushed in transit.",
        "Tracking has not updated since the shipment left the depot.",
    ),
    "account": (
        "Single sign-on stopped working for the whole workspace this morning.",
        "The onboarding portal rejects the invite link for every new hire.",
        "Password resets arrive but the link has already expired on arrival.",
    ),
}

FOLLOW_UPS: tuple[str, ...] = (
    "Please confirm who is handling this and by when.",
    "The team is blocked and needs a written answer before Friday.",
    "We have already sent these details twice.",
    "This is the third time we are writing about the same problem.",
    "The last update we received simply asked us to wait.",
)


def build_frame(n: int = 300, seed: int = 0, ambiguous_rate: float = 0.15) -> pd.DataFrame:
    """Small labeled ticket corpus: one text column, one queue label.

    An ``ambiguous_rate`` share of tickets is written only from queue-agnostic
    follow-up sentences, so their label is genuinely unrecoverable from the text.
    That keeps the example's accuracy short of a suspicious 1.0.
    """
    rng = np.random.default_rng(seed)
    queues = list(QUEUE_SENTENCES)
    rows = []
    for index in range(n):
        queue = queues[index % len(queues)]
        if rng.random() < ambiguous_rate:
            parts = [str(rng.choice(FOLLOW_UPS)) for _ in range(3)]
        else:
            pool = QUEUE_SENTENCES[queue]
            parts = [
                str(rng.choice(pool)),
                str(rng.choice(pool)),
                str(rng.choice(FOLLOW_UPS)),
            ]
        rows.append({"body": " ".join(parts), "queue": queue})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def new_session(frame: pd.DataFrame) -> Session:
    return (
        Session.ingest(frame.copy())
        .set_roles({"body": "feature", "queue": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )


def main() -> None:
    frame = build_frame()
    session = new_session(frame)

    # 1. What can this installation actually do?
    matrix = Session.nlp_capability_matrix()
    print("backends", {k: v["available"] for k, v in matrix["backends"].items()})

    # 2. Screen the corpus before trusting any score.
    profile = session.profile_text_corpus(near_duplicate_threshold=0.9)
    print(
        "profile",
        profile.n_documents,
        "vocab",
        profile.vocabulary_size,
        "exact_overlap",
        profile.train_holdout_exact_overlap,
        "near_dupes",
        profile.train_holdout_near_duplicate,
    )
    for finding in profile.findings:
        print("  finding:", finding)

    # 3. Fit on train only; the normalization plan is stored with the model.
    fit = session.fit_text_classifier(
        estimator="logistic",
        ngram_range=(1, 2),
        min_df=2,
        class_weight="balanced",
        random_state=0,
    )
    print("fit", fit.backend, fit.estimator, "vocab", fit.vocabulary_size)

    # 4. Choose on validation, then read test once.
    print("validation", session.evaluate_text_classifier(partition="validation").metrics)
    test = session.evaluate_text_classifier(partition="test")
    print("test", test.metrics, "oov", test.oov_rate)

    # 5. Exact token attribution — an identity for a linear head. Explain the row
    #    against the class it was actually predicted as, not an arbitrary default.
    first_prediction = session.predict_text(partition="test").predictions[0]
    interpret = session.interpret_text_prediction(
        partition="test", top_k=5, max_documents=1, target_class=first_prediction
    )
    print("explaining", interpret.target_class, "via", interpret.method)
    for item in interpret.document_attributions[0]:
        print(f"  {item.token:<24} {item.contribution:+.4f}")

    # 6. Unsupervised structure, fitted on train and assigned to holdout.
    topics = session.fit_topics(method="nmf", n_topics=3, min_df=3, random_state=0)
    print("topics", [t.label for t in topics.topics], "coherence", topics.mean_coherence)
    print("holdout share", session.assign_topics(partition="test").topic_share)

    # 7. Description surfaces that claim no quality metric.
    keyphrases = session.extract_keyphrases(partition="train", top_n=6)
    print("keyphrases", [k.phrase for k in keyphrases.corpus_keyphrases])
    summary = session.summarize_text(partition="test", n_sentences=1, max_documents=1)
    print("summary", summary.summaries[0])
    entities = session.extract_entities(
        partition="test", gazetteers={"TERM": ["invoice", "portal", "shipment"]}
    )
    print("entities", entities.label_counts)
    sentiment = session.analyze_sentiment(partition="test")
    print("sentiment", sentiment.negative_rate, "matched", sentiment.matched_term_rate)
    print("language", session.detect_language(partition="all").dominant_language)

    # 8. Bundle carries the normalization plan, so the reload scores identically.
    out = Path("artifacts") / "nlp_demo_bundle"
    session.save_nlp_bundle(out)
    reloaded = new_session(frame)
    reloaded.load_nlp_bundle(out)
    print("reloaded", reloaded.evaluate_text_classifier(partition="test").metrics)


if __name__ == "__main__":
    main()
