"""Session-facing slice tests for the natural-language processing domain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.explain.concepts import CONCEPT_NOTES

NLP_OPERATIONS = (
    "nlp_capability_matrix",
    "profile_text_corpus",
    "detect_language",
    "fit_text_classifier",
    "predict_text",
    "evaluate_text_classifier",
    "interpret_text_prediction",
    "fit_topics",
    "assign_topics",
    "extract_keyphrases",
    "analyze_sentiment",
    "extract_entities",
    "summarize_text",
    "save_nlp_bundle",
    "load_nlp_bundle",
)

NLP_CONCEPTS = (
    "nlp-document-representation",
    "nlp-text-normalization",
    "nlp-token-attribution",
    "nlp-topic-models",
    "nlp-keyphrases-vs-topics",
    "nlp-lexicon-sentiment",
    "nlp-rule-vs-statistical-ner",
    "nlp-extractive-summarization",
    "nlp-language-identification",
    "nlp-corpus-contamination",
    "nlp-vs-rag",
    "nlp-bundle-boundary",
)

_POSITIVE = (
    "The delivery arrived ahead of schedule and the packaging was spotless.",
    "Support answered within minutes and resolved the invoice question politely.",
    "Setup took five minutes and every step of the onboarding portal was clear.",
    "Build quality is excellent; the hinge feels solid after months of daily use.",
)
_NEGATIVE = (
    "The delivery arrived nine days late and the packaging was crushed.",
    "Support ignored three tickets and the invoice still shows a duplicate charge.",
    "Setup failed twice and the onboarding portal contradicted its own instructions.",
    "Build quality is poor; the hinge snapped within a week of light use.",
)


def _text_frame(n: int = 240, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(n):
        positive = bool(index % 2)
        pool = _POSITIVE if positive else _NEGATIVE
        base = str(pool[int(rng.integers(0, len(pool)))])
        tail = str(rng.choice(["", " Reference TKT-10422.", " Filed via the portal."]))
        rows.append(
            {
                "review": f"{base}{tail}",
                "channel": str(rng.choice(["web", "app", "phone"])),
                "sentiment": "positive" if positive else "negative",
            }
        )
    return pd.DataFrame(rows)


def _text_session(seed: int = 5) -> Session:
    return (
        Session.ingest(_text_frame(seed=seed))
        .set_roles(
            {
                "review": "feature",
                "channel": "feature",
                "sentiment": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.15, random_state=0, stratify=True)
    )


def test_public_surface_catalog_concepts_and_ai_tools() -> None:
    import buildml.nlp as nlp

    for name in (
        "fit_text_classifier",
        "predict_text",
        "evaluate_text_classifier",
        "interpret_text_prediction",
        "fit_topics",
        "assign_topics",
        "extract_keyphrases",
        "analyze_sentiment",
        "extract_entities",
        "summarize_text",
        "detect_language",
        "profile_text_corpus",
        "save_nlp_bundle",
        "load_nlp_bundle",
        "nlp_capability_matrix",
        "build_normalize_plan",
        "normalize_document",
        "tokenize_document",
        "split_sentences",
    ):
        assert callable(getattr(nlp, name)), name

    for name in NLP_OPERATIONS:
        assert hasattr(Session, name), name
        assert name in OPERATION_CATALOG, name

    for key in NLP_CONCEPTS:
        assert key in CONCEPT_NOTES, key

    registry = build_default_registry()
    for name in NLP_OPERATIONS:
        assert registry.get(name) is not None, name

    links = OPERATION_CATALOG["fit_text_classifier"].concept_links
    assert "nlp-document-representation" in links
    assert "nlp-vs-rag" in links


def test_every_nlp_tool_reaches_a_session_method(tmp_path: Path) -> None:
    """A registered tool with no executor branch dies as 'No dispatch handler'.

    Registration and dispatch are separate lists in the AI layer, so the only
    way to know a tool is callable is to call it. Every NLP tool is exercised
    here in workflow order against a real Session.
    """
    from buildml.ai.executor import execute_tool, propose_tool_execution

    session = _text_session().ai_configure(provider="mock")
    registry = build_default_registry()
    calls: tuple[tuple[str, dict[str, object]], ...] = (
        ("nlp_capability_matrix", {}),
        ("profile_text_corpus", {"near_duplicate_threshold": 0.95}),
        ("detect_language", {"partition": "all"}),
        ("fit_text_classifier", {"estimator": "logistic", "random_state": 0}),
        ("predict_text", {"partition": "test"}),
        ("evaluate_text_classifier", {"partition": "test"}),
        ("interpret_text_prediction", {"partition": "test", "max_documents": 2}),
        ("fit_topics", {"method": "nmf", "n_topics": 2, "random_state": 0}),
        ("assign_topics", {"partition": "test"}),
        ("extract_keyphrases", {"partition": "train", "top_n": 5}),
        ("analyze_sentiment", {"partition": "test"}),
        ("extract_entities", {"partition": "test", "max_documents": 3}),
        ("summarize_text", {"partition": "test", "max_documents": 3}),
        ("save_nlp_bundle", {"path": str(tmp_path / "tool_bundle")}),
        ("load_nlp_bundle", {"path": str(tmp_path / "tool_bundle")}),
    )
    for name, arguments in calls:
        proposal = propose_tool_execution(name, arguments, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)
        assert result.error is None, f"{name}: {result.error}"

    # Write tools must also announce what they will change before running.
    preview = propose_tool_execution("fit_text_classifier", {}, registry)
    assert any("train only" in change for change in preview.expected_changes)


def test_capability_matrix_is_honest_about_extras() -> None:
    matrix = Session.nlp_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert matrix["backends"]["sklearn"]["token_attributions"] is True
    assert matrix["backends"]["embedding"]["token_attributions"] is False
    assert matrix["default_backend_when_installed"] == "sklearn"
    assert "buildml[nlp]" in matrix["install_hints"]["nlp"]
    assert any("generative" in item.lower() for item in matrix["non_goals"])


def test_fit_predict_evaluate_interpret_and_bundle(tmp_path: Path) -> None:
    session = _text_session()

    profile = session.profile_text_corpus(near_duplicate_threshold=0.95)
    assert profile.text_column == "review"
    assert profile.n_documents == 240
    assert profile.vocabulary_size > 0
    assert session.nlp_profile_result is profile

    fit = session.fit_text_classifier(estimator="logistic", stopword_language=None)
    assert fit.backend == "sklearn"
    assert fit.estimator == "logistic"
    assert fit.text_column == "review"
    assert fit.vocabulary_size == fit.n_features > 0
    assert set(fit.classes) == {"negative", "positive"}
    assert session.nlp_text_plan is not None

    ev = session.evaluate_text_classifier(partition="validation")
    assert ev.partition == "validation"
    assert 0.0 <= ev.metrics["accuracy"] <= 1.0
    assert "balanced_accuracy" in ev.metrics
    assert len(ev.confusion) == len(ev.classes)
    assert ev.oov_rate is not None

    pred = session.predict_text(partition="test")
    assert pred.n_rows == len(pred.predictions)
    assert pred.probabilities  # logistic exposes calibrated probabilities
    assert len(pred.probabilities[0]) == len(pred.classes)

    interpret = session.interpret_text_prediction(partition="test", max_documents=4)
    assert interpret.n_documents == 4
    assert interpret.method == "linear-coefficient x feature-value"
    assert len(interpret.document_attributions) == 4
    assert set(interpret.global_top_tokens) == {"negative", "positive"}
    first = interpret.document_attributions[0][0]
    assert first.contribution == pytest.approx(first.weight * first.value)

    bundle = tmp_path / "nlp_bundle"
    session.save_nlp_bundle(bundle)
    assert (bundle / "meta.json").is_file()
    assert (bundle / "nlp_text_plan.joblib").is_file()

    other = _text_session()
    other.load_nlp_bundle(bundle, trusted=True)
    assert other.nlp_text_plan is not None
    assert other.nlp_text_plan.estimator == "logistic"
    reloaded = other.evaluate_text_classifier(partition="test")
    assert "accuracy" in reloaded.metrics


def test_margin_only_head_reports_missing_probabilities() -> None:
    session = _text_session()
    session.fit_text_classifier(estimator="linear_svm", stopword_language=None)
    pred = session.predict_text(partition="test")
    assert pred.probabilities == ()
    assert any("predict_proba" in item for item in pred.warnings)

    ev = session.evaluate_text_classifier(partition="validation")
    assert "log_loss" not in ev.metrics
    assert "roc_auc" not in ev.metrics


def test_interpret_refuses_representations_without_a_vocabulary() -> None:
    session = _text_session()
    session.fit_text_classifier(vectorizer="hashing", estimator="logistic")
    assert session.nlp_fit_result.vocabulary_size == 0
    with pytest.raises(ValidationError, match="invertible vocabulary"):
        session.interpret_text_prediction(partition="test")


def test_topics_fit_on_train_and_assign_is_transform_only() -> None:
    session = _text_session()
    topics = session.fit_topics(method="nmf", n_topics=3, min_df=2, top_terms=5)
    assert topics.method == "nmf"
    assert len(topics.topics) == 3
    assert all(len(topic.terms) == 5 for topic in topics.topics)
    assert topics.mean_coherence is not None
    assert topics.reconstruction_error is not None

    model_before = session.nlp_topic_plan.model_
    assigned = session.assign_topics(partition="test")
    assert assigned.n_topics == 3
    assert assigned.n_rows == len(assigned.dominant_topics)
    assert sum(assigned.topic_share.values()) == pytest.approx(1.0)
    assert session.nlp_topic_plan.model_ is model_before

    lda = session.fit_topics(method="lda", n_topics=3, min_df=2, max_iter=20)
    assert lda.perplexity is not None


@pytest.mark.parametrize("method", ["tfidf", "rake", "textrank"])
def test_keyphrases_produce_alphabetic_candidates(method: str) -> None:
    session = _text_session()
    result = session.extract_keyphrases(partition="train", method=method, top_n=8)
    assert result.method == method
    assert result.corpus_keyphrases
    for phrase in result.corpus_keyphrases:
        assert phrase.phrase.strip() == phrase.phrase
        assert not phrase.phrase.isdigit()
        assert phrase.document_frequency >= 1
    assert result.document_keyphrases


def test_sentiment_lexicon_and_supervised_backends() -> None:
    session = _text_session()
    lexicon = session.analyze_sentiment(
        partition="test", backend="lexicon", compare_to_target=True
    )
    assert lexicon.backend == "lexicon"
    assert lexicon.matched_term_rate is not None
    total = lexicon.positive_rate + lexicon.negative_rate + lexicon.neutral_rate
    assert total == pytest.approx(1.0)
    assert lexicon.agreement["n_compared"] == lexicon.n_rows

    with pytest.raises(ValidationError, match="fit_text_classifier"):
        session.analyze_sentiment(partition="test", backend="supervised")

    session.fit_text_classifier(estimator="logistic", stopword_language=None)
    supervised = session.analyze_sentiment(partition="test", backend="supervised")
    assert supervised.backend == "supervised"
    assert supervised.matched_term_rate is None


def test_entities_rules_and_gazetteers() -> None:
    session = _text_session()
    result = session.extract_entities(
        partition="test",
        backend="rules",
        gazetteers={"PRODUCT": ["portal", "invoice"]},
        max_documents=5,
    )
    assert result.backend == "rules"
    assert result.n_entities > 0
    assert len(result.document_entities) == 5
    assert any(
        span.source in {"rules", "gazetteer"}
        for mentions in result.document_entities
        for span in mentions
    )
    with pytest.raises(ValidationError, match="not produced by the rules backend"):
        session.extract_entities(partition="test", labels=["GPE"])


@pytest.mark.parametrize("method", ["textrank", "lexrank", "lead"])
def test_summaries_only_reuse_document_sentences(method: str) -> None:
    frame = pd.DataFrame(
        {
            "note": [
                (
                    "The shipment left the depot on Monday. "
                    "A customs hold delayed it by two days. "
                    "The courier rerouted through a second hub. "
                    "It finally arrived on Friday afternoon."
                ),
                (
                    "Billing raised the monthly rate without notice. "
                    "The invoice listed a fee nobody could explain. "
                    "Support escalated the case to finance. "
                    "A credit was issued the following week."
                ),
            ]
            * 20,
            "label": ["logistics", "billing"] * 20,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"note": "feature", "label": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    result = session.summarize_text(partition="test", method=method, n_sentences=2)
    assert result.method == method
    assert result.summaries
    assert all(len(indices) <= 2 for indices in result.selected_sentence_indices)
    assert result.mean_compression is not None and result.mean_compression < 1.0
    documents = [str(doc) for doc in session.dataset._ensure_pandas()["note"]]
    for summary in result.summaries:
        # Extractive means every summary sentence is copied verbatim from a document.
        first_sentence = summary.split(". ")[0].strip()
        assert any(first_sentence in doc for doc in documents)


def test_language_detection_reports_undetermined_instead_of_guessing() -> None:
    session = _text_session()
    result = session.detect_language(partition="all", backend="native")
    assert result.dominant_language == "en"
    assert result.language_counts["en"] > 0
    assert 0.0 <= result.undetermined_rate <= 1.0

    short = pd.DataFrame({"txt": ["ok", "no", "yes", "hm"] * 10, "y": [0, 1] * 20})
    tiny = Session.ingest(short).set_roles({"txt": "feature", "y": "target"})
    tiny_result = tiny.detect_language(partition="all")
    assert tiny_result.language_counts.get("und", 0) == 40
    assert tiny_result.undetermined_rate == pytest.approx(1.0)


def test_fit_requires_a_split() -> None:
    session = Session.ingest(_text_frame(n=40)).set_roles(
        {"review": "feature", "channel": "feature", "sentiment": "target"}
    )
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_text_classifier()
    with pytest.raises((ValidationError, LeakageError)):
        session.fit_topics(n_topics=2)


def test_operations_requiring_a_plan_refuse_without_one() -> None:
    session = _text_session()
    with pytest.raises(ValidationError, match="No NLP text plan"):
        session.predict_text(partition="test")
    with pytest.raises(ValidationError, match="No NLP text plan"):
        session.evaluate_text_classifier(partition="validation")
    with pytest.raises(ValidationError, match="No NLP text plan"):
        session.interpret_text_prediction(partition="test")
    with pytest.raises(ValidationError, match="No NLP topic plan"):
        session.assign_topics(partition="test")
    with pytest.raises(ValidationError, match="No NLP plan to save"):
        session.save_nlp_bundle("unused")


def test_ambiguous_and_missing_text_columns_are_refused() -> None:
    frame = pd.DataFrame(
        {
            "left": ["a fairly long sentence about shipping delays"] * 30,
            "right": ["a fairly long sentence about billing disputes"] * 30,
            "y": ([0, 1] * 15),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"left": "feature", "right": "feature", "y": "target"})
        .split(test_size=0.3, random_state=0)
    )
    with pytest.raises(ValidationError, match="text_column"):
        session.fit_text_classifier()
    assert session.fit_text_classifier(text_column="left").text_column == "left"

    with pytest.raises(ValidationError, match="not a dataset column"):
        session.extract_keyphrases(text_column="missing")


def test_history_and_walkthrough_disclose_nlp_state() -> None:
    session = _text_session()
    session.profile_text_corpus()
    session.fit_text_classifier(estimator="logistic", stopword_language=None)
    session.evaluate_text_classifier(partition="validation")
    session.fit_topics(n_topics=2, min_df=2)

    operations = [record["operation_id"] for record in session.history]
    for expected in (
        "profile_text_corpus",
        "fit_text_classifier",
        "evaluate_text_classifier",
        "fit_topics",
    ):
        assert expected in operations
    fit_record = next(
        record for record in session.history if record["operation_id"] == "fit_text_classifier"
    )
    assert fit_record["result_summary"]["estimator"] == "logistic"

    report = session.walkthrough()
    payload = report.to_dict()
    assert payload["nlp_status"]["enabled"] is True
    assert payload["nlp_status"]["has_text_plan"] is True
    assert payload["nlp_status"]["has_topic_plan"] is True
    assert "rag" in payload["nlp_status"]["boundary"].lower()
    assert any("train only" in item for item in payload["nlp_status"]["disclosures"])
