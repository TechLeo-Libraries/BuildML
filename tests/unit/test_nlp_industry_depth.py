"""Industry-depth tests for the optional NLP backends.

Each optional path is asserted twice: once for the installed case and once for
the missing case. The point of the domain is that a missing extra produces a
named :class:`MissingExtraError` rather than a silent fallback, and that the
core bag-of-n-grams, lexicon, rule, and graph paths keep working regardless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.nlp.catalog import (
    backend_available,
    list_nlp_backends,
    nlp_capability_matrix,
    resolve_backend_estimator,
)
from buildml.nlp.extras import (
    langdetect_available,
    nlp_industry_available,
    nltk_available,
    require_langdetect,
    require_nltk,
    require_sentence_transformers,
    require_spacy,
    require_transformers,
    sentence_transformers_available,
    spacy_available,
    spacy_model_available,
    spacy_spec_present,
    transformers_available,
    transformers_spec_present,
)
from buildml.nlp.normalize import build_normalize_plan, tokenize_document
from buildml.nlp.types import TextNormalizeConfig

_BODIES = (
    "Invoice INV-44821 charged the annual fee twice and finance flagged it.",
    "The shipment arrived nine days late with two cartons crushed in transit.",
    "Single sign-on stopped working and every password reset link has expired.",
    "The hinge on unit HW-1180 snapped within a week of light daily use.",
)
_QUEUES = ("billing", "shipping", "account", "hardware")


def _session(n: int = 120, seed: int = 3) -> Session:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        slot = index % len(_BODIES)
        rows.append(
            {
                "body": f"{_BODIES[slot]} Case {int(rng.integers(1000, 9999))}.",
                "queue": _QUEUES[slot],
            }
        )
    return (
        Session.ingest(pd.DataFrame(rows))
        .set_roles({"body": "feature", "queue": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )


def test_core_backend_is_always_available_and_default() -> None:
    matrix = nlp_capability_matrix()
    assert matrix["backends"]["sklearn"]["available"] is True
    assert matrix["default_backend_when_installed"] == "sklearn"
    assert "sklearn" in list_nlp_backends(available_only=True)
    assert backend_available("sklearn") is True


def test_capability_matrix_availability_matches_the_extras_probes() -> None:
    matrix = nlp_capability_matrix()
    assert matrix["backends"]["embedding"]["available"] is sentence_transformers_available()
    assert matrix["backends"]["transformer"]["available"] is transformers_available()
    assert matrix["nltk_present"] is nltk_available()
    assert matrix["langdetect_present"] is langdetect_available()
    assert matrix["spacy_present"] is spacy_available()
    assert matrix["spacy_model_present"] is spacy_model_available()
    assert matrix["industry_runtime_present"] is nlp_industry_available()
    assert matrix["industry_extra_present"] is (
        spacy_spec_present() or transformers_spec_present()
    )
    assert matrix["backends"]["embedding"]["extra"] == "nlp"
    assert matrix["backends"]["transformer"]["extra"] == "nlp"


def test_embedding_backend_is_resolved_or_named_as_missing() -> None:
    if sentence_transformers_available():
        backend, estimator = resolve_backend_estimator(
            backend="embedding", estimator=None
        )
        assert (backend, estimator) == ("embedding", "logistic")
        require_sentence_transformers()
    else:
        with pytest.raises(MissingExtraError) as excinfo:
            resolve_backend_estimator(backend="embedding", estimator=None)
        assert excinfo.value.extra == "nlp"
        with pytest.raises(MissingExtraError):
            require_sentence_transformers()


def test_transformer_backend_is_resolved_or_named_as_missing() -> None:
    if transformers_available():
        backend, estimator = resolve_backend_estimator(
            backend="transformer", estimator="linear_svm"
        )
        assert (backend, estimator) == ("transformer", "linear_svm")
        require_transformers()
    else:
        with pytest.raises(MissingExtraError) as excinfo:
            resolve_backend_estimator(backend="transformer", estimator=None)
        assert excinfo.value.extra == "nlp"
        with pytest.raises(MissingExtraError):
            require_transformers()


def test_dense_backends_reject_the_hashing_vectorizer() -> None:
    with pytest.raises((ValidationError, MissingExtraError)):
        resolve_backend_estimator(
            backend="embedding", estimator=None, vectorizer="hashing"
        )


def test_dense_backends_have_no_token_attribution_claim() -> None:
    matrix = nlp_capability_matrix()
    assert matrix["backends"]["embedding"]["token_attributions"] is False
    assert matrix["backends"]["transformer"]["token_attributions"] is False
    assert matrix["backends"]["sklearn"]["token_attributions"] is True


def test_session_embedding_backend_refuses_cleanly_without_the_extra() -> None:
    session = _session()
    if sentence_transformers_available():
        pytest.skip("sentence-transformers installed; covered by the resolved path")
    with pytest.raises(MissingExtraError) as excinfo:
        session.fit_text_classifier(backend="embedding")
    assert "buildml[nlp]" in str(excinfo.value)


def test_langdetect_backend_is_used_or_named_as_missing() -> None:
    session = _session()
    if langdetect_available():
        result = session.detect_language(partition="all", backend="langdetect")
        assert result.backend == "langdetect"
        assert result.dominant_language in {"en", "und"}
        require_langdetect()
    else:
        with pytest.raises(MissingExtraError) as excinfo:
            session.detect_language(partition="all", backend="langdetect")
        assert excinfo.value.extra == "nlp"
    # The native scorer never depends on the extra.
    native = session.detect_language(partition="all", backend="native")
    assert native.backend == "native"


def test_spacy_ner_is_used_or_named_as_missing() -> None:
    session = _session()
    if spacy_model_available():
        result = session.extract_entities(
            partition="test", backend="spacy", max_documents=3
        )
        assert result.backend == "spacy"
        assert any("trained outside this Session" in note for note in result.disclosures)
        require_spacy()
    elif spacy_available():
        # spaCy present but the pipeline package is not downloaded.
        with pytest.raises(ValidationError, match="spacy download"):
            session.extract_entities(partition="test", backend="spacy")
    else:
        with pytest.raises(MissingExtraError) as excinfo:
            session.extract_entities(partition="test", backend="spacy")
        assert excinfo.value.extra == "nlp-industry"
    rules = session.extract_entities(partition="test", backend="rules")
    assert rules.backend == "rules"


def test_transformer_sentiment_is_used_or_named_as_missing() -> None:
    session = _session()
    if not transformers_available():
        with pytest.raises(MissingExtraError) as excinfo:
            session.analyze_sentiment(partition="test", backend="transformer")
        assert excinfo.value.extra == "nlp"
    lexicon = session.analyze_sentiment(partition="test", backend="lexicon")
    assert lexicon.backend == "lexicon"
    assert lexicon.matched_term_rate is not None


def test_nltk_morphology_degrades_to_shipped_suffix_rules() -> None:
    plan = build_normalize_plan(TextNormalizeConfig(stem=True))
    if nltk_available():
        assert plan.stem_backend == "nltk-porter"
        require_nltk()
    else:
        assert plan.stem_backend == "native-suffix"
        assert any("install buildml[nlp]" in note for note in plan.disclosures)
        with pytest.raises(MissingExtraError):
            require_nltk()
    # Either way the plan stems.
    assert len(set(tokenize_document("shipping shipped shipments", plan))) < 3


def test_lemmatization_requires_the_extra_explicitly() -> None:
    if nltk_available():
        pytest.skip("NLTK installed; lemmatization resolves to nltk-wordnet")
    with pytest.raises(MissingExtraError) as excinfo:
        build_normalize_plan(TextNormalizeConfig(lemmatize=True))
    assert excinfo.value.extra == "nlp"


def test_core_path_reaches_a_full_result_with_no_extras_at_all() -> None:
    session = _session()
    session.profile_text_corpus()
    session.fit_text_classifier(estimator="logistic")
    ev = session.evaluate_text_classifier(partition="validation")
    session.interpret_text_prediction(partition="test", max_documents=2)
    session.fit_topics(n_topics=2, min_df=2)
    session.assign_topics(partition="test")
    session.extract_keyphrases(partition="train", method="rake", top_n=5)
    session.summarize_text(partition="test", method="lexrank", max_documents=3)
    assert 0.0 <= ev.metrics["accuracy"] <= 1.0
    matrix = nlp_capability_matrix()
    assert matrix["tasks"]["text_classification"]["available"] is True
    assert matrix["tasks"]["summarization"]["available"] is True
