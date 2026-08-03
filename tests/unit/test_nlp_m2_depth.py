"""Depth tests for the NLP domain's primitives and honesty guarantees.

The slice tests cover the Session surface. These cover the pieces underneath it:
normalization determinism, the tokenizer's contract, lexicon rules, NPMI
coherence, graph summarization, rule entity precision, native language ID, and
the bundle boundary.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from buildml.core.errors import ValidationError
from buildml.nlp.catalog import (
    backend_available,
    list_nlp_backends,
    nlp_capability_matrix,
    resolve_backend_estimator,
    resolve_entity_backend,
    resolve_language_backend,
)
from buildml.nlp.entities import compile_gazetteers, extract_rule_entities
from buildml.nlp.explain_hooks import NLP_OPERATION_IDS, nlp_status
from buildml.nlp.keyphrases import extract_keyphrases
from buildml.nlp.language import (
    UNDETERMINED,
    detect_document_language,
    script_shares,
)
from buildml.nlp.lexicons import (
    RULE_ENTITY_LABELS,
    SCRIPT_LABELS,
    SENTIMENT_LEXICON,
    SUPPORTED_STOPWORD_LANGUAGES,
    stopwords_for,
)
from buildml.nlp.normalize import (
    VALID_NORMALIZE_STEPS,
    build_analyzer,
    build_normalize_plan,
    normalize_document,
    split_sentences,
    tokenize_document,
)
from buildml.nlp.sentiment import score_document
from buildml.nlp.summarize import summarize_document
from buildml.nlp.topics import npmi_coherence
from buildml.nlp.types import TextNormalizeConfig

_TICKETS = (
    (
        "Invoice INV-44821 charged the annual fee twice on the same card. "
        "Finance flagged the discrepancy during reconciliation. "
        "Please reverse the duplicate and reissue a corrected invoice.",
        "billing",
    ),
    (
        "The shipment left the depot on Monday and arrived nine days late. "
        "Two cartons were crushed and the outer seal was already broken. "
        "Please send a replacement carton on an expedited service.",
        "shipping",
    ),
    (
        "Single sign-on stopped working for the entire workspace this morning. "
        "Password resets arrive but every link has already expired. "
        "Please restore access for the affected users before the audit.",
        "account",
    ),
    (
        "The hinge on unit HW-1180 snapped within a week of light use. "
        "Diagnostics report no fault yet the device powers off regardless. "
        "Please advise on a warranty replacement rather than another repair.",
        "hardware",
    ),
)


@pytest.fixture
def nlp_frame_session():
    """Small labeled ticket Session with a split, shared by the depth tests."""
    import pandas as pd

    from buildml import Session

    rows = [
        {"body": body, "queue": queue}
        for index in range(30)
        for body, queue in (_TICKETS[index % len(_TICKETS)],)
    ]
    return (
        Session.ingest(pd.DataFrame(rows))
        .set_roles({"body": "feature", "queue": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )


# --------------------------------------------------------------------------
# Normalization and tokenization
# --------------------------------------------------------------------------


def test_normalization_steps_run_in_declared_order_and_are_deterministic() -> None:
    plan = build_normalize_plan(
        TextNormalizeConfig(
            steps=(
                "strip_html",
                "strip_urls",
                "strip_emails",
                "lowercase",
                "strip_accents",
                "collapse_repeats",
                "collapse_whitespace",
            )
        )
    )
    raw = (
        "<p>Visit https://example.com or mail Ana@Example.COM &mdash; "
        "SOOOOO   Café!!!</p>"
    )
    first = normalize_document(raw, plan)
    assert first == normalize_document(raw, plan)
    assert "example.com" not in first
    assert "@" not in first
    assert "<p>" not in first
    assert "cafe" in first
    assert "soo" in first and "sooo" not in first  # 3+ repeats collapse to 2
    assert "  " not in first


def test_unknown_normalization_step_and_bad_lengths_are_refused() -> None:
    with pytest.raises(ValidationError, match="Unknown normalization step"):
        build_normalize_plan(TextNormalizeConfig(steps=("lowercase", "translate")))
    with pytest.raises(ValidationError, match="min_token_length"):
        build_normalize_plan(TextNormalizeConfig(min_token_length=0))
    with pytest.raises(ValidationError, match="max_token_length"):
        build_normalize_plan(
            TextNormalizeConfig(min_token_length=5, max_token_length=3)
        )
    assert "lowercase" in VALID_NORMALIZE_STEPS


def test_tokenizer_drops_punctuation_keeps_symbols_and_respects_flags() -> None:
    keep = build_normalize_plan(TextNormalizeConfig())
    tokens = tokenize_document("Don't panic -- costs rose 12% \u2b50", keep)
    assert "don't" in tokens
    assert "," not in tokens and "." not in tokens and "-" not in tokens
    assert "12" in tokens
    assert "\u2b50" in tokens  # pictographic symbol survives

    strict = build_normalize_plan(
        TextNormalizeConfig(keep_numbers=False, keep_emoji=False)
    )
    strict_tokens = tokenize_document("Don't panic -- costs rose 12% \u2b50", strict)
    assert "12" not in strict_tokens
    assert "\u2b50" not in strict_tokens
    assert "don't" in strict_tokens


def test_stopwords_and_length_filters_apply_after_tokenization() -> None:
    plan = build_normalize_plan(
        TextNormalizeConfig(
            stopword_language="en",
            stopwords=["invoice"],
            min_token_length=3,
        )
    )
    tokens = tokenize_document("The invoice is a very long reconciliation", plan)
    assert "the" not in tokens
    assert "is" not in tokens
    assert "invoice" not in tokens
    assert "reconciliation" in tokens
    assert any("caller-supplied" in note for note in plan.disclosures)


def test_native_stemming_collapses_variants_without_nltk() -> None:
    plan = build_normalize_plan(TextNormalizeConfig(stem=True))
    assert plan.stem_backend in {"nltk-porter", "native-suffix"}
    shipping = tokenize_document("shipping shipped shipments", plan)
    assert len(set(shipping)) < 3


def test_analyzer_is_picklable_and_emits_requested_ngrams() -> None:
    plan = build_normalize_plan(TextNormalizeConfig(stem=True))
    analyzer = build_analyzer(plan, ngram_range=(1, 2))
    grams = analyzer("late delivery again")
    assert "late" in grams
    assert "late delivery" in grams

    restored = pickle.loads(pickle.dumps(analyzer))
    assert restored.ngram_range == (1, 2)
    assert restored("late delivery again") == grams

    with pytest.raises(ValidationError, match="ngram_range"):
        build_analyzer(plan, ngram_range=(2, 1))


def test_sentence_splitter_is_abbreviation_aware() -> None:
    text = (
        "Dr. Nwosu approved the credit on Jan. 4. "
        "The invoice was reissued the next morning. "
        "Finance closed the case."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Dr. Nwosu")
    assert split_sentences("") == []
    assert len(split_sentences(text, max_sentences=2)) == 2


def test_stopword_lists_cover_declared_languages_only() -> None:
    for language in SUPPORTED_STOPWORD_LANGUAGES:
        assert len(stopwords_for(language)) > 20
    assert stopwords_for("EN-GB") == stopwords_for("en")
    with pytest.raises(ValidationError, match="[Nn]o built-in stopword list"):
        stopwords_for("xx")


# --------------------------------------------------------------------------
# Catalog resolution
# --------------------------------------------------------------------------


def test_catalog_resolution_defaults_and_refusals() -> None:
    backend, estimator = resolve_backend_estimator(backend=None, estimator=None)
    assert backend == "sklearn"
    assert estimator == "logistic"

    with pytest.raises(ValidationError, match="Unknown NLP backend"):
        resolve_backend_estimator(backend="telepathy", estimator=None)
    with pytest.raises(ValidationError, match="is not valid for"):
        resolve_backend_estimator(backend="sklearn", estimator="random_forest")
    with pytest.raises(ValidationError):
        resolve_entity_backend("regex")
    with pytest.raises(ValidationError):
        resolve_language_backend("fasttext")

    assert resolve_entity_backend(None) == "rules"
    assert resolve_language_backend(None) == "native"
    assert "sklearn" in list_nlp_backends(available_only=True)
    assert backend_available("sklearn") is True
    assert backend_available("nonexistent") is False


def test_capability_matrix_names_boundaries_and_non_goals() -> None:
    matrix = nlp_capability_matrix()
    assert set(matrix["backends"]) == {"sklearn", "embedding", "transformer"}
    assert matrix["tasks"]["summarization"]["kind"] == "extractive only"
    assert matrix["tasks"]["keyphrase_extraction"]["available"] is True
    assert "rag" in matrix["nlp_vs_neighbours"]
    assert "nlp_bundle" in matrix["nlp_vs_neighbours"]["boundary"]
    assert any("retrieval" in item.lower() for item in matrix["non_goals"])
    assert matrix["tasks"]["sentiment"]["lexicon_terms"] == len(SENTIMENT_LEXICON)


# --------------------------------------------------------------------------
# Lexicon sentiment rules
# --------------------------------------------------------------------------


def test_lexicon_rules_handle_negation_intensity_and_contrast() -> None:
    positive, label, matched = score_document("The service was excellent.")
    assert label == "positive" and positive > 0 and matched >= 1

    negated, negated_label, _ = score_document("The service was not excellent.")
    assert negated_label != "positive"
    assert negated < positive

    intense, _, _ = score_document("The service was extremely excellent.")
    assert intense > positive

    shouted, _, _ = score_document("The service was EXCELLENT!")
    assert shouted > positive

    contrast, _, _ = score_document(
        "The packaging was terrible but the support was excellent."
    )
    assert contrast > 0  # the clause after 'but' carries the verdict

    empty, empty_label, empty_matched = score_document("Reference TKT-4477.")
    assert empty == 0.0 and empty_label == "neutral" and empty_matched == 0


def test_lexicon_scores_stay_bounded() -> None:
    piled_on = "excellent " * 60
    score, label, _ = score_document(piled_on)
    assert -1.0 <= score <= 1.0
    assert label == "positive"


# --------------------------------------------------------------------------
# Topic coherence
# --------------------------------------------------------------------------


def test_npmi_coherence_ranks_co_occurring_terms_above_disjoint_ones() -> None:
    # Terms 0 and 1 co-occur in the first half and are jointly absent from the
    # second; terms 2 and 3 partition the corpus and never co-occur.
    rows = [
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    matrix = sparse.csr_matrix(np.array(rows, dtype=float))
    together = npmi_coherence([0, 1], matrix)
    apart = npmi_coherence([2, 3], matrix)
    assert together is not None and apart is not None
    assert together == pytest.approx(1.0, abs=1e-6)
    assert apart == pytest.approx(-1.0)
    assert -1.0 <= apart < together <= 1.0
    assert npmi_coherence([0], matrix) is None
    assert npmi_coherence([0, 1], sparse.csr_matrix(np.ones((1, 4)))) is None


# --------------------------------------------------------------------------
# Extractive summarization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["textrank", "lexrank", "lead"])
def test_summarize_document_selects_original_sentences_in_reading_order(
    method: str,
) -> None:
    plan = build_normalize_plan(TextNormalizeConfig(stopword_language="en"))
    document = (
        "The depot released the shipment on Monday. "
        "Customs held the container for two days. "
        "The courier rerouted through a second hub. "
        "The shipment reached the site on Friday."
    )
    summary, indices, n_input = summarize_document(
        document,
        method=method,
        n_sentences=2,
        normalize_plan=plan,
        max_input_sentences=50,
    )
    assert n_input == 4
    assert len(indices) == 2
    assert list(indices) == sorted(indices)
    sentences = split_sentences(document)
    for index in indices:
        assert sentences[index] in summary


def test_short_documents_are_returned_unchanged() -> None:
    plan = build_normalize_plan(TextNormalizeConfig(stopword_language="en"))
    summary, indices, n_input = summarize_document(
        "One sentence only.",
        method="textrank",
        n_sentences=3,
        normalize_plan=plan,
        max_input_sentences=50,
    )
    assert n_input == 1
    assert indices == (0,)
    assert summary == "One sentence only."

    blank, blank_indices, blank_input = summarize_document(
        "   ",
        method="lexrank",
        n_sentences=2,
        normalize_plan=plan,
        max_input_sentences=50,
    )
    assert blank == "" and blank_indices == () and blank_input == 0


# --------------------------------------------------------------------------
# Rule entity extraction
# --------------------------------------------------------------------------


def test_rule_entities_recognize_structured_mentions_with_exact_offsets() -> None:
    document = (
        "Contact ana.diaz@example.com or see https://example.com/help. "
        "Invoice INV-448210 charged $1,250 (12.5%) on 2024-03-14 at 09:45 AM "
        "from 10.0.0.14. Ms. Ana Diaz at Meridian Holdings signed off."
    )
    spans = extract_rule_entities(document)
    labels = {span.label for span in spans}
    for expected in ("EMAIL", "URL", "MONEY", "PERCENT", "DATE", "TIME", "ID"):
        assert expected in labels, expected
    for span in spans:
        assert document[span.start : span.end] == span.text
        assert span.source == "rules"
    starts = [span.start for span in spans]
    assert starts == sorted(starts)
    # Overlap resolution keeps one span per region.
    for left, right in zip(spans, spans[1:], strict=False):
        assert left.end <= right.start


def test_rule_entities_are_precision_first_about_free_form_names() -> None:
    spans = extract_rule_entities("ana diaz called about the crushed carton")
    assert spans == ()
    assert "PERSON" in RULE_ENTITY_LABELS  # titled names only


def test_gazetteers_match_whole_words_case_insensitively() -> None:
    compiled = compile_gazetteers({"queue_term": ["Invoice", "portal"]})
    spans = extract_rule_entities(
        "The invoice in the PORTAL, not the portalgun.", gazetteers=compiled
    )
    texts = [span.text.lower() for span in spans]
    assert "invoice" in texts
    assert "portal" in texts
    assert "portalgun" not in texts
    assert all(span.label == "QUEUE_TERM" for span in spans)

    with pytest.raises(ValidationError, match="is empty"):
        compile_gazetteers({"empty": []})
    assert compile_gazetteers(None) == ()


def test_label_filter_restricts_rule_output() -> None:
    document = "Invoice INV-448210 charged $1,250 on 2024-03-14."
    only_money = extract_rule_entities(document, labels=("MONEY",))
    assert {span.label for span in only_money} == {"MONEY"}


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------


def test_script_shares_identify_non_latin_blocks() -> None:
    assert script_shares("") == {}
    cyrillic = script_shares("Доставка задерживается")
    assert cyrillic.get("ru", 0.0) > 0.9
    assert SCRIPT_LABELS["ru"] == "cyrillic"
    assert script_shares("plain latin text") == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The shipment did not arrive and the invoice is still wrong.", "en"),
        ("La commande n'est pas arrivee et la facture est encore fausse.", "fr"),
        ("El pedido no ha llegado y la factura sigue siendo incorrecta.", "es"),
        ("Die Lieferung ist nicht angekommen und die Rechnung ist falsch.", "de"),
    ],
)
def test_native_language_detection_on_function_words(text: str, expected: str) -> None:
    code, confidence = detect_document_language(text)
    assert code == expected
    assert 0.0 < confidence <= 1.0


def test_detection_refuses_to_guess_without_evidence() -> None:
    short_code, short_confidence = detect_document_language("ok")
    assert short_code == UNDETERMINED
    assert short_confidence == 0.0

    marker_free, _ = detect_document_language("TKT-4477 INV-9912 HW-1180 ORD-2231")
    assert marker_free == UNDETERMINED


# --------------------------------------------------------------------------
# Keyphrase primitives
# --------------------------------------------------------------------------


def test_keyphrase_methods_disagree_but_all_stay_within_the_vocabulary(
    nlp_frame_session,
) -> None:
    session = nlp_frame_session
    rankings = {}
    for method in ("tfidf", "rake", "textrank"):
        result = extract_keyphrases(
            session.dataset,
            session._split_plan,
            partition="train",
            method=method,
            top_n=6,
            per_document=False,
        )
        rankings[method] = [item.phrase for item in result.corpus_keyphrases]
        assert result.method == method
        assert result.document_keyphrases == ()
        assert any("no gold" in note.lower() or "unsupervised" in note.lower()
                   for note in result.disclosures)
    corpus = " ".join(
        session.dataset._ensure_pandas()["body"].astype(str).str.lower().tolist()
    )
    for phrases in rankings.values():
        assert phrases
        for phrase in phrases:
            for word in phrase.split():
                assert word in corpus


# --------------------------------------------------------------------------
# Explain hooks
# --------------------------------------------------------------------------


def test_nlp_status_is_factual_when_nothing_is_fitted() -> None:
    empty = nlp_status()
    assert empty["enabled"] is False
    assert empty["present"] is False
    assert empty["has_text_plan"] is False
    assert "rag" in empty["boundary"].lower()

    described = nlp_status(
        history=[{"operation_id": "extract_keyphrases"}, {"operation_id": "eda"}]
    )
    assert described["enabled"] is False
    assert described["present"] is True
    assert any("hold no fitted state" in note for note in described["disclosures"])
    assert "extract_keyphrases" in NLP_OPERATION_IDS


def test_bundle_round_trip_preserves_the_normalization_plan(
    nlp_frame_session, tmp_path: Path
) -> None:
    from buildml.nlp.checkpoint import load_nlp_bundle, save_nlp_bundle

    session = nlp_frame_session
    session.fit_text_classifier(
        text_column="body", stopword_language="en", stem=True, ngram_range=(1, 2)
    )
    plan = session.nlp_text_plan
    out = save_nlp_bundle(tmp_path / "bundle", plan)
    text_plan, topic_plan = load_nlp_bundle(out, trusted=True)
    assert topic_plan is None
    assert text_plan is not None
    assert text_plan.normalize_plan.to_dict() == plan.normalize_plan.to_dict()
    assert text_plan.feature_names_ == plan.feature_names_
    assert math.isclose(
        float(np.asarray(text_plan.estimator_.coef_).sum()),
        float(np.asarray(plan.estimator_.coef_).sum()),
        rel_tol=1e-12,
    )

    with pytest.raises(ValidationError):
        load_nlp_bundle(tmp_path / "does-not-exist", trusted=True)
