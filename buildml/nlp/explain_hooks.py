"""Condense NLP results into the small records that history and reports show.

A session keeps a history of what was done, and a walkthrough describes where
you are. Neither can hold a full result — a prediction result contains a label
for every document, and an interpretation contains every token attribution.

Each function here reduces one result to the handful of fields that belong in a
timeline: what was done, on what, and the numbers a reader would want at a
glance. Everything else stays on the result object.

Two rules run through all of them. They never raise: a missing or malformed
result becomes an empty dict, because a history entry failing is worse than a
history entry being thin. And they report only what happened, never a
recommendation — the teaching prose lives in :mod:`buildml.explain`, and mixing
the two would put opinions in the audit trail.
"""

from __future__ import annotations

from typing import Any

NLP_OPERATION_IDS: frozenset[str] = frozenset(
    {
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
    }
)


def _payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    if isinstance(result, dict):
        return dict(result)
    return {}


def fit_result_summary(fit_result: Any) -> dict[str, Any]:
    """Condense a text-classifier fit into a history entry.

    Keeps what identifies the model and what characterises the data it saw, so
    a later reader can tell two fits apart without reopening either.

    Parameters
    ----------
    fit_result:
        An :class:`~buildml.nlp.results.NlpFitResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Task, backend, head, columns, training size, feature and vocabulary
        counts, classes, and the in-sample score. Empty when there was nothing
        to summarise.

    Notes
    -----
    ``train_score`` is in-sample and near-perfect on text. It is recorded
    because it is part of what happened, not because it is a quality signal.
    """
    payload = _payload(fit_result)
    if not payload:
        return {}
    return {
        "task": payload.get("task"),
        "backend": payload.get("backend"),
        "estimator": payload.get("estimator"),
        "text_column": payload.get("text_column"),
        "target_column": payload.get("target_column"),
        "n_train_rows": payload.get("n_train_rows"),
        "n_features": payload.get("n_features"),
        "vocabulary_size": payload.get("vocabulary_size"),
        "classes": payload.get("classes"),
        "train_score": payload.get("train_score"),
    }


def eval_result_summary(eval_result: Any) -> dict[str, Any]:
    """Condense a holdout evaluation into a history entry.

    Keeps the metrics and the partition they came from, plus the
    out-of-vocabulary rate — the context that decides whether the metrics can
    be trusted. The per-class table and confusion matrix stay on the result.

    Parameters
    ----------
    eval_result:
        An :class:`~buildml.nlp.results.NlpEvalResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Partition, task, row count, the metric mapping, and the
        out-of-vocabulary rate. Empty when there was nothing to summarise.
    """
    payload = _payload(eval_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "task": payload.get("task"),
        "n_rows": payload.get("n_rows"),
        "metrics": payload.get("metrics"),
        "oov_rate": payload.get("oov_rate"),
    }


def predict_result_summary(predict_result: Any) -> dict[str, Any]:
    """Condense a scoring run into a history entry.

    Records that scoring happened and how much of the text the model could
    read. The predictions themselves are the payload, not the record.

    Parameters
    ----------
    predict_result:
        An :class:`~buildml.nlp.results.NlpPredictResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Partition, row and prediction counts, whether probabilities were
        produced, and the out-of-vocabulary rate. Empty when there was nothing
        to summarise.
    """
    payload = _payload(predict_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "n_rows": payload.get("n_rows"),
        "n_predictions": payload.get("n_predictions"),
        "has_probabilities": payload.get("has_probabilities"),
        "oov_rate": payload.get("oov_rate"),
    }


def interpret_result_summary(interpret_result: Any) -> dict[str, Any]:
    """Condense an interpretation run into a history entry.

    Records the scope and, importantly, the attribution method — linear
    coefficients and naive Bayes log-likelihoods are on different scales, so a
    history without the method invites comparing numbers that are not
    comparable.

    Parameters
    ----------
    interpret_result:
        An :class:`~buildml.nlp.results.NlpInterpretResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, document count, target class, and attribution method. The
        attributions themselves are far too large for a history entry. Empty
        when there was nothing to summarise.
    """
    payload = _payload(interpret_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "n_documents": payload.get("n_documents"),
        "target_class": payload.get("target_class"),
        "method": payload.get("method"),
    }


def topic_result_summary(topic_result: Any) -> dict[str, Any]:
    """Condense a topic fit into a history entry.

    Keeps the topic labels rather than only the count, because a list of labels
    is what makes one topic fit recognisable against another in a timeline —
    "six topics" tells you nothing you did not already know.

    Parameters
    ----------
    topic_result:
        An :class:`~buildml.nlp.results.NlpTopicResult`, an equivalent dict, or
        ``None``.

    Returns
    -------
    dict
        Method, topic count, training size, column, mean coherence, and the
        topic labels. Empty when there was nothing to summarise.
    """
    payload = _payload(topic_result)
    if not payload:
        return {}
    return {
        "method": payload.get("method"),
        "n_topics": payload.get("n_topics"),
        "n_train_rows": payload.get("n_train_rows"),
        "text_column": payload.get("text_column"),
        "mean_coherence": payload.get("mean_coherence"),
        "topic_labels": [
            item.get("label") for item in (payload.get("topics") or [])
        ],
    }


def topic_assign_summary(assign_result: Any) -> dict[str, Any]:
    """Condense a topic assignment into a history entry.

    Keeps the topic share, which is small enough for a record and is the field
    that reveals drift when compared against the fit's training mass.

    Parameters
    ----------
    assign_result:
        An :class:`~buildml.nlp.results.NlpTopicAssignResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, method, row and topic counts, and the topic share. The
        per-document weight matrix stays on the result. Empty when there was
        nothing to summarise.
    """
    payload = _payload(assign_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "n_topics": payload.get("n_topics"),
        "topic_share": payload.get("topic_share"),
    }


def keyphrase_result_summary(keyphrase_result: Any) -> dict[str, Any]:
    """Condense a keyphrase extraction into a history entry.

    Keeps the leading corpus phrases, capped at ten. They are the readable
    output — a history entry saying only "extracted 15 keyphrases" would record
    that something happened without recording what.

    Parameters
    ----------
    keyphrase_result:
        An :class:`~buildml.nlp.results.NlpKeyphraseResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, method, row count, ``top_n``, and up to ten corpus phrases.
        Empty when there was nothing to summarise.
    """
    payload = _payload(keyphrase_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "top_n": payload.get("top_n"),
        "top_phrases": [
            item.get("phrase") for item in (payload.get("corpus_keyphrases") or [])[:10]
        ],
    }


def sentiment_result_summary(sentiment_result: Any) -> dict[str, Any]:
    """Condense a sentiment run into a history entry.

    Keeps ``matched_term_rate`` alongside the distribution, deliberately.
    Recording that a corpus scored 70% neutral without recording that the
    lexicon recognised almost none of its vocabulary would leave a misleading
    entry in the audit trail.

    Parameters
    ----------
    sentiment_result:
        An :class:`~buildml.nlp.results.NlpSentimentResult`, an equivalent
        dict, or ``None``.

    Returns
    -------
    dict
        Partition, backend, row count, the three rates, mean score, and the
        matched-term rate. Empty when there was nothing to summarise.
    """
    payload = _payload(sentiment_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "backend": payload.get("backend"),
        "n_rows": payload.get("n_rows"),
        "positive_rate": payload.get("positive_rate"),
        "negative_rate": payload.get("negative_rate"),
        "neutral_rate": payload.get("neutral_rate"),
        "mean_score": payload.get("mean_score"),
        "matched_term_rate": payload.get("matched_term_rate"),
    }


def entity_result_summary(entity_result: Any) -> dict[str, Any]:
    """Condense an entity extraction into a history entry.

    Keeps the per-label counts, which are what let a later reader see that one
    label suddenly fired ten times more often than usual.

    Parameters
    ----------
    entity_result:
        An :class:`~buildml.nlp.results.NlpEntityResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Partition, backend, row count, total mentions, and the label counts.
        The mentions and their spans stay on the result. Empty when there was
        nothing to summarise.
    """
    payload = _payload(entity_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "backend": payload.get("backend"),
        "n_rows": payload.get("n_rows"),
        "n_entities": payload.get("n_entities"),
        "label_counts": payload.get("label_counts"),
    }


def summary_result_summary(summary_result: Any) -> dict[str, Any]:
    """Condense a summarisation run into a history entry.

    Keeps ``mean_compression``, which is the one number that says whether the
    run achieved anything — near 1.0 means the documents were already short
    enough that nothing was summarised.

    Parameters
    ----------
    summary_result:
        An :class:`~buildml.nlp.results.NlpSummaryResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Partition, method, row count, sentences per summary, and mean
        compression. The summary text stays on the result. Empty when there was
        nothing to summarise.
    """
    payload = _payload(summary_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "method": payload.get("method"),
        "n_rows": payload.get("n_rows"),
        "n_sentences": payload.get("n_sentences"),
        "mean_compression": payload.get("mean_compression"),
    }


def language_result_summary(language_result: Any) -> dict[str, Any]:
    """Condense a language-detection run into a history entry.

    Keeps the full language counts rather than just the dominant language,
    since the whole reason to run detection is to find out whether the corpus
    is mixed.

    Parameters
    ----------
    language_result:
        An :class:`~buildml.nlp.results.NlpLanguageResult`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Partition, backend, row count, dominant language, per-language counts,
        and the undetermined rate. Empty when there was nothing to summarise.
    """
    payload = _payload(language_result)
    if not payload:
        return {}
    return {
        "partition": payload.get("partition"),
        "backend": payload.get("backend"),
        "n_rows": payload.get("n_rows"),
        "dominant_language": payload.get("dominant_language"),
        "language_counts": payload.get("language_counts"),
        "undetermined_rate": payload.get("undetermined_rate"),
    }


def profile_result_summary(profile_result: Any) -> dict[str, Any]:
    """Condense a corpus profile into a history entry.

    Deliberately keeps all three contamination measures. If a holdout score is
    later questioned, the history should already contain the evidence about
    whether the split was clean — reconstructing it afterwards means reprofiling
    a corpus that may have changed.

    Parameters
    ----------
    profile_result:
        An :class:`~buildml.nlp.results.NlpCorpusProfile`, an equivalent dict,
        or ``None``.

    Returns
    -------
    dict
        Column, document count, empty rate, vocabulary size, duplicate rate,
        exact and near-duplicate train/holdout overlap, and the holdout
        out-of-vocabulary rate. Empty when there was nothing to summarise.
    """
    payload = _payload(profile_result)
    if not payload:
        return {}
    return {
        "text_column": payload.get("text_column"),
        "n_documents": payload.get("n_documents"),
        "empty_rate": payload.get("empty_rate"),
        "vocabulary_size": payload.get("vocabulary_size"),
        "duplicate_document_rate": payload.get("duplicate_document_rate"),
        "train_holdout_exact_overlap": payload.get("train_holdout_exact_overlap"),
        "train_holdout_near_duplicate": payload.get("train_holdout_near_duplicate"),
        "holdout_oov_token_rate": payload.get("holdout_oov_token_rate"),
    }


def nlp_status(
    text_plan: Any = None,
    *,
    topic_plan: Any = None,
    fit_result: Any = None,
    eval_result: Any = None,
    profile_result: Any = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Describe where the NLP side of a session currently stands.

    Answers "what text modelling has happened here and what should I know about
    it" — for a walkthrough, a status display, or an audit record. It reports
    what is attached, what the history shows, and the caveats that apply, all
    as statements of fact rather than advice.

    Parameters
    ----------
    text_plan:
        The attached text classifier, if any.
    topic_plan:
        The attached topic model, if any.
    fit_result:
        The most recent fit report.
    eval_result:
        The most recent holdout evaluation, surfaced in the disclosures.
    profile_result:
        The most recent corpus profile. Contamination findings are promoted
        into the disclosures, because a duplicate-ridden split invalidates
        everything downstream of it and should not be several clicks away.
    history:
        Session history records, scanned for NLP operations.

    Returns
    -------
    dict
        Whether an NLP plan is attached, whether NLP appears in history at all,
        which plans are present, and the disclosures.

    Notes
    -----
    **A session can show NLP activity with no plan attached**, and that is
    normal rather than an error. Keyphrases, sentiment, entities, summaries,
    language detection, and profiling hold no fitted state by design — they are
    analyses, not models. The disclosures say so, so an absent plan is not
    mistaken for lost work.

    **Everything here is factual.** Teaching and recommendations live in
    :mod:`buildml.explain`; this is what an audit trail should contain.

    See Also
    --------
    nlp_status_for_session : The same report, read off a Session.
    """
    records = list(history or [])
    saw = any(
        str(record.get("operation_id") or record.get("action")) in NLP_OPERATION_IDS
        for record in records
    )
    enabled = text_plan is not None or topic_plan is not None
    disclosures: list[str] = []

    if text_plan is not None:
        disclosures.extend(
            [
                f"NlpTextPlan task={getattr(text_plan, 'task', None)}, "
                f"backend={getattr(text_plan, 'backend', None)}, "
                f"estimator={getattr(text_plan, 'estimator', None)}, "
                f"text_column={getattr(text_plan, 'text_column', None)!r}, "
                f"n_features={getattr(text_plan, 'n_features', None)}, "
                f"classes={list(getattr(text_plan, 'classes_', ()) or ())}.",
                "The normalization plan, vocabulary, and head were fitted on "
                "Session train only. Validation/test are transform-and-score only.",
                "Session checkpoints do not embed the NLP vectorizer or head; use "
                "save_nlp_bundle / load_nlp_bundle.",
            ]
        )
        for note in getattr(text_plan, "disclosures", ()) or ():
            disclosures.append(str(note))
    if topic_plan is not None:
        disclosures.append(
            f"NlpTopicPlan method={getattr(topic_plan, 'method', None)}, "
            f"n_topics={getattr(topic_plan, 'n_topics', None)}, "
            f"text_column={getattr(topic_plan, 'text_column', None)!r} "
            "(fitted on train documents only)."
        )
    if not enabled and saw:
        disclosures.append(
            "NLP operations appear in history, but no live NLP plan is attached. "
            "Unsupervised surfaces (profile / keyphrases / sentiment / entities / "
            "summaries / language) hold no fitted state by design."
        )

    eval_payload = None
    if eval_result is not None:
        eval_payload = _payload(eval_result)
        disclosures.append(
            "Last NLP eval: "
            f"partition={eval_payload.get('partition')}, "
            f"metrics={eval_payload.get('metrics')}."
        )

    profile_payload = None
    if profile_result is not None:
        profile_payload = _payload(profile_result)
        overlap = int(profile_payload.get("train_holdout_exact_overlap") or 0)
        near = int(profile_payload.get("train_holdout_near_duplicate") or 0)
        if overlap or near:
            disclosures.append(
                f"Corpus profile flagged text contamination: {overlap} exact and "
                f"{near} near-duplicate holdout document(s) matched train."
            )
        else:
            disclosures.append(
                "Corpus profile found no exact or near-duplicate train/holdout "
                "document overlap at the configured threshold."
            )

    from buildml.explain.capability_status import attach_capability_matrix

    return attach_capability_matrix(
        {
        "enabled": enabled,
        "present": enabled or saw,
        "has_text_plan": text_plan is not None,
        "has_topic_plan": topic_plan is not None,
        "task": None if text_plan is None else getattr(text_plan, "task", None),
        "backend": None if text_plan is None else getattr(text_plan, "backend", None),
        "estimator": None if text_plan is None else getattr(text_plan, "estimator", None),
        "text_column": (
            getattr(text_plan, "text_column", None)
            if text_plan is not None
            else getattr(topic_plan, "text_column", None)
            if topic_plan is not None
            else None
        ),
        "n_features": None if text_plan is None else getattr(text_plan, "n_features", None),
        "classes": (
            None
            if text_plan is None
            else list(getattr(text_plan, "classes_", ()) or ())
        ),
        "n_topics": None if topic_plan is None else getattr(topic_plan, "n_topics", None),
        "has_fit_result": fit_result is not None,
        "has_eval_result": eval_result is not None,
        "has_profile_result": profile_result is not None,
        "eval": eval_payload,
        "profile": profile_payload,
        "disclosures": disclosures,
        "boundary": (
            "NLP models and analyses a text column that lives on the Session "
            "dataset: classify documents, interpret tokens, fit topics, extract "
            "keyphrases and entities, score sentiment, build extractive summaries, "
            "detect language, and profile corpus health. Distinct from RAG "
            "(retrieval for generation), from Session.text_features (tabular "
            "column expansion), from the Torch text path (fine-tuning), and from "
            "buildml.ai (external LLM providers)."
        ),
    },
        "nlp_capability_matrix",
    )


def nlp_status_for_session(session: Any) -> dict[str, Any]:
    """Report NLP status by reading the plans and history off a session.

    The convenience form of :func:`nlp_status` for callers that already hold a
    session: it gathers the attached plans, the most recent results, and the
    history, then delegates.

    Parameters
    ----------
    session:
        A :class:`~buildml.session.Session`. Attributes are read defensively,
        so a session with no NLP work done reports an empty status rather than
        failing.

    Returns
    -------
    dict
        The same structure :func:`nlp_status` returns.

    See Also
    --------
    nlp_status : The underlying report, and what each field means.
    """
    return nlp_status(
        getattr(session, "_nlp_text_plan", None),
        topic_plan=getattr(session, "_nlp_topic_plan", None),
        fit_result=getattr(session, "_nlp_fit_result", None),
        eval_result=getattr(session, "_nlp_eval_result", None),
        profile_result=getattr(session, "_nlp_profile_result", None),
        history=list(getattr(session, "_history", ()) or ()),
    )


__all__ = [
    "NLP_OPERATION_IDS",
    "entity_result_summary",
    "eval_result_summary",
    "fit_result_summary",
    "interpret_result_summary",
    "keyphrase_result_summary",
    "language_result_summary",
    "nlp_status",
    "nlp_status_for_session",
    "predict_result_summary",
    "profile_result_summary",
    "sentiment_result_summary",
    "summary_result_summary",
    "topic_assign_summary",
    "topic_result_summary",
]
