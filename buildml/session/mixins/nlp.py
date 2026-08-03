"""Session mixin: nlp domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import nlp_ops
from buildml.session.mixins._shared import *  # noqa: F403


class NlpSessionMixin:
    """Public Session methods for the nlp domain."""
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _nlp_entity_result: Any
        _nlp_eval_result: Any
        _nlp_fit_result: Any
        _nlp_interpret_result: Any
        _nlp_keyphrase_result: Any
        _nlp_language_result: Any
        _nlp_predict_result: Any
        _nlp_profile_result: Any
        _nlp_sentiment_result: Any
        _nlp_summary_result: Any
        _nlp_text_plan: Any
        _nlp_topic_assign_result: Any
        _nlp_topic_plan: Any
        _nlp_topic_result: Any

    @staticmethod
    def nlp_capability_matrix() -> dict[str, Any]:
        """Honest capability matrix for NLP backends and task surfaces.

        Session facade over :func:`buildml.session.nlp_ops.nlp_capability_matrix_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        dict[str, Any]
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.nlp_capability_matrix_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("dict[str, Any]", nlp_ops.nlp_capability_matrix_op())

    def profile_text_corpus(
        self,
        *,
        text_column: str | None = None,
        top_tokens: int = 25,
        near_duplicate_threshold: float = 0.9,
        detect_languages: bool = True,
        stopword_language: str | None = None,
    ) -> NlpCorpusProfile:
        """Profile corpus health and screen the split for text contamination.

        Session facade over :func:`buildml.session.nlp_ops.profile_text_corpus_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpCorpusProfile
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.profile_text_corpus_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpCorpusProfile", nlp_ops.profile_text_corpus_op(
            self,
            text_column=text_column,
            top_tokens=top_tokens,
            near_duplicate_threshold=near_duplicate_threshold,
            detect_languages=detect_languages,
            stopword_language=stopword_language,
        ))

    def detect_language(
        self,
        *,
        partition: PartitionName | Literal["all"] = "all",
        backend: str | None = "native",
        text_column: str | None = None,
        min_characters: int = 12,
    ) -> NlpLanguageResult:
        """Identify the language of every document in a partition.

        Session facade over :func:`buildml.session.nlp_ops.detect_language_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpLanguageResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.detect_language_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpLanguageResult", nlp_ops.detect_language_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            min_characters=min_characters,
        ))

    def fit_text_classifier(
        self,
        *,
        backend: str | None = None,
        estimator: str | None = None,
        text_column: str | None = None,
        vectorizer: str = "tfidf",
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 20000,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = True,
        binary: bool = False,
        n_hash_features: int = 2**18,
        normalize_steps: list[str] | None = None,
        stopwords: list[str] | None = None,
        stopword_language: str | None = None,
        min_token_length: int = 1,
        max_token_length: int = 40,
        stem: bool = False,
        lemmatize: bool = False,
        class_weight: str | None = None,
        C: float = 1.0,
        alpha: float = 1.0,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_tokens: int = 256,
        device: str = "cpu",
        random_state: int | None = 0,
    ) -> NlpFitResult:
        """Fit a single-label document classifier on Session train.

        Session facade over :func:`buildml.session.nlp_ops.fit_text_classifier_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpFitResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.fit_text_classifier_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpFitResult", nlp_ops.fit_text_classifier_op(
            self,
            backend=backend,
            estimator=estimator,
            text_column=text_column,
            vectorizer=vectorizer,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            binary=binary,
            n_hash_features=n_hash_features,
            normalize_steps=normalize_steps,
            stopwords=stopwords,
            stopword_language=stopword_language,
            min_token_length=min_token_length,
            max_token_length=max_token_length,
            stem=stem,
            lemmatize=lemmatize,
            class_weight=class_weight,
            C=C,
            alpha=alpha,
            embedding_model_name=embedding_model_name,
            max_seq_tokens=max_seq_tokens,
            device=device,
            random_state=random_state,
        ))

    def predict_text(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        return_probabilities: bool = True,
    ) -> NlpPredictResult:
        """Score a partition with the train-fitted text plan.

        Session facade over :func:`buildml.session.nlp_ops.predict_text_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpPredictResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.predict_text_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpPredictResult", nlp_ops.predict_text_op(
            self,
            partition=partition,
            return_probabilities=return_probabilities,
        ))

    def evaluate_text_classifier(
        self,
        *,
        partition: PartitionName | Literal["all"] = "validation",
    ) -> NlpEvalResult:
        """Evaluate the text classifier on a holdout partition.

        Session facade over :func:`buildml.session.nlp_ops.evaluate_text_classifier_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpEvalResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.evaluate_text_classifier_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpEvalResult", nlp_ops.evaluate_text_classifier_op(self, partition=partition))

    def interpret_text_prediction(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        target_class: Any = None,
        top_k: int = 12,
        max_documents: int = 10,
        include_global: bool = True,
    ) -> NlpInterpretResult:
        """Explain document decisions with per-token contributions.

        Session facade over :func:`buildml.session.nlp_ops.interpret_text_prediction_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpInterpretResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.interpret_text_prediction_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpInterpretResult", nlp_ops.interpret_text_prediction_op(
            self,
            partition=partition,
            target_class=target_class,
            top_k=top_k,
            max_documents=max_documents,
            include_global=include_global,
        ))

    def fit_topics(
        self,
        *,
        method: str = "nmf",
        n_topics: int = 6,
        text_column: str | None = None,
        top_terms: int = 10,
        max_features: int | None = 20000,
        min_df: int | float = 2,
        max_df: int | float = 0.95,
        ngram_range: tuple[int, int] = (1, 1),
        normalize_steps: list[str] | None = None,
        stopwords: list[str] | None = None,
        stopword_language: str | None = "en",
        stem: bool = False,
        max_iter: int = 300,
        random_state: int | None = 0,
    ) -> NlpTopicResult:
        """Fit an unsupervised topic model on Session train documents.

        Session facade over :func:`buildml.session.nlp_ops.fit_topics_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpTopicResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.fit_topics_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpTopicResult", nlp_ops.fit_topics_op(
            self,
            method=method,
            n_topics=n_topics,
            text_column=text_column,
            top_terms=top_terms,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            normalize_steps=normalize_steps,
            stopwords=stopwords,
            stopword_language=stopword_language,
            stem=stem,
            max_iter=max_iter,
            random_state=random_state,
        ))

    def assign_topics(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
    ) -> NlpTopicAssignResult:
        """Transform a partition into per-document topic weights.

        Session facade over :func:`buildml.session.nlp_ops.assign_topics_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpTopicAssignResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.assign_topics_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpTopicAssignResult", nlp_ops.assign_topics_op(self, partition=partition))

    def extract_keyphrases(
        self,
        *,
        partition: PartitionName | Literal["all"] = "train",
        method: str = "tfidf",
        text_column: str | None = None,
        top_n: int = 15,
        max_phrase_words: int = 3,
        per_document: bool = True,
        max_documents: int = 25,
        stopword_language: str | None = "en",
        stopwords: list[str] | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        window: int = 4,
        random_state: int | None = 0,
    ) -> NlpKeyphraseResult:
        """Rank keyphrases for a partition with an unsupervised scorer.

        Session facade over :func:`buildml.session.nlp_ops.extract_keyphrases_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpKeyphraseResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.extract_keyphrases_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpKeyphraseResult", nlp_ops.extract_keyphrases_op(
            self,
            partition=partition,
            method=method,
            text_column=text_column,
            top_n=top_n,
            max_phrase_words=max_phrase_words,
            per_document=per_document,
            max_documents=max_documents,
            stopword_language=stopword_language,
            stopwords=stopwords,
            min_df=min_df,
            max_df=max_df,
            window=window,
            random_state=random_state,
        ))

    def analyze_sentiment(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: str = "lexicon",
        text_column: str | None = None,
        threshold: float = 0.05,
        compare_to_target: bool = False,
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
    ) -> NlpSentimentResult:
        """Score a partition's documents for sentiment.

        Session facade over :func:`buildml.session.nlp_ops.analyze_sentiment_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpSentimentResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.analyze_sentiment_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpSentimentResult", nlp_ops.analyze_sentiment_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            threshold=threshold,
            compare_to_target=compare_to_target,
            transformer_model=transformer_model,
            device=device,
        ))

    def extract_entities(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        backend: str | None = "rules",
        text_column: str | None = None,
        labels: list[str] | None = None,
        gazetteers: dict[str, list[str]] | None = None,
        spacy_model: str = "en_core_web_sm",
        max_documents: int = 25,
        batch_size: int = 32,
    ) -> NlpEntityResult:
        """Extract entity mentions from a partition's documents.

        Session facade over :func:`buildml.session.nlp_ops.extract_entities_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpEntityResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.extract_entities_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpEntityResult", nlp_ops.extract_entities_op(
            self,
            partition=partition,
            backend=backend,
            text_column=text_column,
            labels=labels,
            gazetteers=gazetteers,
            spacy_model=spacy_model,
            max_documents=max_documents,
            batch_size=batch_size,
        ))

    def summarize_text(
        self,
        *,
        partition: PartitionName | Literal["all"] = "test",
        method: str = "textrank",
        text_column: str | None = None,
        n_sentences: int = 3,
        max_documents: int = 25,
        max_input_sentences: int = 200,
        stopword_language: str | None = "en",
        stopwords: list[str] | None = None,
    ) -> NlpSummaryResult:
        """Build extractive summaries for a partition's documents.

        Session facade over :func:`buildml.session.nlp_ops.summarize_text_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        NlpSummaryResult
            Structured domain result recorded on the Session for follow-up evaluate/explain/export ...

        See Also
        --------
        :func:`buildml.session.nlp_ops.summarize_text_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("NlpSummaryResult", nlp_ops.summarize_text_op(
            self,
            partition=partition,
            method=method,
            text_column=text_column,
            n_sentences=n_sentences,
            max_documents=max_documents,
            max_input_sentences=max_input_sentences,
            stopword_language=stopword_language,
            stopwords=stopwords,
        ))

    @property
    def nlp_text_plan(self) -> NlpTextPlan | None:
        """Return the text plan built by the most recent classifier fit.

        Session-held result for ``nlp_text_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpTextPlan | None", self._nlp_text_plan)

    @property
    def nlp_topic_plan(self) -> NlpTopicPlan | None:
        """Return the topic model built by the most recent topic fit.

        Session-held result for ``nlp_topic_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpTopicPlan | None", self._nlp_topic_plan)

    @property
    def nlp_fit_result(self) -> NlpFitResult | None:
        """Return the report from the most recent classifier fit.

        Session-held result for ``nlp_fit_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpFitResult | None", self._nlp_fit_result)

    @property
    def nlp_eval_result(self) -> NlpEvalResult | None:
        """Return the metrics from the most recent classifier evaluation.

        Session-held result for ``nlp_eval_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpEvalResult | None", self._nlp_eval_result)

    @property
    def nlp_predict_result(self) -> NlpPredictResult | None:
        """Return the predictions from the most recent scoring call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpPredictResult` or None
            ``None`` until ``predict_text`` has run.
        """
        return cast("NlpPredictResult | None", self._nlp_predict_result)

    @property
    def nlp_interpret_result(self) -> NlpInterpretResult | None:
        """Return the token attributions from the most recent interpretation.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpInterpretResult` or None
            ``None`` until ``interpret_text_prediction`` has run — which it
            cannot for hashing or dense representations.
        """
        return cast("NlpInterpretResult | None", self._nlp_interpret_result)

    @property
    def nlp_topic_result(self) -> NlpTopicResult | None:
        """Return the report from the most recent topic fit.

        Session-held result for ``nlp_topic_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpTopicResult | None", self._nlp_topic_result)

    @property
    def nlp_topic_assign_result(self) -> NlpTopicAssignResult | None:
        """Return the topic assignment from the most recent assignment call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpTopicAssignResult` or None
            ``None`` until ``assign_topics`` has run.
        """
        return cast("NlpTopicAssignResult | None", self._nlp_topic_assign_result)

    @property
    def nlp_keyphrase_result(self) -> NlpKeyphraseResult | None:
        """Return the phrases from the most recent keyphrase extraction.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpKeyphraseResult` or None
            ``None`` until ``extract_keyphrases`` has run.
        """
        return cast("NlpKeyphraseResult | None", self._nlp_keyphrase_result)

    @property
    def nlp_sentiment_result(self) -> NlpSentimentResult | None:
        """Return the scores from the most recent sentiment analysis.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpSentimentResult` or None
            ``None`` until ``analyze_sentiment`` has run. Check
            ``matched_term_rate`` before quoting any rate from it.
        """
        return cast("NlpSentimentResult | None", self._nlp_sentiment_result)

    @property
    def nlp_entity_result(self) -> NlpEntityResult | None:
        """Return the mentions from the most recent entity extraction.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpEntityResult` or None
            ``None`` until ``extract_entities`` has run.
        """
        return cast("NlpEntityResult | None", self._nlp_entity_result)

    @property
    def nlp_summary_result(self) -> NlpSummaryResult | None:
        """Return the summaries from the most recent summarisation call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpSummaryResult` or None
            ``None`` until ``summarize_text`` has run.
        """
        return cast("NlpSummaryResult | None", self._nlp_summary_result)

    @property
    def nlp_language_result(self) -> NlpLanguageResult | None:
        """Return the languages from the most recent detection call.

        Returns
        -------
        :class:`~buildml.nlp.results.NlpLanguageResult` or None
            ``None`` until ``detect_language`` has run. ``profile_text_corpus``
            reports a language mix too, but does not populate this accessor.
        """
        return cast("NlpLanguageResult | None", self._nlp_language_result)

    @property
    def nlp_profile_result(self) -> NlpCorpusProfile | None:
        """Return the report from the most recent corpus profile.

        Session-held result for ``nlp_profile_result``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("NlpCorpusProfile | None", self._nlp_profile_result)

    def save_nlp_bundle(self, path: str | Path) -> Path:
        """Persist the active NLP plan(s) as ``buildml.nlp_bundle.v1``.

        Session facade over :func:`buildml.session.nlp_ops.save_nlp_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        pathlib.Path
            The directory written.

        See Also
        --------
        :func:`buildml.session.nlp_ops.save_nlp_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Path", nlp_ops.save_nlp_bundle_op(self, path=path))

    def load_nlp_bundle(self, path: str | Path, *, trusted: bool = False) -> Session:
        """Restore a saved text plan, and topic plan, into this Session.

        Session facade over :func:`buildml.session.nlp_ops.load_nlp_bundle_op`. Canonical Parameters, Raises, Notes, and Examples live on that ops function — keep this method as a thin delegate.

        Returns
        -------
        Session
            This Session, so the call chains.

        See Also
        --------
        :func:`buildml.session.nlp_ops.load_nlp_bundle_op`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", nlp_ops.load_nlp_bundle_op(self, path=path, trusted=trusted))
