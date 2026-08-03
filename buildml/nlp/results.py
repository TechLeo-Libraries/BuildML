"""The objects every NLP operation hands back.

Each result carries three kinds of thing, and it helps to know which you are
reading.

The **findings** are what you asked for — the predictions, the metrics, the
topics. The **context** is what the operation saw while producing them, such as
how many rows, how much of the vocabulary was unseen, how many documents were
blank. And the **disclosures and warnings** are the caveats: what this number
does not mean, and what looked wrong on the way.

That last part is deliberate. A number without its caveats invites confident
misuse, and text results are unusually easy to misread — a strong accuracy on a
corpus where half the holdout duplicates the training set is not a strong model.
Rather than making you go and look, the caveat travels with the number.

Every result has a ``to_dict`` for logging and comparison. Fitted estimators and
vectorizers are summarised rather than serialised in those dictionaries; use
:mod:`buildml.nlp.checkpoint` to persist the model itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.nlp.normalize import TextNormalizePlan


@dataclass(slots=True)
class NlpTextPlan:
    """A fitted text classifier: normalisation, representation, and head together.

    Everything needed to turn a raw string into a predicted label, held as one
    object. Keeping the three stages together is what guarantees a document
    scored next year goes through the identical pipeline — mismatch any one of
    them and the head receives features that mean something different from what
    it was trained on, silently.

    Persist it with :func:`~buildml.nlp.checkpoint.save_nlp_bundle`.

    This is single-label document classification over a train-fitted
    representation. Not sequence labelling, not generation, and not document
    retrieval — :mod:`buildml.rag` owns that.

    Attributes
    ----------
    task:
        What the plan does.
    backend, estimator:
        Which representation family and which head were used.
    text_column, target_column:
        The columns fitted against, so scoring needs no reminder of either.
    normalize_plan:
        The resolved normalisation recipe, replayed on every new document.
    vectorize:
        The vectorizer settings, as plain data.
    n_train_rows, n_features:
        How many documents the fit saw and how wide the representation is.
        Features far exceeding rows is the classic overfitting shape for text.
    classes_:
        The fitted class labels, in the head's own order — which is also the
        column order of any probability output.
    vectorizer_, estimator_:
        The fitted objects themselves. Excluded from ``repr`` because printing
        a vocabulary of fifty thousand terms is not useful.
    embedding_model_name:
        Which pretrained encoder was used, for the neural backends.
    feature_names_:
        The vocabulary, stored so token attributions remain possible after
        reload. Empty for representations with no recoverable vocabulary.
    supports_proba:
        Whether the head can produce probabilities, so prediction can warn
        rather than fail when asked for them.
    disclosures, warnings:
        Caveats about the fit, and things that looked wrong during it.
    config:
        The hyperparameters, for the record.
    """

    task: str
    backend: str
    estimator: str
    text_column: str
    target_column: str
    normalize_plan: TextNormalizePlan
    vectorize: dict[str, Any]
    n_train_rows: int
    n_features: int
    classes_: tuple[Any, ...]
    vectorizer_: Any = field(repr=False, default=None)
    estimator_: Any = field(repr=False, default=None)
    embedding_model_name: str | None = None
    feature_names_: tuple[str, ...] = ()
    supports_proba: bool = False
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Describe the fitted plan as plain JSON-safe values.

        The fitted vectorizer and head are deliberately omitted — they are
        arbitrary Python objects, not data. The vocabulary is reduced to
        ``n_feature_names`` for the same reason a model card does not list
        fifty thousand terms. What remains is enough to identify and compare
        models; use :func:`~buildml.nlp.checkpoint.save_nlp_bundle` to persist
        one for scoring.

        Returns
        -------
        dict
            Task, backend, head, columns, the normalisation and vectorisation
            settings, shape, classes, disclosures, warnings, and configuration.
        """
        return {
            "kind": "nlp_text_classifier",
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "normalize": self.normalize_plan.to_dict(),
            "vectorize": dict(self.vectorize),
            "n_train_rows": self.n_train_rows,
            "n_features": self.n_features,
            "classes": list(self.classes_),
            "embedding_model_name": self.embedding_model_name,
            "n_feature_names": len(self.feature_names_),
            "supports_proba": self.supports_proba,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class NlpFitResult:
    """What the fit saw — the report that comes back alongside the model.

    Separate from the plan on purpose: the plan is the thing that scores
    documents, this is the thing you read to decide whether it should. It
    answers the questions that decide whether the model is worth evaluating at
    all — is the vocabulary a sane size, are the classes hopelessly imbalanced,
    are the documents long enough to model.

    Attributes
    ----------
    task, backend, estimator:
        What was fitted and how.
    text_column, target_column:
        Which columns were used.
    n_train_rows:
        How many documents the fit saw.
    n_features:
        The representation's width. Compare against ``n_train_rows``: far more
        features than documents is the standard shape for text and the standard
        reason to regularise.
    vocabulary_size:
        How many distinct terms were learned. Zero means the representation has
        no vocabulary at all — hashing or an embedding backend — not that
        nothing was learned.
    classes:
        The fitted class labels.
    class_counts:
        Documents per class. Read every metric that follows in light of this.
    train_score:
        In-sample accuracy. Near-perfect on text and close to meaningless; use
        :class:`NlpEvalResult` for a number you can trust.
    mean_document_tokens:
        Average surviving tokens per document. Very low means sparse features
        and unstable metrics.
    empty_document_rate:
        Share of training documents with no text, which can only be predicted
        from the class prior.
    disclosures, warnings:
        Caveats, and anything that looked wrong during the fit.
    """

    task: str
    backend: str
    estimator: str
    text_column: str
    target_column: str
    n_train_rows: int
    n_features: int
    vocabulary_size: int
    classes: tuple[Any, ...]
    class_counts: dict[str, int]
    train_score: float | None = None
    mean_document_tokens: float | None = None
    empty_document_rate: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the fit report as plain JSON-safe values.

        Everything here is already plain data, so this is a faithful record
        rather than a summary — suitable for logging every fit in an experiment
        and diffing them later.

        Returns
        -------
        dict
            Every field, with tuples and mappings copied to lists and dicts.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "estimator": self.estimator,
            "text_column": self.text_column,
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_features": self.n_features,
            "vocabulary_size": self.vocabulary_size,
            "classes": list(self.classes),
            "class_counts": dict(self.class_counts),
            "train_score": self.train_score,
            "mean_document_tokens": self.mean_document_tokens,
            "empty_document_rate": self.empty_document_rate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpPredictResult:
    """Predictions for a set of documents, with how much of them was legible.

    The predictions are the point, but the coverage fields are what tell you
    whether to believe them. A prediction made on a document whose words the
    model has never seen is drawn from the class prior, and it looks exactly
    like a prediction made on a document the model understood.

    Attributes
    ----------
    partition:
        Which rows were scored.
    task:
        What the plan does.
    n_rows:
        How many documents were scored.
    predictions:
        The predicted class label per document, in partition order.
    probabilities:
        Per-class probabilities aligned to ``classes``, or empty when the head
        has none. Poorly calibrated on high-dimensional text — read them as a
        ranking, not as frequencies.
    classes:
        The class labels, in the same order as the probability columns.
    oov_rate:
        Share of tokens outside the training vocabulary. ``None`` when the
        representation cannot be inspected, which is not the same as zero.
    empty_document_rate:
        Share of scored documents with no text.
    disclosures, warnings:
        Caveats, and anything that looked wrong while scoring.
    """

    partition: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    probabilities: tuple[tuple[float, ...], ...] = ()
    classes: tuple[Any, ...] = ()
    oov_rate: float | None = None
    empty_document_rate: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the scoring run as plain JSON-safe values.

        The predictions themselves are counted rather than listed, and
        probabilities reduced to a boolean. This is a run record, not a
        results export — reach for ``predictions`` directly when you want the
        labels.

        Returns
        -------
        dict
            Partition, row and prediction counts, whether probabilities were
            produced, the class list, the coverage rates, and the caveats.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "has_probabilities": bool(self.probabilities),
            "classes": list(self.classes),
            "oov_rate": self.oov_rate,
            "empty_document_rate": self.empty_document_rate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpEvalResult:
    """How the classifier performed on documents it was not trained on.

    Three views of the same predictions, deliberately, because each hides
    something the others show. The headline metrics are comparable across
    models; the per-class breakdown reveals a category the model has quietly
    given up on; the confusion matrix names which categories it mistakes for
    which.

    Attributes
    ----------
    partition:
        Which holdout was measured.
    task:
        What the plan does.
    n_rows:
        How many documents were evaluated. Small holdouts give metrics with
        wide error bars, whatever the decimal places suggest.
    metrics:
        Accuracy, balanced accuracy, macro and weighted F1, macro precision and
        recall — plus log loss and ROC AUC when probabilities allowed them.
    per_class:
        Precision, recall, F1, and support for each class.
    confusion:
        Row-major counts ordered by ``classes``: rows are true classes, columns
        predicted.
    classes:
        The class order for both ``per_class`` and ``confusion``. May include
        labels absent from training, which cannot be predicted correctly and
        are flagged in ``warnings``.
    oov_rate:
        Share of holdout tokens outside the training vocabulary. A high value
        undermines the whole measurement.
    disclosures, warnings:
        Caveats, and anything that looked wrong during evaluation.
    """

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: tuple[tuple[int, ...], ...] = ()
    classes: tuple[Any, ...] = ()
    oov_rate: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the full evaluation as plain JSON-safe values.

        Nothing is summarised away here — the confusion matrix and per-class
        table come through in full, because an evaluation record is only useful
        for later comparison if it is complete.

        Returns
        -------
        dict
            Partition, row count, metrics, per-class table, confusion matrix as
            nested lists, class order, out-of-vocabulary rate, and the caveats.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "per_class": {key: dict(value) for key, value in self.per_class.items()},
            "confusion": [list(row) for row in self.confusion],
            "classes": list(self.classes),
            "oov_rate": self.oov_rate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class TokenAttribution:
    """How much one token pushed one document toward or away from a class.

    The unit of a text explanation. Keeping the three numbers separate rather
    than reporting only the product is what makes an attribution diagnosable:
    a small contribution can come from a token the model barely cares about, or
    from a token it cares about greatly that barely appears in this document,
    and those call for different responses.

    Attributes
    ----------
    token:
        The term, exactly as it appears in the vocabulary — so already
        normalised, and stemmed if the plan stems.
    weight:
        The model's coefficient for this term and class. A property of the
        model, identical across every document.
    value:
        This term's value in this document — a TF-IDF weight, or a count.
        Zero for terms the document does not contain.
    contribution:
        ``weight * value``: the amount this token actually added to this
        document's decision score. Positive supports the target class,
        negative argues against it.
    """

    token: str
    weight: float
    value: float
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        """Return the attribution as plain JSON-safe values.

        All three numbers are kept rather than just the product, so a stored
        explanation stays as diagnosable as a live one.

        Returns
        -------
        dict
            The token, its model weight, its value in this document, and their
            product.
        """
        return {
            "token": self.token,
            "weight": self.weight,
            "value": self.value,
            "contribution": self.contribution,
        }


@dataclass(slots=True)
class NlpInterpretResult:
    """Why individual documents scored as they did, and what the model relies on.

    Two views held together because they answer different questions. The
    per-document attributions explain a particular decision — useful when a
    specific prediction looks wrong. The global view shows the vocabulary the
    model leans on across every document, which is where you find out it
    learned a proxy: a date, an agent's name, a template phrase that happens to
    correlate with one class.

    Attributes
    ----------
    partition:
        Which rows were explained.
    n_documents:
        How many documents are covered by the per-document view.
    target_class:
        The class the attributions are oriented toward. Positive contributions
        support this class.
    method:
        How the weights were derived — exact linear coefficients, or centred
        naive Bayes log-likelihoods. The two are not on the same scale, so this
        matters when comparing across models.
    document_attributions:
        Per document, the top tokens by absolute contribution, so the strongest
        evidence on both sides appears.
    document_predictions:
        What each explained document was predicted as.
    document_row_labels:
        Row identifiers, so an explanation can be traced back to its source
        record.
    global_top_tokens:
        Per class, the highest-weight vocabulary across the model as a whole,
        independent of any particular document.
    disclosures, warnings:
        Caveats, including the limits of the attribution method used.
    """

    partition: str
    n_documents: int
    target_class: Any
    method: str
    document_attributions: tuple[tuple[TokenAttribution, ...], ...] = ()
    document_predictions: tuple[Any, ...] = ()
    document_row_labels: tuple[Any, ...] = ()
    global_top_tokens: dict[str, tuple[TokenAttribution, ...]] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the explanation as plain JSON-safe values.

        Nested attributions are expanded in full, so the output can grow large
        — it scales with documents times tokens per document. Cap
        ``max_documents`` and ``top_k`` at call time if you intend to log this.

        Returns
        -------
        dict
            Partition, document count, target class, method, the nested
            per-document and global attributions, and the caveats.
        """
        return {
            "partition": self.partition,
            "n_documents": self.n_documents,
            "target_class": self.target_class,
            "method": self.method,
            "document_attributions": [
                [item.to_dict() for item in row] for row in self.document_attributions
            ],
            "document_predictions": list(self.document_predictions),
            "document_row_labels": [str(label) for label in self.document_row_labels],
            "global_top_tokens": {
                str(key): [item.to_dict() for item in value]
                for key, value in self.global_top_tokens.items()
            },
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class Topic:
    """One discovered topic — the terms that define it and how prevalent it is.

    A topic is not a category the model was told about; it is a cluster of
    words that tend to appear together, which the algorithm found on its own.
    Whether that cluster corresponds to anything meaningful is a judgement you
    make by reading the terms. Some topics are obviously "billing"; others are
    an artefact of shared boilerplate.

    Attributes
    ----------
    index:
        The topic's position in the model. Arbitrary, and not stable across
        refits.
    label:
        A readable name built from the top terms. Descriptive shorthand, not an
        interpretation.
    terms:
        The defining terms, most characteristic first. This is what you read to
        decide what the topic is.
    weights:
        How strongly each term belongs, aligned with ``terms``. A steep drop
        after the first few means a tightly defined topic; a flat profile means
        a diffuse one that may not be real.
    train_mass:
        The share of training documents this topic accounts for. A topic with
        very little mass was found in a handful of documents and is unlikely to
        generalise.
    coherence:
        How often the top terms genuinely co-occur, where it could be computed.
        Higher is better; a low value means the terms were grouped by the
        arithmetic rather than by actually appearing together.
    """

    index: int
    label: str
    terms: tuple[str, ...]
    weights: tuple[float, ...]
    train_mass: float
    coherence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the topic as plain JSON-safe values.

        Terms and weights stay aligned as parallel lists, since the weight
        profile is what distinguishes a sharply defined topic from a diffuse
        one.

        Returns
        -------
        dict
            Index, label, terms and their weights as lists, train mass, and
            coherence.
        """
        return {
            "index": self.index,
            "label": self.label,
            "terms": list(self.terms),
            "weights": list(self.weights),
            "train_mass": self.train_mass,
            "coherence": self.coherence,
        }


@dataclass(slots=True)
class NlpTopicPlan:
    """A fitted topic model, ready to assign topics to new documents.

    Holds the vectorizer and the decomposition together, for the same reason
    :class:`NlpTextPlan` does: a new document must be vectorised exactly as the
    training documents were, or the decomposition receives coordinates in a
    different space.

    Attributes
    ----------
    method:
        Which decomposition was used.
    text_column:
        The column fitted against.
    n_topics:
        How many topics were requested. Your choice, not a discovery — a
        different number produces a different and equally valid decomposition.
    n_train_rows:
        How many documents the fit saw.
    normalize_plan:
        The normalisation recipe, replayed on new documents.
    vectorize:
        The vectorizer settings, as plain data.
    topics:
        The discovered topics with their terms and mass.
    vectorizer_, model_:
        The fitted objects. Excluded from ``repr``.
    random_state:
        The seed. Topic models are randomised, and without a fixed seed the
        same corpus yields different topics each run.
    disclosures, warnings:
        Caveats about the fit.
    """

    method: str
    text_column: str
    n_topics: int
    n_train_rows: int
    normalize_plan: TextNormalizePlan
    vectorize: dict[str, Any]
    topics: tuple[Topic, ...]
    vectorizer_: Any = field(repr=False, default=None)
    model_: Any = field(repr=False, default=None)
    random_state: int | None = 0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Describe the fitted topic model as plain JSON-safe values.

        The fitted vectorizer and decomposition are omitted; the topics
        themselves come through in full, since the terms are the readable
        output of a topic model and belong in any record of it.

        Returns
        -------
        dict
            Method, column, topic count, training size, the normalisation and
            vectorisation settings, the topics, the seed, and the caveats.
        """
        return {
            "kind": "nlp_topics",
            "method": self.method,
            "text_column": self.text_column,
            "n_topics": self.n_topics,
            "n_train_rows": self.n_train_rows,
            "normalize": self.normalize_plan.to_dict(),
            "vectorize": dict(self.vectorize),
            "topics": [topic.to_dict() for topic in self.topics],
            "random_state": self.random_state,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpTopicResult:
    """What the topic fit found, and how well the topics hold together.

    The report you read to judge whether the decomposition is worth keeping.
    Since topic modelling is unsupervised there is no accuracy to check
    against, so the quality signals here are indirect — and reading the terms
    yourself remains the most reliable check of all.

    Attributes
    ----------
    method:
        Which decomposition was used.
    n_topics:
        How many topics were requested.
    n_train_rows:
        How many documents the fit saw.
    text_column:
        The column fitted against.
    topics:
        The discovered topics with their defining terms.
    mean_coherence:
        Average coherence across topics, where computable. The single best
        proxy for "are these topics real", and the usual way to choose
        ``n_topics`` — fit several and take the peak.
    reconstruction_error:
        How much of the original matrix the decomposition failed to
        reproduce, for the matrix-factorisation methods. Always falls as topics
        are added, so it cannot be used to choose a topic count.
    perplexity:
        How surprised a probabilistic model is by the corpus, for LDA. Lower is
        better, and it notoriously disagrees with human judgements of topic
        quality — treat coherence as the better guide.
    disclosures, warnings:
        Caveats about the fit.
    """

    method: str
    n_topics: int
    n_train_rows: int
    text_column: str
    topics: tuple[Topic, ...]
    mean_coherence: float | None = None
    reconstruction_error: float | None = None
    perplexity: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the topic fit report as plain JSON-safe values.

        Logging this across several topic counts is how you find the coherence
        peak, so the quality signals come through unsummarised.

        Returns
        -------
        dict
            Method, topic count, training size, column, the topics, the three
            quality signals, and the caveats.
        """
        return {
            "method": self.method,
            "n_topics": self.n_topics,
            "n_train_rows": self.n_train_rows,
            "text_column": self.text_column,
            "topics": [topic.to_dict() for topic in self.topics],
            "mean_coherence": self.mean_coherence,
            "reconstruction_error": self.reconstruction_error,
            "perplexity": self.perplexity,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpTopicAssignResult:
    """Which topics each document belongs to, and in what proportion.

    Topic assignment is not classification. A document does not get one topic;
    it gets a weight over all of them, because a support ticket really can be
    60% billing and 40% technical. The dominant topic is a convenience for
    when you need a single answer, and it discards that nuance — a document
    split 34/33/33 gets a dominant topic that means almost nothing.

    Attributes
    ----------
    partition:
        Which rows were assigned.
    method:
        Which decomposition produced the weights.
    n_rows:
        How many documents were assigned.
    n_topics:
        How many topics the model has.
    dominant_topics:
        The highest-weighted topic index per document. Check the weights before
        relying on it.
    topic_weights:
        The full distribution per document, one row each, aligned to topic
        index.
    topic_share:
        What fraction of this partition each topic dominates. Compare against
        the plan's ``train_mass``: a large shift means the documents you are
        assigning are about different things than the ones the model was fitted
        on.
    topic_labels:
        Readable topic names, aligned to topic index.
    disclosures, warnings:
        Caveats about the assignment.
    """

    partition: str
    method: str
    n_rows: int
    n_topics: int
    dominant_topics: tuple[int, ...]
    topic_weights: tuple[tuple[float, ...], ...]
    topic_share: dict[str, float] = field(default_factory=dict)
    topic_labels: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the assignment as plain JSON-safe values.

        The full per-document weight matrix is omitted — it is
        ``n_rows × n_topics`` floats and belongs in an array, not a log entry.
        Read ``topic_weights`` directly when you need it.

        Returns
        -------
        dict
            Partition, method, counts, dominant topic per document, the topic
            share, the labels, and the caveats.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "n_topics": self.n_topics,
            "dominant_topics": list(self.dominant_topics),
            "topic_share": dict(self.topic_share),
            "topic_labels": list(self.topic_labels),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class Keyphrase:
    """A phrase judged salient, with the score and the evidence behind it.

    Attributes
    ----------
    phrase:
        The phrase, as it appears after normalisation.
    score:
        How salient it is by the chosen method. Scores are comparable within
        one result and not across methods or corpora, so use them to rank
        rather than to threshold.
    document_frequency:
        How many documents contain it. The context the score needs: a high
        score with a frequency of one is a phrase from a single document, which
        may be a genuine specific finding or may be noise.
    """

    phrase: str
    score: float
    document_frequency: int

    def to_dict(self) -> dict[str, Any]:
        """Return the keyphrase as plain JSON-safe values.

        The document frequency travels with the score, because a score without
        it cannot be told apart from noise.

        Returns
        -------
        dict
            The phrase, its score, and its document frequency.
        """
        return {
            "phrase": self.phrase,
            "score": self.score,
            "document_frequency": self.document_frequency,
        }


@dataclass(slots=True)
class NlpKeyphraseResult:
    """The phrases that characterise a corpus, and each document within it.

    Both scopes are reported because they answer different questions. Corpus
    keyphrases tell you what the collection is about — a fast way to understand
    a dataset you have just been handed. Per-document keyphrases tell you what
    distinguishes each document from the rest, which is what you want for
    tagging or for a searchable index.

    Attributes
    ----------
    partition:
        Which rows were analysed.
    method:
        Which extraction method was used.
    n_rows:
        How many documents were analysed.
    top_n:
        How many phrases were kept per scope.
    corpus_keyphrases:
        The phrases characterising the collection as a whole.
    document_keyphrases:
        Per document, its own top phrases.
    document_row_labels:
        Row identifiers, so per-document phrases trace back to their source.
    disclosures, warnings:
        Caveats about the extraction.
    """

    partition: str
    method: str
    n_rows: int
    top_n: int
    corpus_keyphrases: tuple[Keyphrase, ...]
    document_keyphrases: tuple[tuple[Keyphrase, ...], ...] = ()
    document_row_labels: tuple[Any, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the keyphrases as plain JSON-safe values.

        Both scopes are expanded in full, so the output scales with documents
        times ``top_n``.

        Returns
        -------
        dict
            Partition, method, counts, corpus and per-document phrases, row
            labels, and the caveats.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "top_n": self.top_n,
            "corpus_keyphrases": [item.to_dict() for item in self.corpus_keyphrases],
            "document_keyphrases": [
                [item.to_dict() for item in row] for row in self.document_keyphrases
            ],
            "document_row_labels": [str(label) for label in self.document_row_labels],
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpSentimentResult:
    """Sentiment across a partition, with the coverage the verdict rests on.

    The rates are what most people want, but ``matched_term_rate`` is what
    tells you whether to believe them. A lexicon that recognised almost none of
    your vocabulary reports a neutral corpus with the same confidence as one
    that read every word.

    Attributes
    ----------
    partition:
        Which rows were scored.
    backend:
        Which method produced the scores. Determines what the numbers mean and
        what they can be compared against.
    n_rows:
        How many documents were scored.
    labels:
        Per document: positive, negative, or neutral.
    scores:
        Per document compound score from −1 to 1. Bounded but not linear, so
        rank by them rather than differencing them.
    positive_rate, negative_rate, neutral_rate:
        The distribution across the partition.
    mean_score:
        Average compound score. Near zero can mean a balanced corpus or a
        polarised one — check the rates before concluding either.
    matched_term_rate:
        Share of tokens the lexicon recognised. The evidence base for the whole
        result; ``None`` for the backends where it does not apply.
    agreement:
        How the predicted sentiment compares against the dataset's target, when
        ``compare_to_target`` was set. The only real validation available for
        an unsupervised backend.
    disclosures, warnings:
        Caveats — including, for the transformer backend, that its training
        data lies outside your split entirely.
    """

    partition: str
    backend: str
    n_rows: int
    labels: tuple[str, ...]
    scores: tuple[float, ...]
    positive_rate: float
    negative_rate: float
    neutral_rate: float
    mean_score: float
    matched_term_rate: float | None = None
    agreement: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the sentiment run as plain JSON-safe values.

        Per-document labels and scores are omitted — they are one entry per row
        and belong in a column, not a log record. Read ``labels`` and
        ``scores`` directly for those.

        Returns
        -------
        dict
            Partition, backend, row count, the three rates, mean score, match
            rate, target agreement, and the caveats.
        """
        return {
            "partition": self.partition,
            "backend": self.backend,
            "n_rows": self.n_rows,
            "positive_rate": self.positive_rate,
            "negative_rate": self.negative_rate,
            "neutral_rate": self.neutral_rate,
            "mean_score": self.mean_score,
            "matched_term_rate": self.matched_term_rate,
            "agreement": dict(self.agreement),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class Entity:
    """One named thing found in a document, and where it was found.

    Attributes
    ----------
    text:
        The mention exactly as it appears in the source, uncleaned — so it can
        be located, highlighted, or redacted in the original.
    label:
        What kind of thing it is: a person, an organisation, a date, a
        monetary amount, an identifier.
    start, end:
        Character offsets into the raw document, which is what makes
        highlighting and redaction possible rather than just detection.
    source:
        Which extractor found it. Rule-based patterns and statistical models
        have very different failure modes, so knowing the origin is part of
        knowing how much to trust the mention.
    """

    text: str
    label: str
    start: int
    end: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Return the entity mention as plain JSON-safe values.

        The character span is preserved, which is what lets a stored mention
        still be highlighted or redacted in the original document.

        Returns
        -------
        dict
            The mention text, its label, its character span, and its source.
        """
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


@dataclass(slots=True)
class NlpEntityResult:
    """Every entity mention found, plus what the corpus contains overall.

    Per-document mentions are what you act on — redacting personal data,
    linking records, pulling structured fields out of prose. The corpus
    aggregates are what you check first: they show at a glance whether the
    extractor is finding what you expected, or whether one label is wildly
    over-firing.

    Attributes
    ----------
    partition:
        Which rows were scanned.
    backend:
        Which extractor was used.
    n_rows:
        How many documents were scanned.
    n_entities:
        Total mentions found. Divide by ``n_rows`` for a per-document rate;
        a very high one usually means a pattern is matching too eagerly.
    label_counts:
        Mentions per label across the corpus.
    document_entities:
        The mentions found in each document, with their spans.
    document_row_labels:
        Row identifiers, so mentions trace back to their source record.
    top_mentions:
        Per label, the most frequent surface forms. The quickest way to spot a
        misfiring pattern — an ``ORG`` list full of ordinary sentence openers
        tells you immediately.
    disclosures, warnings:
        Caveats, including what the backend cannot detect.
    """

    partition: str
    backend: str
    n_rows: int
    n_entities: int
    label_counts: dict[str, int]
    document_entities: tuple[tuple[Entity, ...], ...] = ()
    document_row_labels: tuple[Any, ...] = ()
    top_mentions: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the entity findings as plain JSON-safe values.

        Per-document mentions are expanded in full, including spans, since
        those offsets are the whole point for redaction and highlighting
        workflows.

        Returns
        -------
        dict
            Partition, backend, counts, label counts, per-document mentions,
            row labels, top mentions per label, and the caveats.
        """
        return {
            "partition": self.partition,
            "backend": self.backend,
            "n_rows": self.n_rows,
            "n_entities": self.n_entities,
            "label_counts": dict(self.label_counts),
            "document_entities": [
                [item.to_dict() for item in row] for row in self.document_entities
            ],
            "document_row_labels": [str(label) for label in self.document_row_labels],
            "top_mentions": {
                key: [list(item) for item in value]
                for key, value in self.top_mentions.items()
            },
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpSummaryResult:
    """Shortened documents, built from sentences the originals already contained.

    These summaries are extractive: sentences are scored and the best ones
    kept. Nothing is rewritten and no sentence is invented, which means the
    summary cannot say anything the document did not — a real guarantee, and
    the reason generative summarisation is out of scope here.

    The cost is that the result reads like excerpts rather than prose, and a
    document whose meaning is spread across many sentences summarises badly.

    Attributes
    ----------
    partition:
        Which rows were summarised.
    method:
        Which sentence-scoring method was used.
    n_rows:
        How many documents were summarised.
    n_sentences:
        How many sentences each summary was allowed.
    summaries:
        The summary text per document, sentences joined in their original
        order.
    selected_sentence_indices:
        Which sentences were chosen, per document. Lets a summary be traced
        back to its exact source sentences, or highlighted in place of being
        shown separately.
    document_row_labels:
        Row identifiers for the summarised documents.
    mean_compression:
        Average ratio of summary length to original. Very low means aggressive
        compression and a higher chance of dropping something important; near
        1.0 means the documents were already short enough that summarising
        achieved nothing.
    disclosures, warnings:
        Caveats, including documents too short to summarise meaningfully.
    """

    partition: str
    method: str
    n_rows: int
    n_sentences: int
    summaries: tuple[str, ...]
    selected_sentence_indices: tuple[tuple[int, ...], ...] = ()
    document_row_labels: tuple[Any, ...] = ()
    mean_compression: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the run as plain JSON-safe values.

        The summary text is counted rather than included — it is the payload,
        not metadata about the run. The selected sentence indices are kept,
        since they are what makes a summary auditable against its source.

        Returns
        -------
        dict
            Partition, method, counts, selected sentence indices, row labels,
            mean compression, and the caveats.
        """
        return {
            "partition": self.partition,
            "method": self.method,
            "n_rows": self.n_rows,
            "n_sentences": self.n_sentences,
            "n_summaries": len(self.summaries),
            "selected_sentence_indices": [list(row) for row in self.selected_sentence_indices],
            "document_row_labels": [str(label) for label in self.document_row_labels],
            "mean_compression": self.mean_compression,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpLanguageResult:
    """Which language each document is written in, and what the corpus mixes.

    Worth checking before anything else. A corpus quietly containing three
    languages produces a vocabulary that is mostly one of them plus noise,
    stopword lists that only work on part of it, and a model whose accuracy
    varies by language for reasons nothing in the metrics will reveal.

    Attributes
    ----------
    partition:
        Which rows were checked.
    backend:
        Which detector was used.
    n_rows:
        How many documents were checked.
    languages:
        The detected language code per document.
    confidences:
        How sure the detector was, per document. Short documents get low
        confidence for good reason — five words are often not enough to tell
        two related languages apart.
    language_counts:
        Documents per detected language.
    dominant_language:
        The most common one, or ``None`` when nothing could be determined.
    undetermined_rate:
        Share of documents the detector could not classify. Usually blank or
        very short text rather than an exotic language.
    disclosures, warnings:
        Caveats, including a warning when the corpus is genuinely multilingual.
    """

    partition: str
    backend: str
    n_rows: int
    languages: tuple[str, ...]
    confidences: tuple[float, ...]
    language_counts: dict[str, int]
    dominant_language: str | None
    undetermined_rate: float = 0.0
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise language detection as plain JSON-safe values.

        Per-document languages and confidences are omitted as one-per-row data;
        read ``languages`` and ``confidences`` directly for those.

        Returns
        -------
        dict
            Partition, backend, row count, language counts, the dominant
            language, the undetermined rate, and the caveats.
        """
        return {
            "partition": self.partition,
            "backend": self.backend,
            "n_rows": self.n_rows,
            "language_counts": dict(self.language_counts),
            "dominant_language": self.dominant_language,
            "undetermined_rate": self.undetermined_rate,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class NlpCorpusProfile:
    """Everything worth knowing about a text column before you model it.

    Read the contamination fields first. Duplicates across the split are the
    one problem here that invalidates results rather than merely degrading
    them, and they are invisible to every tabular leakage check — no column is
    shared, no statistic crosses the boundary, and the holdout score is
    nonetheless measuring memorisation.

    Attributes
    ----------
    text_column:
        Which column was profiled.
    partitions:
        Document counts per partition.
    n_documents, n_empty, empty_rate:
        Corpus size and how much of it has no text at all.
    document_length_chars, document_length_tokens:
        Mean, median, 95th percentile, and maximum lengths. A mean far above
        the median means a few very long documents dominate.
    vocabulary_size:
        Distinct terms across the corpus.
    hapax_rate:
        Share of terms appearing exactly once. High values mean typos, names,
        and identifiers — terms that cannot generalise and are worth removing
        with ``min_df``.
    type_token_ratio:
        Distinct terms divided by total tokens. Low means repetitive text,
        often boilerplate; high means diverse vocabulary, which needs more data
        to model.
    top_tokens:
        The most frequent terms with their counts. Scanning these finds
        template headers and signatures that should be stripped.
    duplicate_document_groups, duplicate_document_rate:
        Exact duplicates found anywhere in the corpus.
    train_holdout_exact_overlap:
        Holdout documents identical to a training document. Any non-zero value
        means your holdout score is partly measuring memorisation.
    train_holdout_near_duplicate:
        Holdout documents highly similar to a training one — the same problem,
        harder to see, and usually more common.
    near_duplicate_threshold:
        The similarity cut-off used, so the count can be interpreted.
    holdout_oov_token_rate:
        Share of holdout tokens absent from training text. ``None`` when it
        could not be computed.
    language_counts:
        Detected languages across the corpus, when detection was enabled.
    findings:
        Plain-language statements of what the profile discovered.
    disclosures, warnings:
        Caveats, and the problems that need a decision before you model.
    """

    text_column: str
    partitions: dict[str, int]
    n_documents: int
    n_empty: int
    empty_rate: float
    document_length_chars: dict[str, float]
    document_length_tokens: dict[str, float]
    vocabulary_size: int
    hapax_rate: float
    type_token_ratio: float
    top_tokens: tuple[tuple[str, int], ...]
    duplicate_document_groups: int
    duplicate_document_rate: float
    train_holdout_exact_overlap: int
    train_holdout_near_duplicate: int
    near_duplicate_threshold: float
    holdout_oov_token_rate: float | None
    language_counts: dict[str, int] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the full corpus profile as plain JSON-safe values.

        Nothing is summarised away. This is the record to attach to a dataset
        and keep: profiling the same column again months later and diffing the
        two is how corpus drift becomes visible.

        Returns
        -------
        dict
            Every field, with tuples and mappings copied to lists and dicts.
        """
        return {
            "text_column": self.text_column,
            "partitions": dict(self.partitions),
            "n_documents": self.n_documents,
            "n_empty": self.n_empty,
            "empty_rate": self.empty_rate,
            "document_length_chars": dict(self.document_length_chars),
            "document_length_tokens": dict(self.document_length_tokens),
            "vocabulary_size": self.vocabulary_size,
            "hapax_rate": self.hapax_rate,
            "type_token_ratio": self.type_token_ratio,
            "top_tokens": [list(item) for item in self.top_tokens],
            "duplicate_document_groups": self.duplicate_document_groups,
            "duplicate_document_rate": self.duplicate_document_rate,
            "train_holdout_exact_overlap": self.train_holdout_exact_overlap,
            "train_holdout_near_duplicate": self.train_holdout_near_duplicate,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "holdout_oov_token_rate": self.holdout_oov_token_rate,
            "language_counts": dict(self.language_counts),
            "findings": list(self.findings),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


__all__ = [
    "Entity",
    "Keyphrase",
    "NlpCorpusProfile",
    "NlpEntityResult",
    "NlpEvalResult",
    "NlpFitResult",
    "NlpInterpretResult",
    "NlpKeyphraseResult",
    "NlpLanguageResult",
    "NlpPredictResult",
    "NlpSentimentResult",
    "NlpSummaryResult",
    "NlpTextPlan",
    "NlpTopicAssignResult",
    "NlpTopicPlan",
    "NlpTopicResult",
    "Topic",
    "TokenAttribution",
]
