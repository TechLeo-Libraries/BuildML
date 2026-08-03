"""Judge whether a document reads as positive, negative, or neutral.

Three routes, and the right one depends on whether you have labels.

The **lexicon** route needs nothing. Each word carries a valence score from a
shipped dictionary, and the rules adjust those scores for the things that flip
or amplify meaning: negation ("not good"), intensifiers ("very good"), emphasis
from capitals and exclamation marks, and contrastive clauses, where "the food
was cold but the service was wonderful" weights the half after "but" more
heavily — which is how people actually read such sentences. It works
immediately on any corpus, it explains itself, and it is domain-blind: it does
not know that "sick" is praise in some contexts, or that "unpredictable" is
positive for a novel and negative for a car.

The **supervised** route reuses a classifier you fitted on your own labelled
data. It learns your domain's vocabulary, so it will discover that "delayed" is
the strongest negative signal in your particular corpus. It needs labels.

The **transformer** route runs a pretrained sentiment model. Far more accurate
on general prose than a lexicon and needs no labels from you — but the model was
trained on somebody else's data, outside your split entirely, so its quality on
your text is an assumption rather than a measurement. That is disclosed on every
result.

A caution that applies to all three: sentiment is not a fact about a document.
Sarcasm, mixed opinions, and domain-specific praise defeat every method here,
and the compound score is a model's reading rather than a measurement of how a
writer felt.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.catalog import SENTIMENT_BACKENDS
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.lexicons import EMOTICONS, INTENSIFIERS, NEGATORS, SENTIMENT_LEXICON
from buildml.nlp.results import NlpSentimentResult, NlpTextPlan

# Normalization constant for the compound score (same shape as VADER's alpha):
# score / sqrt(score^2 + ALPHA) squashes an unbounded sum into (-1, 1).
COMPOUND_ALPHA = 15.0
NEGATION_WINDOW = 3
NEGATION_DAMPING = -0.74
EXCLAMATION_BOOST = 0.28
QUESTION_DAMPING = -0.10
ALL_CAPS_BOOST = 0.30
CONTRAST_MARKERS: frozenset[str] = frozenset({"but", "however", "although", "though"})

_TOKEN = re.compile(r"[A-Za-z]+(?:['\u2019\-][A-Za-z]+)*|[^\w\s]+|\d+")


def _classify(compound: float, *, threshold: float) -> str:
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"


def score_document(text: Any, *, threshold: float = 0.05) -> tuple[float, str, int]:
    """Score a single string with the built-in lexicon rules.

    The unit of work behind :func:`analyze_sentiment`, exposed for scoring one
    string at a time — a live comment, a quick check of how the rules read a
    particular phrase.

    Each token is looked up for valence, then adjusted: a negator within the
    preceding three tokens flips and dampens it, an intensifier scales it in
    proportion to how close it sits, capitals and exclamation marks amplify,
    question marks dampen, and a contrast marker halves everything before it
    while boosting everything after. The adjusted valences are summed and
    squashed into the range −1 to 1.

    Parameters
    ----------
    text:
        The string to score. ``None`` and non-strings are coerced; anything
        with no recognisable tokens scores zero.
    threshold:
        How far from zero the compound score must sit before the document is
        called positive or negative rather than neutral. The default of 0.05 is
        deliberately low, so only genuinely balanced documents land in the
        middle. Raise it when you want a confident call and are content to
        label the ambiguous ones neutral.

    Returns
    -------
    tuple
        ``(compound, label, n_matched_terms)``. The compound score runs from
        −1 to 1. The label is ``'positive'``, ``'negative'``, or ``'neutral'``.
        The match count is how many tokens were found in the lexicon at all —
        read it as the score's evidence base. A compound of −0.6 backed by one
        matched term is a single strong word, not a considered judgement.

    Notes
    -----
    **The score is bounded but not linear.** The squashing means an accumulating
    sum of valences approaches 1 without reaching it, so 0.9 and 0.95 are much
    further apart in raw evidence than 0.1 and 0.15. Compare scores by rank
    rather than by difference.

    **Coverage is the failure mode.** Domain jargon, product names, and
    non-English text are simply absent from the lexicon and contribute nothing,
    which produces a neutral verdict that looks like a judgement rather than a
    gap.

    Examples
    --------
    >>> score, label, matched = score_document("The service was not very good.")
    >>> label
    'negative'
    >>> score < 0
    True

    See Also
    --------
    analyze_sentiment : Score a whole partition and report aggregates.
    """
    value = "" if text is None else str(text)
    raw_tokens = _TOKEN.findall(value)
    if not raw_tokens:
        return 0.0, "neutral", 0

    lowered = [token.lower() for token in raw_tokens]
    valences: list[float] = []
    matched = 0
    contrast_at: int | None = None
    for index, token in enumerate(lowered):
        if token in CONTRAST_MARKERS and contrast_at is None:
            contrast_at = index

        weight = SENTIMENT_LEXICON.get(token)
        if weight is None and token in EMOTICONS:
            weight = EMOTICONS[token]
        if weight is None:
            valences.append(0.0)
            continue

        matched += 1
        current = float(weight)
        original = raw_tokens[index]
        if original.isupper() and len(original) > 2:
            current += ALL_CAPS_BOOST * (1.0 if current > 0 else -1.0)

        for distance in range(1, NEGATION_WINDOW + 1):
            previous_index = index - distance
            if previous_index < 0:
                break
            previous = lowered[previous_index]
            booster = INTENSIFIERS.get(previous)
            if booster is not None:
                # Modifiers lose strength with distance and scale the existing
                # magnitude, so "very bad" gets more negative rather than positive.
                scale = max(0.0, 1.0 - (distance - 1) * 0.25)
                current += current * booster * scale
            if previous in NEGATORS:
                current *= NEGATION_DAMPING
                break
        valences.append(current)

    if contrast_at is not None:
        for index in range(len(valences)):
            if index < contrast_at:
                valences[index] *= 0.5
            else:
                valences[index] *= 1.5

    total = float(sum(valences))
    exclamations = min(value.count("!"), 4)
    questions = min(value.count("?"), 4)
    if total != 0.0:
        sign = 1.0 if total > 0 else -1.0
        total += sign * exclamations * EXCLAMATION_BOOST
        total += sign * questions * QUESTION_DAMPING

    compound = total / math.sqrt(total * total + COMPOUND_ALPHA) if total else 0.0
    return float(compound), _classify(compound, threshold=threshold), matched


def analyze_sentiment(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    backend: str = "lexicon",
    text_column: str | None = None,
    threshold: float = 0.05,
    text_plan: NlpTextPlan | None = None,
    compare_to_target: bool = False,
    transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu",
) -> NlpSentimentResult:
    """Score every document in a partition for sentiment, and summarise the result.

    Applies the chosen backend across the partition and reports per-document
    scores alongside the distribution — how much of the corpus reads positive,
    negative, and neutral — which is usually the number people actually want.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to score.
    backend:
        ``'lexicon'``, ``'supervised'``, or ``'transformer'``. See the module
        docstring for what each assumes and what each costs.
    text_column:
        Which column holds the documents. Inferred from roles and dtype when
        omitted.
    threshold:
        How far from zero a compound score must sit to be called positive or
        negative rather than neutral. Applies to the lexicon backend.
    text_plan:
        The fitted classifier the ``'supervised'`` backend scores with,
        normally from :func:`~buildml.nlp.fit.fit_text_classifier`. Required
        for that backend and ignored by the others.
    compare_to_target:
        Also compare the predicted sentiment against the dataset's target
        column. This is how you find out whether the lexicon actually agrees
        with your labels — worth doing before trusting an unsupervised score,
        and only meaningful when the target really is a sentiment label.
    transformer_model:
        Which pretrained sentiment model the ``'transformer'`` backend loads.
        The default is a DistilBERT fine-tuned on movie reviews, which
        transfers reasonably to general prose and less well to specialised
        domains.
    device:
        Where to run the transformer. ``'cuda'`` is far faster where available.

    Returns
    -------
    ~buildml.nlp.results.NlpSentimentResult
        Per-document scores and labels, the distribution across the partition,
        the optional agreement with the target, and the disclosures that apply
        to the backend used.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The backend name is unknown, ``'supervised'`` was requested without a
        ``text_plan``, the text column cannot be resolved, or
        ``compare_to_target`` was set on a dataset with no target role.
    ~buildml.core.errors.MissingExtraError
        The transformer backend was requested without ``buildml[nlp]``
        installed.

    Notes
    -----
    **Check the match count before trusting a lexicon score.** A document where
    no token appeared in the lexicon scores exactly zero and is labelled
    neutral — indistinguishable in the output from a document that genuinely
    balances. On domain-specific text this can be most of your corpus.

    **The transformer's training data is outside your split.** It saw text you
    have no record of, so nothing here can tell you whether it generalises to
    your documents. Use ``compare_to_target`` against a labelled sample if the
    answer matters.

    **Neutral is not the midpoint of an opinion.** All three backends assign
    neutral both to balanced documents and to documents they cannot read.
    Treat a large neutral share as a coverage question first.

    Examples
    --------
    >>> result = analyze_sentiment(dataset, split_plan, partition="test")  # doctest: +SKIP
    >>> result.distribution  # doctest: +SKIP
    {'positive': 0.61, 'neutral': 0.22, 'negative': 0.17}

    See Also
    --------
    score_document : Score a single string.
    buildml.nlp.fit.fit_text_classifier : Train the supervised backend's model.
    """
    backend_key = str(backend).lower()
    if backend_key not in SENTIMENT_BACKENDS:
        raise ValidationError(
            f"backend={backend!r} is not supported. "
            f"Choose from {list(SENTIMENT_BACKENDS)}."
        )
    if not 0.0 <= threshold < 1.0:
        raise ValidationError("threshold must be within [0.0, 1.0).")

    column = (
        text_plan.text_column
        if (backend_key == "supervised" and text_plan is not None and text_column is None)
        else resolve_text_column(dataset, text_column)
    )
    documents, frame = documents_for(
        dataset, split_plan, partition, column, operation="analyze_sentiment"
    )

    warnings: list[str] = []
    disclosures: list[str] = []
    matched_rate: float | None = None

    if backend_key == "lexicon":
        scored = [score_document(item, threshold=threshold) for item in documents]
        scores = tuple(float(item[0]) for item in scored)
        labels = tuple(str(item[1]) for item in scored)
        matched_total = sum(item[2] for item in scored)
        matched_documents = sum(1 for item in scored if item[2] > 0)
        matched_rate = float(matched_documents / len(documents)) if documents else 0.0
        disclosures.extend(
            [
                f"Rule-based lexicon backend: {len(SENTIMENT_LEXICON)} valence terms "
                f"plus negation, degree modifiers, emoticons, caps, and punctuation "
                f"emphasis. Compound scores are squashed into (-1, 1).",
                f"{matched_rate:.1%} of documents matched at least one lexicon term "
                f"({matched_total} matches in total).",
                "Unsupervised and English-centred: it was not fitted on this "
                "dataset and carries no domain calibration.",
            ]
        )
        if matched_rate < 0.35:
            warnings.append(
                f"Only {matched_rate:.1%} of documents matched a lexicon term; "
                "domain vocabulary is likely uncovered. Prefer "
                "backend='supervised' with labelled data."
            )
    elif backend_key == "supervised":
        if text_plan is None:
            raise ValidationError(
                "backend='supervised' requires a fitted text classifier. Call "
                "fit_text_classifier(...) first (its classes become the sentiment "
                "labels)."
            )
        from buildml.nlp.predict import predict_documents

        predictions, probabilities = predict_documents(
            text_plan, documents, return_probabilities=True
        )
        labels = tuple(str(item) for item in predictions)
        scores = _supervised_scores(text_plan, labels, probabilities)
        disclosures.extend(
            [
                f"Supervised backend: reuses the fitted "
                f"{text_plan.estimator} head over "
                f"{text_plan.backend} features (classes={list(text_plan.classes_)}).",
                "Scoring is transform-only; the classifier was fitted on train "
                "documents only.",
            ]
        )
        if not text_plan.supports_proba:
            warnings.append(
                f"estimator='{text_plan.estimator}' has no predict_proba; scores "
                "are signed label codes rather than calibrated intensities."
            )
    else:
        labels, scores, notes = _transformer_sentiment(
            documents, model_name=transformer_model, device=device, threshold=threshold
        )
        disclosures.extend(notes)

    counts = {
        "positive": sum(1 for item in labels if item == "positive"),
        "negative": sum(1 for item in labels if item == "negative"),
        "neutral": sum(1 for item in labels if item not in {"positive", "negative"}),
    }
    total_rows = max(1, len(labels))
    agreement: dict[str, float] = {}
    if compare_to_target:
        agreement = _target_agreement(dataset, frame, labels)
        if agreement:
            disclosures.append(
                "Agreement with the labelled target is descriptive only; a "
                "rule-based scorer is not a fitted model of this dataset."
            )
        else:
            warnings.append(
                "compare_to_target=True but no comparable target column was "
                "found; agreement was not computed."
            )

    if partition in {"validation", "test", "all"} and backend_key == "lexicon":
        disclosures.append(
            f"partition={partition!r}: the lexicon backend learns nothing, so this "
            "is scoring rather than fitting."
        )

    return NlpSentimentResult(
        partition=str(partition),
        backend=backend_key,
        n_rows=len(documents),
        labels=labels,
        scores=scores,
        positive_rate=float(counts["positive"] / total_rows),
        negative_rate=float(counts["negative"] / total_rows),
        neutral_rate=float(counts["neutral"] / total_rows),
        mean_score=float(np.mean(scores)) if scores else 0.0,
        matched_term_rate=matched_rate,
        agreement=agreement,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _supervised_scores(
    plan: NlpTextPlan,
    labels: tuple[str, ...],
    probabilities: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    """Map classifier output onto a signed intensity in [-1, 1]."""
    classes = [str(item) for item in plan.classes_]
    polarity: dict[str, float] = {}
    for label in classes:
        lowered = label.lower()
        if lowered in {"positive", "pos", "1", "true", "good", "up"}:
            polarity[label] = 1.0
        elif lowered in {"negative", "neg", "0", "false", "bad", "down"}:
            polarity[label] = -1.0
        else:
            polarity[label] = 0.0
    if all(value == 0.0 for value in polarity.values()) and len(classes) == 2:
        polarity[classes[0]] = -1.0
        polarity[classes[1]] = 1.0

    if not probabilities:
        return tuple(float(polarity.get(label, 0.0)) for label in labels)
    out: list[float] = []
    for row in probabilities:
        total = 0.0
        for index, label in enumerate(classes):
            if index < len(row):
                total += polarity[label] * float(row[index])
        out.append(float(max(-1.0, min(1.0, total))))
    return tuple(out)


def _transformer_sentiment(
    documents: list[str],
    *,
    model_name: str,
    device: str,
    threshold: float,
) -> tuple[tuple[str, ...], tuple[float, ...], list[str]]:
    from buildml.nlp.extras import require_transformers

    transformers = require_transformers(feature="transformer sentiment scoring")
    try:
        pipeline = transformers.pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1 if device == "cpu" else 0,
            truncation=True,
        )
    except Exception as exc:  # pragma: no cover - network / model errors
        raise ValidationError(
            f"Could not load sentiment model {model_name!r}: {exc}"
        ) from exc

    labels: list[str] = []
    scores: list[float] = []
    for output in pipeline([item or "" for item in documents], batch_size=16):
        raw_label = str(output.get("label", "")).lower()
        confidence = float(output.get("score", 0.0))
        signed = confidence if raw_label.startswith("pos") else -confidence
        if raw_label.startswith("neu"):
            signed = 0.0
        scores.append(signed)
        labels.append(_classify(signed, threshold=threshold))
    notes = [
        f"Transformer backend: pretrained sentiment head {model_name!r}; the "
        "encoder and head were trained outside this Session.",
        "The model's training corpus is not covered by the Session split; treat "
        "agreement with your labels as an external benchmark.",
    ]
    return tuple(labels), tuple(scores), notes


def _target_agreement(
    dataset: Dataset,
    frame: Any,
    labels: tuple[str, ...],
) -> dict[str, float]:
    """Compare predicted polarity against a binary/ternary sentiment target."""
    try:
        target_column = dataset.require_target()
    except Exception:
        return {}
    if target_column not in frame.columns:
        return {}
    truth_raw = frame[target_column]
    if truth_raw.isna().any():
        return {}

    mapping = {
        "1": "positive", "0": "negative", "-1": "negative",
        "true": "positive", "false": "negative",
        "pos": "positive", "neg": "negative", "neu": "neutral",
        "positive": "positive", "negative": "negative", "neutral": "neutral",
        "good": "positive", "bad": "negative",
    }
    truth = [mapping.get(str(item).strip().lower()) for item in truth_raw]
    usable = [
        (expected, actual)
        for expected, actual in zip(truth, labels, strict=False)
        if expected is not None
    ]
    if not usable:
        return {}
    hits = sum(1 for expected, actual in usable if expected == actual)
    non_neutral = [
        (expected, actual) for expected, actual in usable if actual != "neutral"
    ]
    directional = (
        sum(1 for expected, actual in non_neutral if expected == actual)
        / len(non_neutral)
        if non_neutral
        else 0.0
    )
    return {
        "n_compared": float(len(usable)),
        "agreement": float(hits / len(usable)),
        "directional_agreement": float(directional),
        "neutral_share": float(
            sum(1 for _, actual in usable if actual == "neutral") / len(usable)
        ),
    }


__all__ = [
    "COMPOUND_ALPHA",
    "analyze_sentiment",
    "score_document",
]
