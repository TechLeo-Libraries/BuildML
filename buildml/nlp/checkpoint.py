"""Save and reload trained text models, so they can score documents later.

An NLP bundle is a directory holding the fitted text plan: the normalisation
recipe, the train-fitted representation, and the classifier head: plus an
optional topic plan and a readable ``meta.json`` describing what is inside.

All three parts must travel together, and that is the entire reason this exists
rather than a plain pickle of the estimator. A head separated from its
vectorizer receives features in a different column layout, and a vectorizer
separated from its normalisation plan receives different tokens. Neither
mismatch raises; both silently produce wrong predictions.

BuildML has several persistence formats, and they are complementary rather than
alternatives. A **Session checkpoint** stores data, roles, splits, and history so
you can resume working: it does not embed the NLP vectorizer or head. An **NLP
bundle** stores the text model so it can score documents, and knows nothing about
your session. Use both if you need both; neither substitutes for the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.nlp.results import (
    NlpEvalResult,
    NlpFitResult,
    NlpTextPlan,
    NlpTopicPlan,
)

BUNDLE_FORMAT = "buildml.nlp_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "NLP bundles, classical pipeline bundles, Torch trainer bundles, RAG bundles, "
    "CBR bundles, symbolic bundles, and Session checkpoints are complementary, not "
    "interchangeable. An NLP bundle (buildml.nlp_bundle.v1) stores the trained "
    "text plan (normalization plan, train-fitted representation, fitted head) and "
    "optionally a topic plan. A Session checkpoint stores data, roles, splits, "
    "history, and optional classical preprocess plans; it does not embed the NLP "
    "vectorizer or head. Reload the tabular workflow via checkpoint_load; reload "
    "the text model via load_nlp_bundle. Honesty: document-level text modelling "
    "and analysis: not document retrieval for generation (buildml.rag), not "
    "transformer fine-tuning (buildml.dl text path)."
)

_TEXT_PLAN_FILE = "nlp_text_plan.joblib"
_TOPIC_PLAN_FILE = "nlp_topic_plan.joblib"


def save_nlp_bundle(
    path: str | Path,
    plan: NlpTextPlan | None,
    *,
    topic_plan: NlpTopicPlan | None = None,
    fit_result: NlpFitResult | None = None,
    eval_result: NlpEvalResult | None = None,
) -> Path:
    """Save a trained text model so it can score documents in another process.

    Writes a directory containing the fitted plans and a readable manifest.
    The manifest is plain JSON, so what a bundle contains: which columns,
    which classes, which normalisation, and any warnings from the fit: can be
    inspected without loading the model or even having BuildML installed.

    Parameters
    ----------
    path:
        Directory to write to. Created if it does not exist; an existing
        bundle at the same path is overwritten.
    plan:
        The fitted text classifier from
        :func:`~buildml.nlp.fit.fit_text_classifier`. May be ``None`` if you
        are saving only a topic plan.
    topic_plan:
        A fitted topic model from :func:`~buildml.nlp.topics.fit_topics`, saved
        alongside. Both can live in one bundle.
    fit_result:
        The fit report, recorded in the manifest. Worth including: it is what
        lets someone later see the class balance and vocabulary size the model
        was built on.
    eval_result:
        The holdout evaluation, recorded in the manifest: the record of how
        well this model performed when it was made, which is what you compare
        against when it starts behaving differently in production.

    Returns
    -------
    ~pathlib.Path
        The bundle directory.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        Neither a text plan nor a topic plan was supplied. An empty bundle
        would fail confusingly at load time instead.

    Notes
    -----
    **The plans are pickled with joblib**, which means they carry the
    scikit-learn objects and their version expectations with them. A bundle
    written under one scikit-learn version may warn or fail under a very
    different one; the manifest records the BuildML version to help diagnose
    that.

    **Only the manifest is human-readable.** Everything needed to score is in
    the joblib files.

    See Also
    --------
    load_nlp_bundle : Read a bundle back.
    """
    if plan is None and topic_plan is None:
        raise ValidationError(
            "No NLP plan to save. Call fit_text_classifier(...) or fit_topics(...) "
            "before save_nlp_bundle."
        )
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        joblib.dump({"plan": plan}, destination / _TEXT_PLAN_FILE)
    if topic_plan is not None:
        joblib.dump({"plan": topic_plan}, destination / _TOPIC_PLAN_FILE)

    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "kind": "nlp",
        "has_text_plan": plan is not None,
        "has_topic_plan": topic_plan is not None,
        "text_plan": None if plan is None else plan.to_dict(),
        "topic_plan": None if topic_plan is None else topic_plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return destination


def load_nlp_bundle(path: str | Path, *, trusted: bool = False) -> tuple[NlpTextPlan | None, NlpTopicPlan | None]:
    """Restore a saved text model, ready to score documents.

    Reads the manifest, checks the format, and loads whichever plans the bundle
    contains. The returned plans are immediately usable: no dataset, no split,
    and no refitting required.

    Parameters
    ----------
    path:
        The bundle directory written by :func:`save_nlp_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    tuple
        ``(text_plan, topic_plan)``. Either may be ``None`` when the bundle
        does not contain it, but not both.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The directory has no ``meta.json``; the format is not
        ``buildml.nlp_bundle.v1``; the bundle declares the format but contains
        no plan files; or a plan file holds something other than the expected
        plan type.

    Notes
    -----
    **Check for ``None`` before using either plan.** A bundle saved with only a
    topic model returns ``None`` for the text plan, and the failure otherwise
    surfaces as an attribute error somewhere less obvious.

    **Loading executes pickled objects.** Only load bundles you trust or
    produced yourself: this is a property of the pickle format, not of
    BuildML.

    Examples
    --------
    >>> text_plan, topic_plan = load_nlp_bundle("artifacts/ticket-model")  # doctest: +SKIP
    >>> predict_documents(text_plan, ["my card was declined"])  # doctest: +SKIP

    See Also
    --------
    save_nlp_bundle : Write a bundle.
    buildml.nlp.predict.predict_documents : Score raw strings with a loaded plan.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise ValidationError(
            f"Incomplete NLP bundle at {root}. Expected meta.json ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported NLP bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )

    text_plan = _load_plan(root / _TEXT_PLAN_FILE, NlpTextPlan, "text", trusted=trusted)
    topic_plan = _load_plan(root / _TOPIC_PLAN_FILE, NlpTopicPlan, "topic", trusted=trusted)
    if text_plan is None and topic_plan is None:
        raise ValidationError(
            f"NLP bundle at {root} declares {BUNDLE_FORMAT} but contains neither "
            f"{_TEXT_PLAN_FILE} nor {_TOPIC_PLAN_FILE}."
        )
    return text_plan, topic_plan


def _load_plan(path: Path, expected: type, kind: str, *, trusted: bool) -> Any:
    if not path.is_file():
        return None
    loaded = joblib_load_trusted(path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, expected):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            f"{path.name} must contain a plan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, expected):
        raise ValidationError(
            f"Loaded {kind} plan object is not a {expected.__name__}."
        )
    return plan


__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "load_nlp_bundle",
    "save_nlp_bundle",
]
