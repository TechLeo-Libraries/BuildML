"""Save an ensemble with the record of how it was combined.

BuildML has several artifact formats, and the reason they are separate is that
they answer different questions. A Session checkpoint resumes your work: data,
roles, splits, history. A pipeline bundle deploys a model: preprocessing plus
estimator. An ensemble bundle preserves *the ensemble as an ensemble*: the
strategy, the base model names, the meta-learner, and the disclosures about
which rows were used where.

That last part is what a pipeline bundle would lose. Pickling a fitted
``StackingClassifier`` keeps the object but not the account of how it was built,
and for an ensemble that account is the difference between a defensible model
and an opaque one. Which rows the meta-learner saw is exactly the question a
reviewer asks.

See Also
--------
buildml.pipeline.bundle : Preprocessing plus estimator, for serving.
buildml.checkpoint.bundle : Session state, for resuming work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.ensemble.results import EnsembleFitResult, EnsemblePlan
from buildml.model.supervised import FitResult

BUNDLE_FORMAT = "buildml.ensemble_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Ensemble bundles, classical pipeline bundles, unsupervised bundles, Torch "
    "trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. "
    "An ensemble bundle (buildml.ensemble_bundle.v1) stores a train-fitted "
    "EnsemblePlan (strategy disclosures + sklearn-compatible ensemble estimator) "
    "and the classical FitResult feature contract. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the ensemble. "
    "Prefer save_pipeline when preprocess plans must travel with the estimator; "
    "use save_ensemble_bundle when strategy disclosures and EnsemblePlan matter."
)


def save_ensemble_bundle(
    path: str | Path,
    plan: EnsemblePlan,
    *,
    fit_result: FitResult | None = None,
    ensemble_fit_result: EnsembleFitResult | None = None,
) -> Path:
    """Write the ensemble to a directory: one file to load, one file to read.

    Two files, deliberately. ``ensemble_plan.joblib`` holds the fitted objects
    and can only be read back by Python with the same libraries installed.
    ``meta.json`` holds the same facts as text: strategy, bases, disclosures,
    metrics: and can be read by anything, forever.

    That second file is what makes a bundle auditable. A year from now, when
    scikit-learn has moved on and the pickle no longer loads, the JSON still
    says what the model was and how it was fitted.

    Parameters
    ----------
    path:
        Directory to write. Created if absent; existing files with these names
        are overwritten.
    plan:
        The fitted plan from one of the ``fit_*_ensemble`` functions.
    fit_result:
        The standard fit result. Worth including: its feature contract is what
        lets a reloaded ensemble check that incoming data matches.
    ensemble_fit_result:
        The disclosure record, written into the metadata only.

    Returns
    -------
    Path
        The directory written.

    Raises
    ------
    ValidationError
        If ``plan`` is ``None``, which usually means an ensemble was never
        fitted in this session.
    OSError
        If the directory cannot be created or written.

    Notes
    -----
    **The joblib file is a pickle.** Only load bundles you produced or trust;
    unpickling runs code. It is also tied to the library versions that wrote it,
    which is why the JSON exists alongside.

    **Preprocessing is not included.** If the ensemble was fitted on transformed
    features, the transformations live in the Session's plans, and reloading
    this bundle alone will not reproduce them. Use
    :func:`~buildml.pipeline.bundle.save_pipeline_bundle` when the preprocessing
    must travel with the model.

    Examples
    --------
    ::

        plan, ensemble, fit = fit_stacking_ensemble(dataset, split_plan, models)
        save_ensemble_bundle(
            "artifacts/churn_stack",
            plan,
            fit_result=fit,
            ensemble_fit_result=ensemble,
        )

    See Also
    --------
    load_ensemble_bundle : Reading it back.
    """
    if plan is None:
        raise ValidationError("No EnsemblePlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan,
        "fit_result": fit_result,
    }
    joblib.dump(payload, destination / "ensemble_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "ensemble_fit": None
        if ensemble_fit_result is None
        else ensemble_fit_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_ensemble_bundle(path: str | Path, *, trusted: bool = False) -> tuple[EnsemblePlan, FitResult | None]:
    """Read a bundle back, reconstructing the fit result if it was not stored.

    Checks the format marker before unpickling, so a directory that is not a
    BuildML ensemble bundle fails with a clear message rather than an obscure
    unpickling error.

    The fit result is always returned, even for bundles written before it was
    stored: everything it needs: estimator, task, feature columns, target, row
    count: is already in the plan, so it can be rebuilt. That keeps the return
    type stable, and means downstream code never has to branch on the age of the
    bundle it was handed.

    Parameters
    ----------
    path:
        The bundle directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    tuple
        ``(EnsemblePlan, FitResult)``. The second element is typed optional for
        backwards compatibility but is populated in practice.

    Raises
    ------
    ValidationError
        If either file is missing, if the format marker does not match, or if
        the payload does not contain an ``EnsemblePlan``.
    OSError
        If the files cannot be read.

    Notes
    -----
    **Unpickling executes code.** Load only bundles from a source you trust.

    **Version drift shows up here.** A bundle written under a different
    scikit-learn may load with warnings or fail outright; ``meta.json`` records
    the BuildML version that wrote it, which is where to start when it does.

    Examples
    --------
    ::

        plan, fit = load_ensemble_bundle("artifacts/churn_stack")
        print(plan.strategy, plan.estimator_names)
        for note in plan.disclosures:
            print(note)

    See Also
    --------
    save_ensemble_bundle : Writing it.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "ensemble_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete ensemble bundle at {root}. "
            f"Expected meta.json and ensemble_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported ensemble bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, EnsemblePlan):
        plan = loaded
        fit_result = None
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        fit_result = loaded.get("fit_result")
    else:
        raise ValidationError(
            "ensemble_plan.joblib must contain an EnsemblePlan or a payload with key 'plan'."
        )
    if not isinstance(plan, EnsemblePlan):
        raise ValidationError("Loaded plan object is not an EnsemblePlan")
    if fit_result is not None and not isinstance(fit_result, FitResult):
        # Reconstruct a FitResult from the plan estimator when payload is partial.
        fit_result = FitResult(
            estimator=plan.estimator_,
            task=plan.task,
            feature_columns=plan.feature_columns,
            target_column=plan.target_column,
            n_train_rows=plan.n_train_rows,
        )
    elif fit_result is None:
        fit_result = FitResult(
            estimator=plan.estimator_,
            task=plan.task,
            feature_columns=plan.feature_columns,
            target_column=plan.target_column,
            n_train_rows=plan.n_train_rows,
        )
    return plan, fit_result
