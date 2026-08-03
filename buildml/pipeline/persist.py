"""Save just the estimator, when the transforms are handled elsewhere.

The minimal artifact: the fitted estimator and the metadata describing what it
expects. No plans, no contract, no card.

That makes it the wrong choice most of the time. A model trained on preprocessed
data needs those transforms to predict correctly, and saving it alone is how a
deployed model ends up scoring raw rows against training-time assumptions.
Reach for :mod:`buildml.pipeline.bundle` unless the preprocessing genuinely
lives outside BuildML: an upstream feature store, an ETL job: in which case
this is exactly right and a bundle would only add empty files.

See Also
--------
buildml.pipeline.bundle : The full artifact, for models that own their transforms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.model.supervised import FitResult


def save_fit_result(path: str | Path, fit_result: FitResult) -> Path:
    """Write the estimator and its feature contract to a directory.

    Two files: ``model.joblib`` holding the pickled estimator, and ``meta.json``
    recording the task, feature columns in order, target name, training row
    count, and the BuildML version that wrote it. The metadata is readable
    without unpickling, which is what lets you check what an artifact expects
    before loading it.

    Parameters
    ----------
    path:
        The destination directory, created if missing. Existing files are
        overwritten.
    fit_result:
        The fitted result to save. Only the estimator and its contract are
        written; any training scores it carries are not.

    Returns
    -------
    Path
        The directory written to.

    Raises
    ------
    OSError
        If the directory cannot be created or the files written.

    Notes
    -----
    **Feature column order is preserved and matters.** Most estimators depend
    on positional order, so a score frame must present the same columns in the
    same sequence.

    **No preprocessing is saved.** If the model was trained on transformed data,
    this artifact alone cannot reproduce a prediction.

    See Also
    --------
    load_fit_result : Reading it back.
    buildml.pipeline.bundle.save_pipeline_bundle : Saving the transforms too.
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_result.estimator, root / "model.joblib")
    meta = {
        "buildml_version": __version__,
        **fit_result.to_dict(),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def load_fit_result(path: str | Path, *, trusted: bool = False) -> FitResult:
    """Restore the estimator and the feature contract it was saved with.

    Both files must be present. An estimator without its metadata gives no way
    to know which columns it expects or in what order, and guessing would
    produce predictions from misaligned features: so a partial artifact is
    rejected rather than half-loaded.

    Parameters
    ----------
    path:
        The directory written by :func:`save_fit_result`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    FitResult
        The estimator with its task, feature columns, target name, and training
        row count.

    Raises
    ------
    ValidationError
        If either file is missing.
    KeyError
        If the metadata lacks a required field, meaning it was not written by
        :func:`save_fit_result`.
    json.JSONDecodeError
        If the metadata is not valid JSON.

    Notes
    -----
    **Loading unpickles an estimator, which executes code.** Only load artifacts
    from a source you trust.

    **A scikit-learn version mismatch may warn or fail**, since the pickle is
    tied to the library that produced it.

    See Also
    --------
    save_fit_result : Writing it.
    """
    root = Path(path)
    model_path = root / "model.joblib"
    meta_path = root / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise ValidationError(f"Fit artifact incomplete at '{root}'")
    estimator = joblib_load_trusted(model_path, trusted=trusted, artifact="joblib plan")
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return FitResult(
        estimator=estimator,
        task=meta["task"],
        feature_columns=tuple(meta["feature_columns"]),
        target_column=meta["target_column"],
        n_train_rows=int(meta["n_train_rows"]),
    )
