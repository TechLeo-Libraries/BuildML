"""Save a fitted case-based reasoner to disk and load it back.

A bundle is a directory holding two files: ``cbr_plan.joblib`` with the plan
itself, and ``meta.json`` describing it in plain text. The description exists so
a bundle can be identified: format, version, configuration, case counts, and
any recorded scores: without loading the plan, which matters both for tooling
and for deciding whether you trust the file before executing it.

**A CBR bundle contains your training data.** Case memory is the training rows,
so the bundle carries whatever those rows contained. Treat it with the same care
as the source dataset.

**This is not a Session checkpoint.** A checkpoint stores data, roles, splits,
history, and classical plans; it does not store case memory. Restore a
checkpoint and there is no reasoner until a bundle is loaded or a fit is re-run.
The boundary is written into every bundle's metadata for the same reason it is
written here.

See Also
--------
buildml.cbr.fit.fit_cbr : Producing what gets saved.
buildml.cbr.results.CbrPlan : What a bundle holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.cbr.results import CbrEvalResult, CbrFitResult, CbrPlan
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError

BUNDLE_FORMAT = "buildml.cbr_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "CBR bundles, classical pipeline bundles, Torch trainer bundles, "
    "RAG bundles, symbolic bundles, and Session checkpoints are complementary, "
    "not interchangeable. A CBR bundle (buildml.cbr_bundle.v1) stores a CbrPlan "
    "(train-built case memory + metric/reuse config). A Session checkpoint stores "
    "data, roles, splits, history, and optional classical preprocess plans; it "
    "does not embed the case memory. Reload tabular workflow via checkpoint_load; "
    "reload the learner via load_cbr_bundle. Honesty: tabular case→solution CBR "
    ": not RAG document retrieval, not a vector DB product."
)


def save_cbr_bundle(
    path: str | Path,
    plan: CbrPlan,
    *,
    fit_result: CbrFitResult | None = None,
    eval_result: CbrEvalResult | None = None,
) -> Path:
    """Persist a fitted reasoner, with a readable description alongside it.

    Writes the plan to ``cbr_plan.joblib`` and a summary to ``meta.json``. The
    optional fit and evaluation results are recorded in the metadata, so a
    bundle can carry evidence of how well it performed rather than only what it
    is.

    Parameters
    ----------
    path:
        Destination directory. Created if needed; existing bundle files are
        overwritten.
    plan:
        The fitted reasoner.
    fit_result:
        The fit report, recorded in the metadata.
    eval_result:
        The holdout evaluation, recorded in the metadata. Worth including :
        without it, a loaded bundle gives no indication of whether it works.

    Returns
    -------
    Path
        The bundle directory.

    Raises
    ------
    ValidationError
        If ``plan`` is ``None``.
    OSError
        If the directory cannot be created or written.

    Notes
    -----
    **Size scales with the training set**, since the case base is the training
    rows. There is no compression step to shrink it.

    **The plan is pickled via joblib**, which means loading executes code from
    the file. Only load bundles you produced or trust.

    **Writes are not atomic.** An interrupted save can leave a partial bundle;
    :func:`load_cbr_bundle` checks for both files, so a partial bundle fails
    cleanly rather than loading something wrong.

    Examples
    --------
    Save with its evaluation attached::

        save_cbr_bundle("artifacts/triage_cbr", plan, eval_result=metrics)

    See Also
    --------
    load_cbr_bundle : The other half.
    """
    if plan is None:
        raise ValidationError("No CbrPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "cbr_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "kind": "cbr",
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return destination


def load_cbr_bundle(path: str | Path, *, trusted: bool = False) -> CbrPlan:
    """Restore a fitted reasoner from a bundle directory.

    Checks the format marker before unpickling, so a Session checkpoint or a
    bundle from another domain fails with a clear message rather than an obscure
    deserialisation error. Accepts both the current payload layout and a bare
    plan, since early bundles were written that way.

    Parameters
    ----------
    path:
        The bundle directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    CbrPlan
        The reasoner, ready to predict, evaluate, retrieve, and retain.

    Raises
    ------
    ValidationError
        If either file is missing, the format marker does not match
        ``buildml.cbr_bundle.v1``, or the payload does not contain a plan.
    OSError
        If the files cannot be read.

    Notes
    -----
    **Loading executes code from the file**, as any joblib load does. Only load
    bundles from a source you trust.

    **The whole case base loads into memory**, so a bundle built from a large
    training set needs that much resident before the first query.

    **Read ``meta.json`` first when a bundle's provenance is unclear.** It is
    plain text and describes the plan without executing anything.

    Examples
    --------
    Reload and predict::

        plan = load_cbr_bundle("artifacts/triage_cbr")
        result = predict_cbr(dataset, plan, split_plan, partition="test")

    See Also
    --------
    save_cbr_bundle : Producing the bundle.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "cbr_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete CBR bundle at {root}. "
            f"Expected meta.json and cbr_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported CBR bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, CbrPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "cbr_plan.joblib must contain a plan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, CbrPlan):
        raise ValidationError("Loaded plan object is not a CbrPlan.")
    return plan
