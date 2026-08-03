"""Adapt deep learning results into the shapes the explain surfaces expect.

Session history, walkthroughs, and the Teaching Studio all want a JSON-safe
summary of what an operation produced. The DL result objects already know how to
describe themselves, so most of what happens here is calling ``to_dict`` at a
name the explain layer recognises.

The indirection earns its place in two ways. It gives the explain layer a stable
set of names to import, so a rename inside :mod:`buildml.dl.results` does not
ripple outward. And it is where the non-trivial cases live: the training curve
is rebuilt when a reloaded result did not carry one, and the Session-level status
handles the case where training happened but the live result is gone.

Teaching prose belongs in :mod:`buildml.explain`; API contracts belong in
docstrings. This module carries neither, only the adaptation between them.

See Also
--------
buildml.dl.results : The result objects being summarised.
buildml.dl.curves : Training curves and their interpretation.
"""

from __future__ import annotations

from typing import Any

from buildml.dl.curves import build_training_curve, torch_training_status
from buildml.dl.results import DLEvaluateResult, LoaderReport, TrainResult

try:
    from buildml.dl.catalog import dl_capability_matrix
except ImportError:  # pragma: no cover
    dl_capability_matrix = None  # type: ignore[assignment,misc]


def loader_summary(report: LoaderReport) -> dict[str, Any]:
    """Summarise loader construction for Session history.

    Records what ``make_torch_loaders`` built: partition sizes, batching,
    whether normalisation ran, the columns and task, and the split-integrity
    checks.

    Parameters
    ----------
    report:
        The report from a loader bundle.

    Returns
    -------
    dict
        The report's JSON-safe form.

    See Also
    --------
    buildml.dl.results.LoaderReport : What is being summarised.
    """
    return report.to_dict()


def train_summary(result: TrainResult) -> dict[str, Any]:
    """Summarise a training run for Session history.

    Records what ``fit_torch`` produced. Weights and optimiser state are
    described rather than embedded: history is a log, and a log with tensors in
    it is neither readable nor small.

    Parameters
    ----------
    result:
        The training outcome.

    Returns
    -------
    dict
        The result's JSON-safe form.

    See Also
    --------
    buildml.dl.results.TrainResult : What is being summarised.
    buildml.dl.checkpoint : Where the actual weights are persisted.
    """
    return result.to_dict()


def evaluate_summary(result: DLEvaluateResult) -> dict[str, Any]:
    """Summarise an evaluation for Session history.

    Records what ``evaluate_torch`` found, including the confusion matrix or
    residual summary: the parts that distinguish one kind of failure from
    another.

    Parameters
    ----------
    result:
        The evaluation outcome.

    Returns
    -------
    dict
        The result's JSON-safe form.

    See Also
    --------
    buildml.dl.results.DLEvaluateResult : What is being summarised.
    """
    return result.to_dict()


def curve_summary(result: TrainResult) -> dict[str, Any]:
    """Produce the training curve payload the teaching surfaces render.

    Uses the curve already on the result when there is one, and derives it from
    the epoch history otherwise: which is the case after loading a bundle,
    where the history survives but the derived curve does not.

    Parameters
    ----------
    result:
        The training outcome.

    Returns
    -------
    dict
        The curve's JSON-safe form: the series, plus the interpretation,
        limitations, and disclosures that come with them.

    See Also
    --------
    buildml.dl.curves.build_training_curve : The derivation.
    """
    curve = result.training_curve or build_training_curve(result)
    return curve.to_dict()


def training_status_for_session(session: Any) -> dict[str, Any]:
    """Report Torch training state for a Session, live result or not.

    Reads the Session's attached training result when present, and falls back to
    scanning its history when it is not.

    Parameters
    ----------
    session:
        A Session. Read defensively, so one that has never touched the DL path
        works fine.

    Returns
    -------
    dict
        The status: whether a live result is attached, whether training appears
        in the history, and the curve and device detail when available.

    Notes
    -----
    **Training present without a live result is a real state, not an error.**
    It happens after reloading a Session from a checkpoint, since checkpoints
    hold data and history rather than model weights. The returned disclosure
    says so, which is more useful than reporting that nothing was trained.

    See Also
    --------
    buildml.dl.curves.torch_training_status : Where the logic lives.
    """
    status = torch_training_status(
        train_result=getattr(session, "dl_train_result", None),
        history=list(getattr(session, "history", []) or []),
    )
    if dl_capability_matrix is not None:
        status = {**status, "capability_matrix": dl_capability_matrix()}
    return status
