"""Save a fitted policy so it outlives the session that produced it.

A Session checkpoint and a policy bundle answer different questions, and
conflating them is the mistake this module exists to prevent. **A checkpoint
resumes your work** — data, roles, splits, history, classical preprocessing —
and it does not embed IL or RL policies. **A bundle deploys a policy** — the
fitted model, its columns, its action vocabulary — and it carries none of your
data.

Fitted a bandit and want to score new rows next week? Save a bundle. Stopped
mid-analysis and want to pick up where you left off? Save a checkpoint. Doing
both is normal.

Each bundle is a directory of two files. ``meta.json`` is human-readable and
holds the configuration, the fit report, and any evaluation, so you can tell
what a bundle contains without loading it. The ``.joblib`` file holds the live
model objects.

Loading validates the format string before unpickling, which catches a wrong
directory early. It is not a security boundary: joblib executes code on load, so
only load bundles you produced or trust.

See Also
--------
buildml.rl.results : The plans these bundles carry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted
from buildml.core.errors import ValidationError
from buildml.rl.results import (
    ImitationEvalResult,
    ImitationFitResult,
    ImitationPlan,
    RlEvalResult,
    RlFitResult,
    RlPlan,
)

BUNDLE_FORMAT_IMITATION = "buildml.imitation_bundle.v1"
BUNDLE_FORMAT_RL = "buildml.rl_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Imitation bundles, RL bundles, CBR bundles, classical pipeline bundles, "
    "Torch trainer bundles, RAG bundles, and Session checkpoints are complementary, "
    "not interchangeable. An imitation bundle (buildml.imitation_bundle.v1) stores a "
    "train-fitted ImitationPlan (behavioral cloning policy). An RL bundle "
    "(buildml.rl_bundle.v1) stores a train-fitted RlPlan (contextual bandit or "
    "Gymnasium REINFORCE-lite policy). A Session checkpoint stores data, roles, "
    "splits, history, and optional classical preprocess plans; it does not embed "
    "IL/RL policies. Reload tabular workflow via checkpoint_load; reload policies "
    "via load_imitation_bundle / load_rl_bundle. Honesty: not MuJoCo/robotics."
)


def save_imitation_bundle(
    path: str | Path,
    plan: ImitationPlan,
    *,
    fit_result: ImitationFitResult | None = None,
    eval_result: ImitationEvalResult | None = None,
) -> Path:
    """Save a cloned policy to disk, with its provenance beside it.

    Writes a two-file directory: the live model in joblib form, and a readable
    ``meta.json`` recording what the policy is, how it was fitted, and how it
    scored.

    Parameters
    ----------
    path:
        The destination directory. Created if absent; an existing bundle at the
        same path is overwritten.
    plan:
        The fitted policy from
        :func:`~buildml.rl.imitation.fit_imitation`.
    fit_result:
        The fit report, recorded in the metadata. Optional but worth passing —
        it is the record of what the policy was trained on, and a bundle
        without it cannot answer that later.
    eval_result:
        The holdout evaluation, recorded in the metadata. Also worth passing:
        a deployed policy with no recorded score is a policy nobody can defend.

    Returns
    -------
    pathlib.Path
        The bundle directory.

    Raises
    ------
    ValidationError
        If ``plan`` is ``None``.

    Notes
    -----
    **This is not a Session checkpoint.** It holds the policy and no data, no
    splits, and no history. Save a checkpoint as well if you want to resume the
    analysis.

    See Also
    --------
    load_imitation_bundle : Read one back.
    save_rl_bundle : The RL counterpart.
    """
    if plan is None:
        raise ValidationError("No ImitationPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"plan": plan}, destination / "imitation_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT_IMITATION,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_imitation_bundle(path: str | Path, *, trusted: bool = False) -> ImitationPlan:
    """Read a saved cloning policy back into memory.

    Checks that both files are present and that the format string matches
    before unpickling, so a wrong path fails with a clear message rather than a
    confusing deserialisation error.

    Parameters
    ----------
    path:
        The bundle directory written by :func:`save_imitation_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    ImitationPlan
        The policy, ready for
        :func:`~buildml.rl.imitation.predict_imitation_action` and
        :func:`~buildml.rl.imitation.evaluate_imitation`.

    Raises
    ------
    ValidationError
        If either file is missing, if the format string is not
        ``buildml.imitation_bundle.v1``, or if the payload does not contain an
        :class:`~buildml.rl.results.ImitationPlan`.

    Notes
    -----
    **The bundle carries no data.** The reloaded policy expects the same state
    columns, in the same order, that it was fitted on — read them from
    ``plan.columns``.

    **Loading executes pickled code.** Only load bundles you produced or trust.

    See Also
    --------
    save_imitation_bundle : Write one.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "imitation_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete imitation bundle at {root}. "
            f"Expected meta.json and imitation_plan.joblib ({BUNDLE_FORMAT_IMITATION})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT_IMITATION:
        raise ValidationError(
            f"Unsupported imitation bundle format {fmt!r}; expected {BUNDLE_FORMAT_IMITATION}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, ImitationPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "imitation_plan.joblib must contain an ImitationPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, ImitationPlan):
        raise ValidationError("Loaded plan object is not an ImitationPlan")
    return plan


def save_rl_bundle(
    path: str | Path,
    plan: RlPlan,
    *,
    fit_result: RlFitResult | None = None,
    eval_result: RlEvalResult | None = None,
) -> Path:
    """Save an RL policy to disk, with its provenance beside it.

    Writes a two-file directory: the live policy in joblib form, and a readable
    ``meta.json`` recording the mode, algorithm, configuration, and any results.
    Works for all four modes — a bandit's columns and encoder, or an
    environment policy's weights, are carried the same way.

    Parameters
    ----------
    path:
        The destination directory. Created if absent; an existing bundle at the
        same path is overwritten.
    plan:
        The fitted policy from :func:`~buildml.rl.fit.fit_rl`.
    fit_result:
        The fit report, recorded in the metadata.
    eval_result:
        The evaluation, recorded in the metadata. Particularly worth passing
        here, because it carries the ``offline`` flag — a saved bandit metric
        that has lost track of whether it was estimated or measured is a metric
        that will eventually be over-read.

    Returns
    -------
    pathlib.Path
        The bundle directory.

    Raises
    ------
    ValidationError
        If ``plan`` is ``None``.

    Notes
    -----
    **This is not a Session checkpoint.** It holds the policy and none of your
    data or history.

    See Also
    --------
    load_rl_bundle : Read one back.
    save_imitation_bundle : The imitation counterpart.
    """
    if plan is None:
        raise ValidationError("No RlPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"plan": plan}, destination / "rl_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT_RL,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_rl_bundle(path: str | Path, *, trusted: bool = False) -> RlPlan:
    """Read a saved RL policy back into memory.

    Checks that both files are present and that the format string matches
    before unpickling.

    Parameters
    ----------
    path:
        The bundle directory written by :func:`save_rl_bundle`.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    RlPlan
        The policy, ready for :func:`~buildml.rl.act.act_rl` and
        :func:`~buildml.rl.evaluate.evaluate_rl`.

    Raises
    ------
    ValidationError
        If either file is missing, if the format string is not
        ``buildml.rl_bundle.v1``, or if the payload does not contain an
        :class:`~buildml.rl.results.RlPlan`.

    Notes
    -----
    **An environment policy needs its environment back.** The bundle carries the
    ``env_id`` but not the environment itself, so reloading on a machine without
    ``buildml[rl]`` succeeds and then fails at the first rollout.

    **Loading executes pickled code.** Only load bundles you produced or trust.

    See Also
    --------
    save_rl_bundle : Write one.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "rl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete RL bundle at {root}. "
            f"Expected meta.json and rl_plan.joblib ({BUNDLE_FORMAT_RL})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT_RL:
        raise ValidationError(
            f"Unsupported RL bundle format {fmt!r}; expected {BUNDLE_FORMAT_RL}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, RlPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "rl_plan.joblib must contain an RlPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, RlPlan):
        raise ValidationError("Loaded plan object is not an RlPlan")
    return plan
