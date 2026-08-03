"""Thin Session facades over buildml.causal."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

from buildml.causal.checkpoint import load_causal_bundle, save_causal_bundle
from buildml.causal.estimate import estimate_causal
from buildml.causal.evaluate import evaluate_causal
from buildml.causal.explain_hooks import (
    assumptions_summary,
    estimate_result_summary,
    eval_result_summary,
    fit_result_summary,
    refute_result_summary,
)
from buildml.causal.fit import fit_causal
from buildml.causal.refute import refute_causal
from buildml.causal.types import (
    CausalAssumptions,
    CausalBackend,
    CausalMethod,
    CausalRefuteKind,
)
from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName

PartitionOrAll = PartitionName | Literal["all"]


def declare_causal_assumptions_op(
    session,
    *,
    treatment: str,
    outcome: str,
    confounders: Sequence[str] | None,
    estimand: str = "ATE",
    identification: str = "backdoor",
    instruments: Sequence[str] | None = None,
    acknowledge_unconfoundedness: bool = False,
    acknowledge_positivity: bool = False,
    allow_empty_confounders: bool = False,
) -> CausalAssumptions:
    """Declare identification assumptions required before causal estimation.

    Captures treatment, outcome, confounders, and explicit acknowledgements
    of unconfoundedness and positivity. Stores a validated
    :class:`~buildml.causal.types.CausalAssumptions` object on the Session for
    downstream fit and estimate APIs.

    Parameters
    ----------
    session:
        Active Session with dataset attached.
    treatment:
        Column name for the treatment or exposure variable.
    outcome:
        Column name for the outcome of interest.
    confounders:
        Confounder column names; pass ``[]`` only with
        ``allow_empty_confounders=True``.
    estimand:
        Target estimand (``ATE`` by default).
    identification:
        Identification strategy (``backdoor`` by default).
    instruments:
        Optional instrument column names for IV-style paths.
    acknowledge_unconfoundedness:
        Explicit acknowledgement that unconfoundedness holds.
    acknowledge_positivity:
        Explicit acknowledgement that positivity/overlap holds.
    allow_empty_confounders:
        When True, permit an empty confounder list after validation.

    Returns
    -------
    CausalAssumptions
        Validated assumptions object stored on the Session.

    Raises
    ------
    ValidationError
        When ``confounders`` is ``None`` or validation fails.

    Notes
    -----
    EDA / association / feature-importance results never satisfy these
    acknowledgements. Estimation APIs refuse to run without a validated
    :class:`CausalAssumptions` object on the Session (or passed explicitly).
    """
    if confounders is None:
        raise ValidationError(
            "declare_causal_assumptions requires confounders=... "
            "(pass [] only with allow_empty_confounders=True). "
            "Causal estimation refuses incomplete assumption objects; "
            "EDA associations are not a substitute."
        )
    assumptions = CausalAssumptions(
        treatment=str(treatment),
        outcome=str(outcome),
        confounders=tuple(str(c) for c in confounders),
        estimand=estimand,  # type: ignore[arg-type]
        identification=identification,  # type: ignore[arg-type]
        instruments=tuple(str(c) for c in (instruments or ())),
        acknowledge_unconfoundedness=bool(acknowledge_unconfoundedness),
        acknowledge_positivity=bool(acknowledge_positivity),
        allow_empty_confounders=bool(allow_empty_confounders),
    )
    assumptions.validate()
    session._causal_assumptions = assumptions
    session._record(
        "declare_causal_assumptions",
        assumptions.to_dict(),
        result_summary=assumptions_summary(assumptions),
    )
    return assumptions


def _resolve_assumptions(
    session,
    assumptions: CausalAssumptions | dict[str, Any] | None,
) -> CausalAssumptions:
    if assumptions is not None:
        resolved = CausalAssumptions.from_mapping(assumptions)
        resolved.validate()
        return resolved
    stored = getattr(session, "_causal_assumptions", None)
    if stored is None:
        raise ValidationError(
            "No CausalAssumptions declared. Call "
            "declare_causal_assumptions(...) first (or pass assumptions=). "
            "Causal estimation refuses to run from EDA alone."
        )
    stored.validate()
    return cast(CausalAssumptions, stored)


def fit_causal_op(
    session,
    *,
    backend: CausalBackend | None = None,
    method: CausalMethod = "aipw",
    assumptions: CausalAssumptions | dict[str, Any] | None = None,
    bootstrap_samples: int = 200,
    random_state: int | None = 0,
    clip_propensity: tuple[float, float] = (0.01, 0.99),
    outcome_model: str = "ridge",
    propensity_model: str = "logistic_regression",
) -> Any:
    """Fit causal models on Session train and estimate ATE.

    Delegates to :func:`buildml.causal.fit.fit_causal`, stores the
    :class:`~buildml.causal.results.CausalPlan` on Session, and records the
    fit. Follow with :func:`estimate_causal_op` or :func:`evaluate_causal_op`.

    Parameters
    ----------
    session:
        Active Session with dataset, split plan, and declared assumptions.
    backend:
        Optional backend override (``native``, ``dowhy``, ``econml``).
    method:
        Estimator method (``aipw`` by default).
    assumptions:
        Optional assumptions override; uses Session-stored assumptions when
        omitted.
    bootstrap_samples:
        Number of bootstrap draws for uncertainty intervals.
    random_state:
        Seed for stochastic nuisance-model steps.
    clip_propensity:
        Min/max propensity clipping bounds for IPW-style methods.
    outcome_model:
        Outcome nuisance model identifier.
    propensity_model:
        Propensity nuisance model identifier.

    Returns
    -------
    CausalFitResult
        Serializable fit summary including ATE point estimate and warnings.

    Notes
    -----
    **Leakage:** Requires a split. Nuisance models fit on train only.
    **Assumptions:** Requires validated CausalAssumptions: refused otherwise.
    Backends: native (T-learner/IPW/AIPW), dowhy, econml when
    ``buildml[causal-industry]`` is installed. Not causal discovery; EDA
    remains associational.
    """
    resolved = _resolve_assumptions(session, assumptions)
    session._causal_assumptions = resolved
    session.assert_can_fit("train")
    plan, result = fit_causal(
        session.dataset,
        session._split_plan,
        resolved,
        backend=backend,
        method=method,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
        clip_propensity=clip_propensity,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
    )
    session._causal_plan = plan
    session._causal_fit_result = result
    session._causal_estimate_result = None
    session._causal_eval_result = None
    session._causal_refute_result = None
    session._record(
        "fit_causal",
        {
            "backend": backend or plan.backend,
            "method": method,
            "bootstrap_samples": bootstrap_samples,
            "random_state": random_state,
            "clip_propensity": list(clip_propensity),
            "outcome_model": outcome_model,
            "propensity_model": propensity_model,
            "assumptions": resolved.to_dict(),
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def estimate_causal_op(
    session,
    *,
    partition: PartitionOrAll = "train",
    bootstrap_samples: int | None = None,
    random_state: int | None = None,
) -> Any:
    """Estimate ATE on a partition using the fitted CausalPlan.

    Delegates to :func:`buildml.causal.estimate.estimate_causal` without
    refitting nuisance models. Useful for re-scoring train or scoring
    holdout partitions with bootstrap overrides.

    Parameters
    ----------
    session:
        Active Session with a causal plan from :func:`fit_causal_op`.
    partition:
        Partition to score (``train``, ``validation``, ``test``, or ``all``).
    bootstrap_samples:
        Optional bootstrap override; uses plan default when omitted.
    random_state:
        Optional seed override for bootstrap resampling.

    Returns
    -------
    CausalEstimateResult
        Partition ATE estimate with optional bootstrap interval.

    Raises
    ------
    ValidationError
        When no causal plan exists on the Session.
    """
    plan = getattr(session, "_causal_plan", None)
    if plan is None:
        raise ValidationError("No causal plan. Call fit_causal(...) first.")
    result = estimate_causal(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    session._causal_estimate_result = result
    session._record(
        "estimate_causal",
        {
            "partition": partition,
            "bootstrap_samples": bootstrap_samples,
            "random_state": random_state,
        },
        warnings=tuple(result.warnings),
        result_summary=estimate_result_summary(result),
    )
    return result


def evaluate_causal_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
    bootstrap_samples: int | None = None,
) -> Any:
    """Evaluate nuisance predictive quality and ATE on a holdout partition.

    Delegates to :func:`buildml.causal.evaluate.evaluate_causal` to report
    outcome/propensity model quality alongside partition-level ATE checks.

    Parameters
    ----------
    session:
        Active Session with a causal plan from :func:`fit_causal_op`.
    partition:
        Holdout partition for evaluation (``validation`` by default).
    bootstrap_samples:
        Optional bootstrap override for partition ATE uncertainty.

    Returns
    -------
    CausalEvalResult
        Nuisance metrics and partition ATE evaluation summary.

    Raises
    ------
    ValidationError
        When no causal plan exists on the Session.
    """
    plan = getattr(session, "_causal_plan", None)
    if plan is None:
        raise ValidationError("No causal plan. Call fit_causal(...) first.")
    result = evaluate_causal(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        bootstrap_samples=bootstrap_samples,
    )
    session._causal_eval_result = result
    session._record(
        "evaluate_causal",
        {"partition": partition, "bootstrap_samples": bootstrap_samples},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def refute_causal_op(
    session,
    *,
    kind: CausalRefuteKind = "placebo_treatment",
    random_state: int | None = 0,
) -> Any:
    """Simple placebo / random-confounder sensitivity disclosure.

    Delegates to :func:`buildml.causal.refute.refute_causal` to stress-test
    the fitted plan with placebo treatments or random confounders.

    Parameters
    ----------
    session:
        Active Session with a causal plan from :func:`fit_causal_op`.
    kind:
        Refutation kind (``placebo_treatment`` by default).
    random_state:
        Seed for stochastic refutation steps.

    Returns
    -------
    CausalRefuteResult
        Refutation outcome and sensitivity disclosures.

    Raises
    ------
    ValidationError
        When no causal plan exists on the Session.
    """
    plan = getattr(session, "_causal_plan", None)
    if plan is None:
        raise ValidationError("No causal plan. Call fit_causal(...) first.")
    result = refute_causal(
        session.dataset,
        plan,
        session._split_plan,
        kind=kind,
        random_state=random_state,
    )
    session._causal_refute_result = result
    session._record(
        "refute_causal",
        {"kind": kind, "random_state": random_state},
        warnings=tuple(result.warnings),
        result_summary=refute_result_summary(result),
    )
    return result


def save_causal_bundle_op(session, path: str | Path) -> Path:
    """Persist the active CausalPlan as ``buildml.causal_bundle.v1``.

    Delegates to :func:`buildml.causal.checkpoint.save_causal_bundle`.
    Reload with :func:`load_causal_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a causal plan from :func:`fit_causal_op`.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no causal plan exists on the Session.
    """
    plan = getattr(session, "_causal_plan", None)
    if plan is None:
        raise ValidationError("No causal plan. Call fit_causal(...) first.")
    out = save_causal_bundle(
        path,
        plan,
        fit_result=getattr(session, "_causal_fit_result", None),
        eval_result=getattr(session, "_causal_eval_result", None),
    )
    session._record(
        "save_causal_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.causal_bundle.v1"},
    )
    return out


def load_causal_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a causal bundle into this Session.

    Delegates to :func:`buildml.causal.checkpoint.load_causal_bundle` and
    clears prior estimate/eval/refute results.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded causal plan.
    path:
        Path to a ``buildml.causal_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with causal plan attached for chaining.
    """
    plan = load_causal_bundle(path, trusted=trusted)
    session._causal_plan = plan
    session._causal_assumptions = plan.assumptions
    session._causal_fit_result = None
    session._causal_estimate_result = None
    session._causal_eval_result = None
    session._causal_refute_result = None
    session._record(
        "load_causal_bundle",
        {"path": str(path)},
        result_summary={
            "path": str(path),
            "method": plan.method,
            "ate": plan.ate,
            "treatment_column": plan.treatment_column,
            "outcome_column": plan.outcome_column,
        },
    )
    return cast("Session", session)