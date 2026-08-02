"""Thin Session facades over buildml.synthetic (synthetic-data systems)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.synthetic.checkpoint import load_synthetic_bundle, save_synthetic_bundle
from buildml.synthetic.evaluate import evaluate_synthetic
from buildml.synthetic.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    sample_result_summary,
)
from buildml.synthetic.fit import fit_synthesizer
from buildml.synthetic.sample import sample_and_maybe_merge
from buildml.synthetic.types import EvalBackend, EvalMode, MergeMode, SyntheticBackend, SynthesizerMethod
from buildml.synthetic.validation import validate_synthetic


def fit_synthesizer_op(
    session,
    *,
    backend: SyntheticBackend | None = None,
    method: SynthesizerMethod = "gaussian_copula",
    columns: Sequence[str] | None = None,
    random_state: int = 42,
    smooth_sigma: float = 0.0,
    correlation_ridge: float = 1e-3,
    target_column: str | None = None,
    k_neighbors: int = 5,
    sampling_strategy: str | float | dict[str, float] = "auto",
    epochs: int = 300,
    batch_size: int = 500,
):
    """Fit a tabular synthesizer on Session **train** only.

    Notes
    -----
    **Leakage:** Always fits on train. Validation/test are never used to
    estimate schema, marginals, or joints. Distinct from
    :meth:`Session.resample` (class-balance preprocess).

    **Privacy:** Not a differential-privacy product.
    """
    if session._split_plan is None:
        raise ValidationError(
            "A split is required before fit_synthesizer. Call split(...) first."
        )
    plan, result = fit_synthesizer(
        session.dataset,
        session._split_plan,
        backend=backend,
        method=method,
        columns=columns,
        random_state=random_state,
        smooth_sigma=smooth_sigma,
        correlation_ridge=correlation_ridge,
        target_column=target_column,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        epochs=epochs,
        batch_size=batch_size,
    )
    session._synthesizer_plan = plan
    session._synthetic_fit_result = result
    session._synthetic_eval_result = None
    session._synthetic_sample_result = None
    session._record(
        "fit_synthesizer",
        {
            "backend": backend,
            "method": method,
            "columns": None if columns is None else list(columns),
            "random_state": random_state,
            "smooth_sigma": smooth_sigma,
            "correlation_ridge": correlation_ridge,
            "target_column": target_column,
            "k_neighbors": k_neighbors,
            "sampling_strategy": sampling_strategy,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def sample_synthetic_op(
    session,
    *,
    n: int | None = None,
    random_state: int | None = None,
    condition: dict[str, Any] | None = None,
    merge_mode: MergeMode = "none",
    provenance_column: str = "_synthetic",
    validate: bool = False,
):
    """Sample from the frozen synthesizer; optionally extend train with provenance."""
    plan = getattr(session, "_synthesizer_plan", None)
    if plan is None:
        raise ValidationError(
            "No SynthesizerPlan. Call fit_synthesizer(...) first."
        )
    if session._split_plan is None:
        raise ValidationError(
            "A split is required for sample_synthetic (needed for merge integrity)."
        )
    result, new_ds, new_split = sample_and_maybe_merge(
        session.dataset,
        session._split_plan,
        plan,
        n=n,
        random_state=random_state,
        condition=condition,
        merge_mode=merge_mode,
        provenance_column=provenance_column,
    )
    if validate and result.frame is not None:
        validation = validate_synthetic(plan, result.frame)
        result.warnings = tuple(
            list(result.warnings)
            + list(validation.warnings)
            + ([f"validate_synthetic passed={validation.passed}"] if validation.passed else [])
        )
        if not validation.passed:
            result.warnings = tuple(
                list(result.warnings)
                + [f"validate_synthetic failed {validation.n_failed} check(s)."]
            )
    if new_ds is not None and new_split is not None:
        session._dataset = new_ds
        session._split_plan = new_split
        if getattr(session, "_fit_result", None) is not None:
            session._fit_result = None
            result.warnings = tuple(
                list(result.warnings)
                + [
                    "Cleared classical FitResult after extend_train merge "
                    "(train membership changed)."
                ]
            )
    session._synthetic_sample_result = result
    session._record(
        "sample_synthetic",
        {
            "n": n,
            "random_state": random_state,
            "condition": condition,
            "merge_mode": merge_mode,
            "provenance_column": provenance_column,
            "validate": validate,
        },
        warnings=tuple(result.warnings),
        result_summary=sample_result_summary(result),
    )
    return result


def evaluate_synthetic_op(
    session,
    *,
    mode: EvalMode = "fidelity",
    eval_backend: EvalBackend = "auto",
    partition: PartitionName = "test",
    n_synthetic: int | None = None,
    random_state: int = 0,
    estimator: Literal["auto", "logistic", "ridge"] = "auto",
):
    """Evaluate the frozen synthesizer (fidelity or TSTR utility)."""
    plan = getattr(session, "_synthesizer_plan", None)
    if plan is None:
        raise ValidationError(
            "No SynthesizerPlan. Call fit_synthesizer(...) first."
        )
    result = evaluate_synthetic(
        session.dataset,
        session._split_plan,
        plan,
        mode=mode,
        eval_backend=eval_backend,
        partition=str(partition),
        n_synthetic=n_synthetic,
        random_state=random_state,
        estimator=estimator,
    )
    session._synthetic_eval_result = result
    session._record(
        "evaluate_synthetic",
        {
            "mode": mode,
            "eval_backend": eval_backend,
            "partition": partition,
            "n_synthetic": n_synthetic,
            "random_state": random_state,
            "estimator": estimator,
        },
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def synthetic_capability_matrix_op() -> dict[str, Any]:
    from buildml.synthetic.catalog import synthetic_capability_matrix

    return synthetic_capability_matrix()


def save_synthetic_bundle_op(session, path: str | Path) -> Path:
    plan = getattr(session, "_synthesizer_plan", None)
    if plan is None:
        raise ValidationError("No SynthesizerPlan to save.")
    destination = save_synthetic_bundle(
        path,
        plan,
        fit_result=getattr(session, "_synthetic_fit_result", None),
        eval_result=getattr(session, "_synthetic_eval_result", None),
        sample_result=getattr(session, "_synthetic_sample_result", None),
    )
    session._record(
        "save_synthetic_bundle",
        {"path": str(destination)},
        result_summary={
            "path": str(destination),
            "format": "buildml.synthetic_bundle.v1",
        },
    )
    return destination


def load_synthetic_bundle_op(session, path: str | Path):
    plan = load_synthetic_bundle(path)
    session._synthesizer_plan = plan
    session._synthetic_fit_result = None
    session._synthetic_eval_result = None
    session._synthetic_sample_result = None
    session._record(
        "load_synthetic_bundle",
        {"path": str(path)},
        result_summary={"path": str(path), "method": plan.method},
    )
    return session
