"""Fit Session-facing decision policies (thresholds, costs, allocations)."""

from __future__ import annotations

from typing import Any, Sequence

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.diagnostics import DiagnosticReport
from buildml.model.supervised import FitResult
from buildml.optimize.allocate import (
    select_knapsack_with_backend,
    select_lp_allocate_with_backend,
    select_topk,
)
from buildml.optimize.catalog import DecisionBackendName, resolve_backend
from buildml.optimize.features import (
    assert_tuning_partition,
    column_scores,
    model_scores,
    partition_frame,
    require_split,
)
from buildml.optimize.policies import fit_cost_matrix_policy, fit_threshold_policy
from buildml.optimize.results import DecisionFitResult, DecisionPlan
from buildml.optimize.types import (
    AllocationObjective,
    DecisionBackend,
    DecisionConfig,
    DecisionMethod,
    KnapsackSolver,
    ScoreSource,
    TuningPartition,
)


def fit_decision_policy(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    fit_result: FitResult | None,
    *,
    method: DecisionMethod = "threshold",
    backend: DecisionBackend | None = None,
    partition: TuningPartition = "validation",
    allow_test_tuning: bool = False,
    fp_cost: float | None = None,
    fn_cost: float | None = None,
    tp_benefit: float = 0.0,
    tn_benefit: float = 0.0,
    cost_matrix: Sequence[Sequence[float]] | None = None,
    class_labels: list[str] | None = None,
    capacity: int | None = None,
    budget: float | None = None,
    score_source: ScoreSource = "model_proba",
    score_column: str | None = None,
    cost_column: str | None = None,
    value_column: str | None = None,
    id_column: str | None = None,
    knapsack_solver: KnapsackSolver = "dp",
    objective: AllocationObjective = "maximize_score",
    min_score: float | None = None,
    lp_max_fraction: float = 1.0,
) -> tuple[DecisionPlan, DecisionFitResult, DiagnosticReport | None]:
    """Fit a leakage-gated decision policy on train or validation.

    Methods
    -------
    threshold
        Cost-sensitive (or F1) binary operating point via classical
        ``threshold_report`` — same engine as ``Session.tune_threshold``.
    cost_matrix
        Multiclass Bayes action under a user cost matrix.
    topk
        Select top-K candidates by model or column scores under capacity.
    knapsack
        0-1 knapsack-lite (exact DP when costs near-integral, else greedy).
    lp_allocate
        Continuous budget shares via ``scipy.optimize.linprog`` or CVXPY when
        ``backend='cvxpy'`` (``buildml[optimize-industry]``).

    backend
        Solver / scorer backend (see ``decision_capability_matrix()``).
        Defaults to industry when installed (PuLP/OR-Tools knapsack MIP,
        CVXPY LP, XGB cost-sensitive threshold); otherwise native scipy/numpy.

    Honesty: decision helpers for ML scores/costs/allocations — not a general
    operations-research platform or digital twin. Never tunes on Session test
    without ``allow_test_tuning=True``.

    Parameters
    ----------
    dataset:
        Tabular dataset with features and target.
    split_plan:
        Train/validation/test split; required for partition-scoped tuning.
    fit_result:
        Session fit result for model-scored methods; optional for pure
        column-driven allocation.
    method:
        Policy type: ``'threshold'``, ``'cost_matrix'``, ``'topk'``,
        ``'knapsack'``, or ``'lp_allocate'``.
    backend:
        Solver/scorer backend; ``None`` picks an installed default.
    partition:
        Split used to select operating points or allocations.
    allow_test_tuning:
        Dangerous opt-in to tune on the test partition.
    fp_cost, fn_cost, tp_benefit, tn_benefit:
        Binary expected-cost knobs for threshold methods.
    cost_matrix, class_labels:
        Multiclass Bayes cost specification for ``method='cost_matrix'``.
    capacity:
        Top-K selection limit for ``method='topk'``.
    budget:
        Cost budget for knapsack/LP allocation methods.
    score_source:
        ``'model_proba'``, ``'model_decision_function'``, or ``'column'``.
    score_column, cost_column, value_column, id_column:
        Column-driven allocation inputs when not using model scores alone.
    knapsack_solver:
        Native knapsack fallback: ``'dp'`` or ``'greedy'``.
    objective:
        Allocation objective: maximize score/value or minimize cost.
    min_score:
        Optional score floor excluding low-scoring candidates.
    lp_max_fraction:
        Per-item cap on fractional LP allocations in ``(0, 1]``.

    Returns
    -------
    tuple[DecisionPlan, DecisionFitResult, DiagnosticReport | None]
        Frozen reusable policy, fit summary, and optional threshold diagnostic
        report (threshold methods only).

    Raises
    ------
    ValidationError
        When splits, methods, backends, or required knobs are missing/invalid.
    LeakageError
        When tuning on test without ``allow_test_tuning=True``.
    MissingExtraError
        When a requested industry backend extra is not installed.
    """
    split = require_split(split_plan)
    assert_tuning_partition(partition, allow_test_tuning=allow_test_tuning)
    if method not in {"threshold", "cost_matrix", "topk", "knapsack", "lp_allocate"}:
        raise ValidationError(f"Unknown decision method: {method!r}")

    resolved_backend: DecisionBackendName | None = None
    if method == "knapsack":
        resolved_backend = resolve_backend(method="knapsack", backend=backend)  # type: ignore[arg-type]
    elif method == "lp_allocate":
        resolved_backend = resolve_backend(method="lp_allocate", backend=backend)  # type: ignore[arg-type]
    elif method == "threshold":
        if backend is not None:
            resolved_backend = resolve_backend(method="threshold", backend=backend)  # type: ignore[arg-type]
        elif fp_cost is not None or fn_cost is not None:
            resolved_backend = resolve_backend(method="threshold", backend=None)  # type: ignore[arg-type]
        else:
            resolved_backend = "native"
    elif backend is not None and backend != "native":
        raise ValidationError(
            f"backend={backend!r} is not supported for method={method!r}."
        )

    config = DecisionConfig(
        method=method,
        backend=resolved_backend if resolved_backend is not None else backend,
        partition=partition,
        allow_test_tuning=allow_test_tuning,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tp_benefit=tp_benefit,
        tn_benefit=tn_benefit,
        cost_matrix=None if cost_matrix is None else [list(map(float, r)) for r in cost_matrix],
        class_labels=None if class_labels is None else list(class_labels),
        capacity=capacity,
        budget=budget,
        score_source=score_source,
        score_column=score_column,
        cost_column=cost_column,
        value_column=value_column,
        id_column=id_column,
        knapsack_solver=knapsack_solver,
        objective=objective,
        min_score=min_score,
        lp_max_fraction=lp_max_fraction,
    )

    diagnostic: DiagnosticReport | None = None
    if method == "threshold":
        if fit_result is None:
            raise ValidationError("method='threshold' requires Session.fit(...).")
        threshold_backend = resolved_backend or "native"
        if threshold_backend == "xgb":
            from buildml.optimize.adapters.xgb_threshold import fit_xgb_threshold_policy

            plan, metrics, diagnostic = fit_xgb_threshold_policy(
                dataset,
                split,
                fit_result,
                partition=partition,
                allow_test_tuning=allow_test_tuning,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                tp_benefit=tp_benefit,
                tn_benefit=tn_benefit,
            )
        elif threshold_backend == "calibrated":
            from buildml.optimize.adapters.calibrated_threshold import (
                fit_calibrated_threshold_policy,
            )

            plan, metrics, diagnostic = fit_calibrated_threshold_policy(
                dataset,
                split,
                fit_result,
                partition=partition,
                allow_test_tuning=allow_test_tuning,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                tp_benefit=tp_benefit,
                tn_benefit=tn_benefit,
            )
        else:
            plan, metrics, diagnostic = fit_threshold_policy(
                dataset,
                split,
                fit_result,
                partition=partition,
                allow_test_tuning=allow_test_tuning,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                tp_benefit=tp_benefit,
                tn_benefit=tn_benefit,
            )
            if threshold_backend == "native":
                plan.config = {**plan.config, "backend": "native"}
                plan.operating_points = {**plan.operating_points, "backend": "native"}
        plan.config = {**plan.config, **config.to_dict()}
        fit_res = DecisionFitResult(
            method=method,
            backend=str(plan.config.get("backend") or threshold_backend),
            partition=partition,
            n_rows=plan.n_rows_fitted,
            threshold=plan.threshold,
            recommendation_basis=plan.recommendation_basis,
            expected_cost=plan.expected_cost_at_fit,
            allow_test_tuning=allow_test_tuning,
            disclosures=plan.disclosures,
            warnings=plan.warnings,
            metrics=metrics,
        )
        return plan, fit_res, diagnostic

    if method == "cost_matrix":
        if fit_result is None:
            raise ValidationError("method='cost_matrix' requires Session.fit(...).")
        plan, metrics = fit_cost_matrix_policy(
            dataset,
            split,
            fit_result,
            partition=partition,
            allow_test_tuning=allow_test_tuning,
            cost_matrix=cost_matrix,
            class_labels=class_labels,
        )
        plan.config = {**plan.config, **config.to_dict()}
        fit_res = DecisionFitResult(
            method=method,
            partition=partition,
            n_rows=plan.n_rows_fitted,
            expected_cost=plan.expected_cost_at_fit,
            allow_test_tuning=allow_test_tuning,
            disclosures=plan.disclosures,
            warnings=plan.warnings,
            metrics=metrics,
        )
        return plan, fit_res, None

    # Allocation methods
    plan, metrics = _fit_allocation_policy(
        dataset,
        split,
        fit_result,
        method=method,
        partition=partition,
        allow_test_tuning=allow_test_tuning,
        capacity=capacity,
        budget=budget,
        score_source=score_source,
        score_column=score_column,
        cost_column=cost_column,
        value_column=value_column,
        id_column=id_column,
        knapsack_solver=knapsack_solver,
        objective=objective,
        min_score=min_score,
        lp_max_fraction=lp_max_fraction,
        config=config,
        backend=resolved_backend,
    )
    fit_res = DecisionFitResult(
        method=method,
        backend=str(plan.config.get("backend") or resolved_backend or "native"),
        partition=partition,
        n_rows=plan.n_rows_fitted,
        n_selected=plan.n_selected_at_fit,
        selected_value=plan.selected_value_at_fit,
        selected_cost=plan.selected_cost_at_fit,
        capacity=plan.capacity,
        budget=plan.budget,
        allow_test_tuning=allow_test_tuning,
        disclosures=plan.disclosures,
        warnings=plan.warnings,
        metrics=metrics,
    )
    return plan, fit_res, None


def _fit_allocation_policy(
    dataset: Dataset,
    split: SplitPlan,
    fit_result: FitResult | None,
    *,
    method: str,
    partition: str,
    allow_test_tuning: bool,
    capacity: int | None,
    budget: float | None,
    score_source: str,
    score_column: str | None,
    cost_column: str | None,
    value_column: str | None,
    id_column: str | None,
    knapsack_solver: str,
    objective: str,
    min_score: float | None,
    lp_max_fraction: float,
    config: DecisionConfig,
    backend: DecisionBackendName | None = None,
) -> tuple[DecisionPlan, dict[str, float]]:
    frame = partition_frame(dataset, split, partition)
    values, costs, ids, used_score_source = _resolve_allocation_inputs(
        dataset,
        split,
        fit_result,
        frame,
        partition=partition,
        score_source=score_source,
        score_column=score_column,
        cost_column=cost_column,
        value_column=value_column,
        id_column=id_column,
        objective=objective,
    )
    disclosures = [
        f"method={method!r} allocation on partition={partition}.",
        "Constrained selection over ML scores/costs — not a general OR / MIP suite.",
        f"backend={backend or 'native'} solver routing via decision_capability_matrix.",
    ]
    warnings: list[str] = []
    if partition == "test" and allow_test_tuning:
        warnings.append(
            "DANGEROUS OPT-IN: allocation policy fitted on Session test "
            "(allow_test_tuning=True)."
        )

    if method == "topk":
        if capacity is None:
            raise ValidationError("method='topk' requires capacity >= 1.")
        selection = select_topk(
            values, capacity=int(capacity), costs=costs, min_score=min_score, ids=ids
        )
        solver_used = "topk"
        approximate = False
    elif method == "knapsack":
        if budget is None:
            raise ValidationError("method='knapsack' requires budget >= 0.")
        selection = select_knapsack_with_backend(
            values,
            costs,
            budget=float(budget),
            backend=backend,
            solver=knapsack_solver,
            min_score=min_score,
            ids=ids,
        )
        solver_used = str(selection.get("solver_used", knapsack_solver))
        approximate = bool(selection.get("approximate", False))
        if approximate:
            warnings.append(
                "Knapsack used density-greedy approximation (costs not near-integral "
                "or DP state too large)."
            )
        disclosures.append(f"Knapsack solver_used={solver_used}.")
    elif method == "lp_allocate":
        if budget is None:
            raise ValidationError("method='lp_allocate' requires budget >= 0.")
        selection = select_lp_allocate_with_backend(
            values,
            costs,
            budget=float(budget),
            backend=backend,
            max_fraction=float(lp_max_fraction),
            min_score=min_score,
            ids=ids,
        )
        solver_used = str(selection.get("solver_used", "linprog"))
        approximate = False
        if backend == "cvxpy":
            disclosures.append("LP solved via CVXPY convex program.")
        else:
            disclosures.append("LP is continuous fractional allocation (not integer MIP).")
    else:
        raise ValidationError(f"Unknown allocation method: {method!r}")

    plan = DecisionPlan(
        method=method,
        partition_fitted=partition,
        allow_test_tuning=allow_test_tuning,
        capacity=capacity,
        budget=budget,
        score_source=used_score_source,
        score_column=score_column,
        cost_column=cost_column,
        value_column=value_column,
        id_column=id_column,
        knapsack_solver=knapsack_solver if method == "knapsack" else None,
        objective=objective,
        min_score=min_score,
        lp_max_fraction=lp_max_fraction,
        n_rows_fitted=int(len(values)),
        selected_value_at_fit=float(selection["selected_value"]),
        selected_cost_at_fit=float(selection["selected_cost"]),
        n_selected_at_fit=int(selection["n_selected"]),
        feature_columns=(
            tuple(fit_result.feature_columns) if fit_result is not None else ()
        ),
        target_column=None if fit_result is None else fit_result.target_column,
        task=None if fit_result is None else fit_result.task,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
        cost_scale_=float(selection.get("cost_scale", 1.0)),
        operating_points={
            "solver_used": solver_used,
            "approximate": approximate,
            "backend": str(selection.get("backend") or backend or "native"),
        },
    )
    metrics = {
        "n_selected": float(selection["n_selected"]),
        "selected_value": float(selection["selected_value"]),
        "selected_cost": float(selection["selected_cost"]),
        "utilization": (
            float(selection["selected_cost"]) / float(budget)
            if budget not in (None, 0) and method != "topk"
            else (
                float(selection["n_selected"]) / float(capacity)
                if capacity
                else 0.0
            )
        ),
    }
    return plan, metrics


def _resolve_allocation_inputs(
    dataset: Dataset,
    split: SplitPlan,
    fit_result: FitResult | None,
    frame,
    *,
    partition: str,
    score_source: str,
    score_column: str | None,
    cost_column: str | None,
    value_column: str | None,
    id_column: str | None,
    objective: str,
) -> tuple[Any, Any, Any, str]:
    import numpy as np

    use_columns = score_source == "column" or score_column is not None or value_column is not None
    if use_columns and score_source == "column":
        ids, values, costs, _pos = column_scores(
            frame,
            score_column=score_column,
            cost_column=cost_column,
            value_column=value_column,
            id_column=id_column,
        )
        if objective == "minimize_cost":
            values = -costs
        return values, costs, ids, "column"

    if score_column is not None or value_column is not None:
        # Explicit columns take precedence when provided alongside model defaults
        ids, values, costs, _pos = column_scores(
            frame,
            score_column=score_column,
            cost_column=cost_column,
            value_column=value_column,
            id_column=id_column,
        )
        if objective == "minimize_cost":
            values = -costs
        return values, costs, ids, "column"

    if fit_result is None:
        raise ValidationError(
            "Allocation requires a fitted estimator (score_source model_*) "
            "or score_column / value_column."
        )
    if score_source not in {"model_proba", "model_decision_function"}:
        raise ValidationError(f"Unsupported score_source: {score_source!r}")
    index, scores, _proba, _y = model_scores(
        dataset,
        split,
        fit_result,
        partition,
        score_source=score_source,
    )
    if cost_column is not None:
        if cost_column not in frame.columns:
            raise ValidationError(f"cost_column {cost_column!r} not in frame.")
        aligned = frame.loc[index]
        costs = aligned[cost_column].to_numpy(dtype=float)
        ids = (
            aligned[id_column].to_numpy()
            if id_column is not None and id_column in aligned.columns
            else index.to_numpy()
        )
    else:
        costs = np.ones(len(scores), dtype=float)
        ids = index.to_numpy()
    values = scores
    if objective == "minimize_cost":
        values = -costs
    elif objective == "maximize_value" and value_column is not None:
        aligned = frame.loc[index]
        values = aligned[value_column].to_numpy(dtype=float)
    return values, costs, ids, score_source
