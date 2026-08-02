"""Thin Session facades over buildml.symbolic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from buildml.core.errors import ValidationError
from buildml.data.splits import PartitionName
from buildml.symbolic.checkpoint import load_symbolic_bundle, save_symbolic_bundle
from buildml.symbolic.evaluate import evaluate_neuro_symbolic, evaluate_symbolic
from buildml.symbolic.explain_hooks import (
    eval_result_summary,
    fit_result_summary,
    predict_result_summary,
)
from buildml.symbolic.fit import fit_neuro_symbolic, fit_symbolic
from buildml.symbolic.predict import predict_neuro_symbolic, predict_symbolic
from buildml.symbolic.results import NeuroSymbolicPlan, SymbolicPlan
from buildml.symbolic.rules import Rule
from buildml.symbolic.types import (
    BaseEstimatorName,
    NeuroSymbolicMode,
    SymbolicSource,
    SymbolicTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_symbolic_op(
    session,
    *,
    source: SymbolicSource = "decision_tree",
    task: SymbolicTask | None = None,
    rules: Sequence[Mapping[str, Any] | Rule] | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    max_rules: int = 32,
    default_consequent: Any = None,
    prefer_reduce_components: bool = True,
) -> Any:
    """Compile or induce a symbolic rule base on Session train.

    Notes
    -----
    **Leakage:** Requires a split. Induction / compile statistics use train
    only. Honesty: structured tabular if-then rules — not Prolog/Z3/AGI.
    """
    session.assert_can_fit("train")
    plan, result = fit_symbolic(
        session.dataset,
        session._split_plan,
        source=source,
        task=task,
        rules=rules,
        columns=columns,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_rules=max_rules,
        default_consequent=default_consequent,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._symbolic_plan = plan
    session._symbolic_fit_result = result
    session._symbolic_eval_result = None
    session._symbolic_predict_result = None
    session._record(
        "fit_symbolic",
        {
            "source": source,
            "task": task,
            "n_declared_rules": None if rules is None else len(list(rules)),
            "columns": columns,
            "random_state": random_state,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_rules": max_rules,
            "prefer_reduce_components": prefer_reduce_components,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def evaluate_symbolic_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate the symbolic plan on a holdout partition."""
    plan = getattr(session, "_symbolic_plan", None)
    if plan is None:
        raise ValidationError("No symbolic plan. Call fit_symbolic(...) first.")
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_symbolic(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
    )
    session._symbolic_eval_result = result
    session._record(
        "evaluate_symbolic",
        {"partition": resolved},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def predict_symbolic_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    return_traces: bool = True,
) -> Any:
    """Predict with the symbolic rule base (no update)."""
    plan = getattr(session, "_symbolic_plan", None)
    if plan is None:
        raise ValidationError("No symbolic plan. Call fit_symbolic(...) first.")
    result = predict_symbolic(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        return_traces=return_traces,
    )
    session._symbolic_predict_result = result
    session._record(
        "predict_symbolic",
        {"partition": partition, "return_traces": return_traces},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def fit_neuro_symbolic_op(
    session,
    *,
    mode: NeuroSymbolicMode = "constraint_overlay",
    base_estimator: BaseEstimatorName = "logistic_regression",
    task: SymbolicTask | None = None,
    rules: Sequence[Mapping[str, Any] | Rule] | None = None,
    rule_source: SymbolicSource = "decision_tree",
    columns: list[str] | None = None,
    random_state: int | None = 0,
    soft_strength: float = 0.5,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    max_rules: int = 24,
    prefer_reduce_components: bool = True,
) -> Any:
    """Fit a sklearn + symbolic hybrid on Session train.

    Notes
    -----
    **Leakage:** Requires a split. Base estimator fit and any rule induction
    use train only. This is a real Session-integrated hybrid — not a
    disconnected "fit then apply rules" pair without shared state.
    """
    session.assert_can_fit("train")
    plan, result = fit_neuro_symbolic(
        session.dataset,
        session._split_plan,
        mode=mode,
        base_estimator=base_estimator,
        task=task,
        rules=rules,
        rule_source=rule_source,
        columns=columns,
        random_state=random_state,
        soft_strength=soft_strength,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_rules=max_rules,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=getattr(session, "_reduce_plan", None),
    )
    session._neuro_symbolic_plan = plan
    session._neuro_symbolic_fit_result = result
    session._symbolic_eval_result = None
    session._neuro_symbolic_predict_result = None
    session._record(
        "fit_neuro_symbolic",
        {
            "mode": mode,
            "base_estimator": base_estimator,
            "task": task,
            "rule_source": rule_source,
            "n_declared_rules": None if rules is None else len(list(rules)),
            "columns": columns,
            "random_state": random_state,
            "soft_strength": soft_strength,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_rules": max_rules,
            "prefer_reduce_components": prefer_reduce_components,
        },
        warnings=tuple(result.warnings),
        result_summary=fit_result_summary(result),
    )
    return result


def evaluate_neuro_symbolic_op(
    session,
    *,
    partition: PartitionOrAll = "validation",
) -> Any:
    """Evaluate the neuro-symbolic plan on a holdout partition."""
    plan = getattr(session, "_neuro_symbolic_plan", None)
    if plan is None:
        raise ValidationError(
            "No neuro-symbolic plan. Call fit_neuro_symbolic(...) first."
        )
    resolved: PartitionOrAll = partition
    split = session._split_plan
    if (
        partition == "validation"
        and split is not None
        and not split.validation_indices
    ):
        resolved = "test"
    result = evaluate_neuro_symbolic(
        session.dataset,
        plan,
        session._split_plan,
        partition=resolved,
    )
    session._symbolic_eval_result = result
    session._record(
        "evaluate_neuro_symbolic",
        {"partition": resolved},
        warnings=tuple(result.warnings),
        result_summary=eval_result_summary(result),
    )
    return result


def predict_neuro_symbolic_op(
    session,
    *,
    partition: PartitionOrAll = "test",
    return_traces: bool = True,
) -> Any:
    """Predict with the neuro-symbolic hybrid (no update)."""
    plan = getattr(session, "_neuro_symbolic_plan", None)
    if plan is None:
        raise ValidationError(
            "No neuro-symbolic plan. Call fit_neuro_symbolic(...) first."
        )
    result = predict_neuro_symbolic(
        session.dataset,
        plan,
        session._split_plan,
        partition=partition,
        return_traces=return_traces,
    )
    session._neuro_symbolic_predict_result = result
    session._record(
        "predict_neuro_symbolic",
        {"partition": partition, "return_traces": return_traces},
        warnings=tuple(result.warnings),
        result_summary=predict_result_summary(result),
    )
    return result


def save_symbolic_bundle_op(session, path: str | Path) -> Path:
    """Persist the active symbolic or neuro-symbolic plan."""
    plan: SymbolicPlan | NeuroSymbolicPlan | None = getattr(
        session, "_neuro_symbolic_plan", None
    ) or getattr(session, "_symbolic_plan", None)
    if plan is None:
        raise ValidationError(
            "No symbolic / neuro-symbolic plan. Call fit_symbolic(...) or "
            "fit_neuro_symbolic(...) first."
        )
    fit_result = (
        getattr(session, "_neuro_symbolic_fit_result", None)
        if isinstance(plan, NeuroSymbolicPlan)
        else getattr(session, "_symbolic_fit_result", None)
    )
    out = save_symbolic_bundle(
        path,
        plan,
        fit_result=fit_result,
        eval_result=getattr(session, "_symbolic_eval_result", None),
    )
    session._record(
        "save_symbolic_bundle",
        {"path": str(out)},
        result_summary={"path": str(out), "format": "buildml.symbolic_bundle.v1"},
    )
    return out


def load_symbolic_bundle_op(session, path: str | Path):
    """Load a symbolic bundle into this Session."""
    plan = load_symbolic_bundle(path)
    if isinstance(plan, NeuroSymbolicPlan):
        session._neuro_symbolic_plan = plan
        session._neuro_symbolic_fit_result = None
        session._neuro_symbolic_predict_result = None
        session._symbolic_plan = None
        session._symbolic_fit_result = None
        session._symbolic_predict_result = None
        summary = {
            "path": str(path),
            "kind": "neuro_symbolic",
            "mode": plan.mode,
            "base_estimator_name": plan.base_estimator_name,
        }
    else:
        session._symbolic_plan = plan
        session._symbolic_fit_result = None
        session._symbolic_predict_result = None
        session._neuro_symbolic_plan = None
        session._neuro_symbolic_fit_result = None
        session._neuro_symbolic_predict_result = None
        summary = {
            "path": str(path),
            "kind": "symbolic",
            "source": plan.source,
            "n_rules": plan.n_rules,
        }
    session._symbolic_eval_result = None
    session._record(
        "load_symbolic_bundle",
        {"path": str(path)},
        result_summary=summary,
    )
    return session
