"""Thin Session facades over buildml.symbolic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

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
    IndustrySymbolicMethod,
    NeuroSymbolicBackend,
    NeuroSymbolicMode,
    SymbolicBackend,
    SymbolicSource,
    SymbolicTask,
)

PartitionOrAll = PartitionName | Literal["all"]


def fit_symbolic_op(
    session,
    *,
    backend: SymbolicBackend | None = None,
    source: SymbolicSource = "decision_tree",
    method: IndustrySymbolicMethod | None = None,
    task: SymbolicTask | None = None,
    rules: Sequence[Mapping[str, Any] | Rule] | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    max_rules: int = 32,
    default_consequent: Any = None,
    prefer_reduce_components: bool = True,
    verify_constraints: bool = False,
) -> Any:
    """Compile or induce a symbolic rule base on Session train.

    Delegates to :func:`buildml.symbolic.fit.fit_symbolic`, stores the
    :class:`~buildml.symbolic.results.SymbolicPlan` on Session, and records
    the fit. Follow with :func:`predict_symbolic_op` or
    :func:`evaluate_symbolic_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        Optional symbolic backend override.
    source:
        Rule source (``decision_tree`` induction or declared rules).
    method:
        Optional industry backend method override.
    task:
        Optional task override (classification/regression).
    rules:
        Optional pre-declared rules to compile instead of inducing.
    columns:
        Optional explicit feature column list.
    random_state:
        Seed for stochastic induction steps.
    max_depth:
        Maximum tree depth when inducing from a decision tree.
    min_samples_leaf:
        Minimum leaf size for tree induction.
    max_rules:
        Cap on emitted rules after induction.
    default_consequent:
        Fallback prediction when no rule fires.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    verify_constraints:
        When True, run optional Z3 constraint verification when available.

    Returns
    -------
    SymbolicFitResult
        Serializable fit summary including rule count and disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Induction / compile statistics use train
    only. Honesty: structured tabular if-then rules: not Prolog/Z3/AGI.
    """
    session.assert_can_fit("train")
    plan, result = fit_symbolic(
        session.dataset,
        session._split_plan,
        backend=backend,
        source=source,
        method=method,
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
        verify_constraints=verify_constraints,
    )
    session._symbolic_plan = plan
    session._symbolic_fit_result = result
    session._symbolic_eval_result = None
    session._symbolic_predict_result = None
    session._record(
        "fit_symbolic",
        {
            "backend": backend,
            "source": source,
            "method": method,
            "task": task,
            "n_declared_rules": None if rules is None else len(list(rules)),
            "columns": columns,
            "random_state": random_state,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_rules": max_rules,
            "prefer_reduce_components": prefer_reduce_components,
            "verify_constraints": verify_constraints,
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
    """Evaluate the symbolic plan on a holdout partition.

    Delegates to :func:`buildml.symbolic.evaluate.evaluate_symbolic` using the
    frozen train rule base. Falls back to ``test`` when validation is empty.

    Parameters
    ----------
    session:
        Active Session with a symbolic plan from :func:`fit_symbolic_op`.
    partition:
        Holdout partition for evaluation (``validation`` by default).

    Returns
    -------
    SymbolicEvalResult
        Holdout metrics and rule-coverage disclosures.

    Raises
    ------
    ValidationError
        When no symbolic plan exists on the Session.
    """
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
    """Predict with the symbolic rule base (no update).

    Delegates to :func:`buildml.symbolic.predict.predict_symbolic` without
    modifying the induced or compiled rules.

    Parameters
    ----------
    session:
        Active Session with a symbolic plan from :func:`fit_symbolic_op`.
    partition:
        Partition to predict on (``test`` by default).
    return_traces:
        When True, include fired-rule traces per row.

    Returns
    -------
    SymbolicPredictResult
        Predictions and optional rule traces for the partition.

    Raises
    ------
    ValidationError
        When no symbolic plan exists on the Session.
    """
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
    backend: NeuroSymbolicBackend | None = None,
    mode: NeuroSymbolicMode = "constraint_overlay",
    base_estimator: BaseEstimatorName = "logistic_regression",
    torch_method: str | None = None,
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
    torch_epochs: int = 60,
    device: str = "cpu",
) -> Any:
    """Fit a sklearn + symbolic hybrid on Session train.

    Delegates to :func:`buildml.symbolic.fit.fit_neuro_symbolic`, stores the
    :class:`~buildml.symbolic.results.NeuroSymbolicPlan` on Session, and
    records the fit. Follow with :func:`predict_neuro_symbolic_op` or
    :func:`evaluate_neuro_symbolic_op`.

    Parameters
    ----------
    session:
        Active Session with dataset and split plan attached.
    backend:
        Optional neuro-symbolic backend override.
    mode:
        Hybrid mode (``constraint_overlay`` by default).
    base_estimator:
        Sklearn base estimator identifier.
    torch_method:
        Optional torch method when backend is torch.
    task:
        Optional task override (classification/regression).
    rules:
        Optional pre-declared rules for the symbolic overlay.
    rule_source:
        Rule induction source when ``rules`` is omitted.
    columns:
        Optional explicit feature column list.
    random_state:
        Seed for stochastic base-estimator and induction steps.
    soft_strength:
        Soft constraint strength for overlay modes.
    max_depth:
        Maximum tree depth for rule induction.
    min_samples_leaf:
        Minimum leaf size for rule induction.
    max_rules:
        Cap on induced rules for the overlay.
    prefer_reduce_components:
        Prefer reduced component columns when a reduce plan exists.
    torch_epochs:
        Training epochs for torch hybrid backend.
    device:
        Torch device string.

    Returns
    -------
    NeuroSymbolicFitResult
        Serializable fit summary including hybrid disclosures.

    Notes
    -----
    **Leakage:** Requires a split. Base estimator fit and any rule induction
    use train only. This is a real Session-integrated hybrid: not a
    disconnected "fit then apply rules" pair without shared state.
    """
    session.assert_can_fit("train")
    plan, result = fit_neuro_symbolic(
        session.dataset,
        session._split_plan,
        backend=backend,
        mode=mode,
        base_estimator=base_estimator,
        torch_method=torch_method,
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
        torch_epochs=torch_epochs,
        device=device,
    )
    session._neuro_symbolic_plan = plan
    session._neuro_symbolic_fit_result = result
    session._symbolic_eval_result = None
    session._neuro_symbolic_predict_result = None
    session._record(
        "fit_neuro_symbolic",
        {
            "backend": backend,
            "mode": mode,
            "base_estimator": base_estimator,
            "torch_method": torch_method,
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
            "torch_epochs": torch_epochs,
            "device": device,
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
    """Evaluate the neuro-symbolic plan on a holdout partition.

    Delegates to :func:`buildml.symbolic.evaluate.evaluate_neuro_symbolic`
    using the frozen hybrid plan. Falls back to ``test`` when validation is
    empty.

    Parameters
    ----------
    session:
        Active Session with a neuro-symbolic plan from
        :func:`fit_neuro_symbolic_op`.
    partition:
        Holdout partition for evaluation (``validation`` by default).

    Returns
    -------
    SymbolicEvalResult
        Holdout metrics and hybrid overlay disclosures.

    Raises
    ------
    ValidationError
        When no neuro-symbolic plan exists on the Session.
    """
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
    """Predict with the neuro-symbolic hybrid (no update).

    Delegates to :func:`buildml.symbolic.predict.predict_neuro_symbolic`
    without refitting the base estimator or rules.

    Parameters
    ----------
    session:
        Active Session with a neuro-symbolic plan from
        :func:`fit_neuro_symbolic_op`.
    partition:
        Partition to predict on (``test`` by default).
    return_traces:
        When True, include overlay/rule traces per row.

    Returns
    -------
    NeuroSymbolicPredictResult
        Predictions and optional traces for the partition.

    Raises
    ------
    ValidationError
        When no neuro-symbolic plan exists on the Session.
    """
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
    """Persist the active symbolic or neuro-symbolic plan.

    Delegates to :func:`buildml.symbolic.checkpoint.save_symbolic_bundle`.
    Reload with :func:`load_symbolic_bundle_op`.

    Parameters
    ----------
    session:
        Active Session with a symbolic or neuro-symbolic plan.
    path:
        Destination directory for the bundle (created if missing).

    Returns
    -------
    pathlib.Path
        Resolved bundle directory path.

    Raises
    ------
    ValidationError
        When no symbolic / neuro-symbolic plan exists on the Session.
    """
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


def load_symbolic_bundle_op(session, path: str | Path, *, trusted: bool = False):
    """Load a symbolic bundle into this Session.

    Delegates to :func:`buildml.symbolic.checkpoint.load_symbolic_bundle` and
    clears prior eval/predict results. Restores either a pure symbolic or a
    neuro-symbolic plan based on bundle contents.

    Parameters
    ----------
    session:
        Session instance to populate with the loaded plan.
    path:
        Path to a ``buildml.symbolic_bundle.v1`` directory.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    Session
        ``session`` with symbolic plan attached for chaining.
    """
    plan = load_symbolic_bundle(path, trusted=trusted)
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
            "source": str(plan.source),
            "n_rules": str(plan.n_rules),
        }
    session._symbolic_eval_result = None
    session._record(
        "load_symbolic_bundle",
        {"path": str(path)},
        result_summary=summary,
    )
    return cast("Session", session)


def symbolic_capability_matrix_op() -> dict[str, Any]:
    """Honest capability matrix for symbolic / neuro-symbolic backends.

    Delegates to :func:`buildml.symbolic.catalog.symbolic_capability_matrix`.
    Use before :func:`fit_symbolic_op` or :func:`fit_neuro_symbolic_op` to
    confirm available backends, sources, and methods for the current install.

    Returns
    -------
    dict
        Nested map of backend identifiers to supported sources and methods.
    """
    from buildml.symbolic.catalog import symbolic_capability_matrix

    return symbolic_capability_matrix()
