"""Propose-confirm-execute flow for AI operator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from buildml.ai.privacy import EgressManifest
from buildml.ai.tools import ToolRegistry
from buildml.ai.types import ConfirmPolicy, ToolCall
from buildml.core.errors import ValidationError

# Allowed estimator name -> (module, class) mapping for LLM tool dispatch.
# Restricts what the LLM can instantiate to a safe, known set.
_ESTIMATOR_REGISTRY: dict[str, tuple[str, str]] = {
    # Classification
    "logisticregression": ("sklearn.linear_model", "LogisticRegression"),
    "randomforestclassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "gradientboostingclassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "svc": ("sklearn.svm", "SVC"),
    "kneighborsclassifier": ("sklearn.neighbors", "KNeighborsClassifier"),
    "decisiontreeclassifier": ("sklearn.tree", "DecisionTreeClassifier"),
    # Regression
    "linearregression": ("sklearn.linear_model", "LinearRegression"),
    "ridge": ("sklearn.linear_model", "Ridge"),
    "lasso": ("sklearn.linear_model", "Lasso"),
    "randomforestregressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "gradientboostingregressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "svr": ("sklearn.svm", "SVR"),
    "kneighborsregressor": ("sklearn.neighbors", "KNeighborsRegressor"),
    "decisiontreeregressor": ("sklearn.tree", "DecisionTreeRegressor"),
}


def _resolve_estimator_map(estimator_names: Any) -> dict[str, Any]:
    """Resolve a list/dict of estimator names into a named estimator map."""
    if isinstance(estimator_names, dict):
        items = [(str(k), v) for k, v in estimator_names.items()]
    elif isinstance(estimator_names, (list, tuple)):
        items = []
        for raw in estimator_names:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                items.append((str(raw[0]), raw[1]))
            else:
                name = str(raw)
                items.append((name.lower(), name))
    else:
        raise ValidationError(
            "estimators must be a list of estimator names or a name→estimator map."
        )
    if len(items) < 2:
        raise ValidationError("Native ensembles require at least two estimators.")
    resolved: dict[str, Any] = {}
    for key, value in items:
        est = _resolve_estimator(value, {})
        base = key
        name = base
        suffix = 2
        while name in resolved:
            name = f"{base}_{suffix}"
            suffix += 1
        resolved[name] = est
    return resolved


def _resolve_estimator(estimator_arg: Any, hyperparameters: dict[str, Any]) -> Any:
    """Resolve an estimator from string name or pass through an instance.

    Parameters
    ----------
    estimator_arg
        Either a string name (e.g. "RandomForestClassifier") or an already-
        instantiated sklearn estimator.
    hyperparameters
        Hyperparameters to pass when instantiating from string.

    Returns
    -------
    Any
        An sklearn-compatible estimator instance.

    Raises
    ------
    ValidationError
        If the estimator name is not in the allowed registry.
    """
    if estimator_arg is None:
        raise ValidationError("fit requires an estimator argument.")

    # If already an instance (has fit method), return as-is
    if hasattr(estimator_arg, "fit") and callable(estimator_arg.fit):
        return estimator_arg

    # Resolve from string name
    if not isinstance(estimator_arg, str):
        raise ValidationError(
            "estimator must be a string name or sklearn instance, "
            f"got {type(estimator_arg).__name__}"
        )

    # Normalize the name for lookup
    normalized = estimator_arg.lower().replace("_", "").replace("-", "")

    if normalized not in _ESTIMATOR_REGISTRY:
        allowed = sorted(set(v[1] for v in _ESTIMATOR_REGISTRY.values()))
        raise ValidationError(
            f"Unknown estimator '{estimator_arg}'. "
            f"Allowed estimators: {', '.join(allowed)}"
        )

    module_name, class_name = _ESTIMATOR_REGISTRY[normalized]

    try:
        import importlib

        module = importlib.import_module(module_name)
        estimator_class = getattr(module, class_name)
        return estimator_class(**hyperparameters)
    except ImportError as e:
        raise ValidationError(f"Failed to import {class_name}: {e}") from e
    except TypeError as e:
        raise ValidationError(
            f"Invalid hyperparameters for {class_name}: {e}"
        ) from e


@dataclass(slots=True)
class ExecutorProposal:
    """A proposed tool execution awaiting confirmation."""

    tool_call: ToolCall
    description: str
    rationale: str
    expected_changes: tuple[str, ...]
    requires_confirmation: bool
    confirm_policy: ConfirmPolicy
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "description": self.description,
            "rationale": self.rationale,
            "expected_changes": list(self.expected_changes),
            "requires_confirmation": self.requires_confirmation,
            "confirm_policy": self.confirm_policy.value,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class ExecutorResult:
    """Result from ai_execute: confirmed tool execution."""

    tool_call: ToolCall
    confirmed: bool
    executed: bool
    result: Any = None
    result_summary: str = ""
    error: str | None = None
    egress_manifest: EgressManifest | None = None
    state_changes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "confirmed": self.confirmed,
            "executed": self.executed,
            "result_summary": self.result_summary,
            "error": self.error,
            "egress_manifest": self.egress_manifest.to_dict() if self.egress_manifest else None,
            "state_changes": list(self.state_changes),
        }


def propose_tool_execution(
    tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry,
) -> ExecutorProposal:
    """Create a proposal for a tool execution.

    Validates the tool is in the registry and determines confirmation requirements.
    """
    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id=f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
    )

    spec = registry.validate_tool_call(call)

    requires_confirm = spec.confirm_policy != ConfirmPolicy.AUTO
    if spec.destructive:
        requires_confirm = True

    warnings: list[str] = []
    if spec.destructive:
        warnings.append("This is a destructive operation that cannot be undone.")
    if not spec.read_only:
        warnings.append("This operation will modify Session state.")

    return ExecutorProposal(
        tool_call=call,
        description=spec.description,
        rationale=f"Tool '{tool_name}' from the allowed registry.",
        expected_changes=_infer_expected_changes(tool_name, arguments),
        requires_confirmation=requires_confirm,
        confirm_policy=spec.confirm_policy,
        warnings=tuple(warnings),
    )


def execute_tool(
    session: Any,
    proposal: ExecutorProposal,
    confirmed: bool,
    registry: ToolRegistry,
) -> ExecutorResult:
    """Execute a proposed tool if confirmed.

    Validates the tool is still in the registry and calls the Session method.
    """
    call = proposal.tool_call

    if proposal.requires_confirmation and not confirmed:
        return ExecutorResult(
            tool_call=call,
            confirmed=False,
            executed=False,
            error="Execution requires confirmation but was not confirmed.",
        )

    spec = registry.validate_tool_call(call)

    if spec.destructive and not confirmed:
        return ExecutorResult(
            tool_call=call,
            confirmed=False,
            executed=False,
            error="Destructive operations always require explicit confirmation.",
        )

    try:
        result, state_changes = _dispatch_tool(session, call, spec)
        result_summary = _summarize_result(result)

        return ExecutorResult(
            tool_call=call,
            confirmed=confirmed,
            executed=True,
            result=result,
            result_summary=result_summary,
            state_changes=state_changes,
        )

    except Exception as e:
        error_msg = _redact_exception_message(str(e))
        return ExecutorResult(
            tool_call=call,
            confirmed=confirmed,
            executed=False,
            error=f"Execution failed: {error_msg}",
        )


def _redact_exception_message(msg: str, max_length: int = 200) -> str:
    """Redact and truncate exception messages before surfacing/storing."""
    import re

    key_patterns = (
        re.compile(r"sk-[a-zA-Z0-9_-]{10,}"),
        re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"'][^\"']+[\"']", re.IGNORECASE),
        re.compile(r"bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    )

    result = msg
    for pattern in key_patterns:
        result = pattern.sub("***REDACTED***", result)

    if len(result) > max_length:
        result = result[:max_length] + "... [truncated]"

    return result


def _dispatch_tool(
    session: Any,
    call: ToolCall,
    spec: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Dispatch a tool call to the appropriate Session method."""
    state_changes: list[str] = []

    if call.tool_name == "set_roles":
        mapping = call.arguments.get("mapping", {})
        if not mapping:
            raise ValidationError("set_roles requires a non-empty mapping argument.")

        before_roles = dict(getattr(session.dataset, "roles", {}) or {})
        session.set_roles(mapping)

        for col, role in mapping.items():
            old_role = before_roles.get(col, "unassigned")
            state_changes.append(f"Column '{col}': {old_role} -> {role}")

        return {"roles_set": mapping}, tuple(state_changes)

    elif call.tool_name == "describe_dataset":
        return session.metadata(), ()

    elif call.tool_name == "explain_operation":
        op = call.arguments.get("operation", "")
        result = session.explain(op)
        return result, ()

    elif call.tool_name == "workflow_status":
        result = session.workflow()
        return result, ()

    elif call.tool_name == "eda_summary":
        result = session.eda()
        return result, ()

    elif call.tool_name == "dry_run_plan":
        plan = call.arguments.get("plan", "")
        result = session.dry_run(plan)
        return result, ()

    elif call.tool_name == "split":
        test_size = call.arguments.get("test_size", 0.2)
        validation_size = call.arguments.get("validation_size")
        stratify = call.arguments.get("stratify", False)
        random_state = call.arguments.get("random_state", 42)
        session.split(
            test_size=test_size,
            validation_size=validation_size,
            stratify=stratify,
            random_state=random_state,
        )
        state_changes.append(f"Created train/test split (test_size={test_size})")
        if validation_size is not None and validation_size > 0:
            state_changes.append(f"Created validation split (validation_size={validation_size})")
        return {"split_created": True}, tuple(state_changes)

    elif call.tool_name == "impute":
        strategy = call.arguments.get("strategy", "median")
        columns = call.arguments.get("columns")
        fill_value = call.arguments.get("fill_value")
        session.impute(
            columns=columns,
            strategy=strategy,
            fill_value=fill_value,
        )
        state_changes.append(f"Imputed missing values (strategy={strategy})")
        return {"imputed": True}, tuple(state_changes)

    elif call.tool_name == "encode":
        method = call.arguments.get("method", "onehot")
        columns = call.arguments.get("columns")
        session.encode(method=method, columns=columns)
        state_changes.append(f"Encoded categorical columns (method={method})")
        return {"encoded": True}, tuple(state_changes)

    elif call.tool_name == "scale":
        method = call.arguments.get("method", "standard")
        columns = call.arguments.get("columns")
        session.scale(method=method, columns=columns)
        state_changes.append(f"Scaled numeric features (method={method})")
        return {"scaled": True}, tuple(state_changes)

    elif call.tool_name == "fit":
        estimator_arg = call.arguments.get("estimator")
        task = call.arguments.get("task", "auto")
        hyperparameters = call.arguments.get("hyperparameters", {})
        estimator = _resolve_estimator(estimator_arg, hyperparameters)
        session.fit(estimator=estimator, task=task)
        state_changes.append(f"Fitted model (estimator={type(estimator).__name__})")
        return {"fitted": True, "estimator": type(estimator).__name__}, tuple(state_changes)

    elif call.tool_name == "evaluate":
        partition = call.arguments.get("partition", "test")
        result = session.evaluate(partition=partition)
        return result, ()

    elif call.tool_name == "walkthrough":
        result = session.walkthrough()
        return result, ()

    elif call.tool_name == "head":
        n = call.arguments.get("n", 5)
        result = session.head(n=n)
        return result, ()

    elif call.tool_name == "drop_columns":
        columns = call.arguments.get("columns", [])
        if not columns:
            raise ValidationError("drop_columns requires a non-empty columns argument.")
        session.drop_columns(columns)
        state_changes.append(f"DROPPED columns: {columns}")
        return {"dropped": columns}, tuple(state_changes)

    elif call.tool_name == "checkpoint_save":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("checkpoint_save requires a path argument.")
        result = session.checkpoint_save(path)
        state_changes.append(f"Saved checkpoint to: {path}")
        return {"checkpoint_path": str(result)}, tuple(state_changes)

    elif call.tool_name == "ai_status":
        result = session.ai_status()
        return result, ()

    elif call.tool_name == "fit_clusters":
        method = call.arguments.get("method", "kmeans")
        kwargs: dict[str, Any] = {
            "method": method,
            "prefer_reduce_components": bool(
                call.arguments.get("prefer_reduce_components", True)
            ),
        }
        if "n_clusters" in call.arguments:
            kwargs["n_clusters"] = call.arguments.get("n_clusters")
        if "eps" in call.arguments:
            kwargs["eps"] = float(call.arguments["eps"])
        if "min_samples" in call.arguments:
            kwargs["min_samples"] = int(call.arguments["min_samples"])
        result = session.fit_clusters(**kwargs)
        state_changes.append(f"Fitted clusters (method={method})")
        return result, tuple(state_changes)

    elif call.tool_name == "assign_clusters":
        partition = call.arguments.get("partition", "test")
        attach = bool(call.arguments.get("attach", False))
        result = session.assign_clusters(partition=partition, attach=attach)
        if attach:
            state_changes.append(f"Attached cluster labels (partition={partition})")
            return result, tuple(state_changes)
        return result, ()

    elif call.tool_name == "evaluate_clusters":
        partition = call.arguments.get("partition", "validation")
        result = session.evaluate_clusters(
            partition=partition,
            external_label_column=call.arguments.get("external_label_column"),
        )
        return result, ()

    elif call.tool_name == "save_unsupervised_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_unsupervised_bundle requires a path argument.")
        result = session.save_unsupervised_bundle(path)
        state_changes.append(f"Saved unsupervised bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_unsupervised_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_unsupervised_bundle requires a path argument.")
        session.load_unsupervised_bundle(path)
        state_changes.append(f"Loaded unsupervised bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_voting":
        estimators = _resolve_estimator_map(call.arguments.get("estimators"))
        kwargs: dict[str, Any] = {
            "estimators": estimators,
            "voting": call.arguments.get("voting", "hard"),
            "task": call.arguments.get("task", "auto"),
        }
        result = session.fit_voting(**kwargs)
        state_changes.append(
            f"Fitted voting ensemble (bases={list(estimators)})."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "fit_stacking":
        estimators = _resolve_estimator_map(call.arguments.get("estimators"))
        kwargs = {
            "estimators": estimators,
            "task": call.arguments.get("task", "auto"),
        }
        if "cv" in call.arguments:
            kwargs["cv"] = int(call.arguments["cv"])
        result = session.fit_stacking(**kwargs)
        state_changes.append(
            f"Fitted stacking ensemble (bases={list(estimators)})."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "fit_blending":
        estimators = _resolve_estimator_map(call.arguments.get("estimators"))
        kwargs = {
            "estimators": estimators,
            "task": call.arguments.get("task", "auto"),
        }
        if "holdout_fraction" in call.arguments:
            kwargs["holdout_fraction"] = float(call.arguments["holdout_fraction"])
        result = session.fit_blending(**kwargs)
        state_changes.append(
            f"Fitted blending ensemble (bases={list(estimators)})."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "evaluate_ensemble":
        partition = call.arguments.get("partition", "test")
        result = session.evaluate_ensemble(partition=partition)
        return result, ()

    elif call.tool_name == "save_ensemble_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_ensemble_bundle requires a path argument.")
        result = session.save_ensemble_bundle(path)
        state_changes.append(f"Saved ensemble bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_ensemble_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_ensemble_bundle requires a path argument.")
        session.load_ensemble_bundle(path)
        state_changes.append(f"Loaded ensemble bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "evolutionary_search":
        param_space = call.arguments.get("param_space")
        if not isinstance(param_space, dict) or not param_space:
            raise ValidationError("evolutionary_search requires a non-empty param_space dict.")
        estimator_arg = call.arguments.get("estimator")
        if not estimator_arg:
            raise ValidationError("evolutionary_search requires an estimator name.")
        estimator = _resolve_estimator(estimator_arg, {})
        evo_kwargs: dict[str, Any] = {
            "param_space": dict(param_space),
            "population_size": int(call.arguments.get("population_size", 8)),
            "n_generations": int(call.arguments.get("n_generations", 3)),
            "cv": int(call.arguments.get("cv", 3)),
            "random_state": 0,
            "refit": bool(call.arguments.get("refit", True)),
            "task": call.arguments.get("task", "auto"),
        }
        recipe_space = call.arguments.get("recipe_space")
        if isinstance(recipe_space, dict) and recipe_space:
            evo_kwargs["recipe_space"] = dict(recipe_space)
        if call.arguments.get("max_evaluations") is not None:
            evo_kwargs["max_evaluations"] = int(call.arguments["max_evaluations"])
        if call.arguments.get("ranking_metric") is not None:
            evo_kwargs["ranking_metric"] = call.arguments["ranking_metric"]
        result = session.evolutionary_search(estimator, **evo_kwargs)
        state_changes.append(
            f"Evolutionary search on {type(estimator).__name__} selected "
            f"params={result.best_params} (score={result.best_score})."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "run_automl":
        kwargs = {
            "method": call.arguments.get("method", "randomized"),
            "selection": call.arguments.get("selection", "cv"),
            "task": call.arguments.get("task", "auto"),
            "include_recipe_search": bool(
                call.arguments.get("include_recipe_search", True)
            ),
            "include_ensembles": bool(call.arguments.get("include_ensembles", False)),
            "n_trials": int(call.arguments.get("n_trials", 12)),
            "cv": 3,
            "random_state": 0,
        }
        families = call.arguments.get("families")
        if families is not None:
            kwargs["families"] = list(families)
        result = session.run_automl(**kwargs)
        state_changes.append(
            f"AutoML selected family={result.best_family} "
            f"recipe={result.best_recipe_strategy}."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "evaluate_automl":
        partition = call.arguments.get("partition", "test")
        result = session.evaluate_automl(partition=partition)
        return result, ()

    elif call.tool_name == "save_automl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_automl_bundle requires a path argument.")
        result = session.save_automl_bundle(path)
        state_changes.append(f"Saved AutoML bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_automl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_automl_bundle requires a path argument.")
        session.load_automl_bundle(path)
        state_changes.append(f"Loaded AutoML bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_forecast":
        kwargs: dict[str, Any] = {
            "method": call.arguments.get("method", "lag_ridge"),
            "horizon": int(call.arguments.get("horizon", 1)),
            "random_state": 0,
        }
        if "lags" in call.arguments and call.arguments["lags"] is not None:
            kwargs["lags"] = list(call.arguments["lags"])
        if "seasonal_period" in call.arguments:
            kwargs["seasonal_period"] = call.arguments.get("seasonal_period")
        if "exog_columns" in call.arguments and call.arguments["exog_columns"] is not None:
            kwargs["exog_columns"] = list(call.arguments["exog_columns"])
        result = session.fit_forecast(**kwargs)
        state_changes.append(
            f"Fitted forecast method={result.method} horizon={result.horizon}."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "generate_forecast":
        kwargs = {
            "origin": call.arguments.get("origin", "train_end"),
        }
        if "horizon" in call.arguments and call.arguments["horizon"] is not None:
            kwargs["horizon"] = int(call.arguments["horizon"])
        result = session.generate_forecast(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_forecast":
        result = session.evaluate_forecast(
            partition=call.arguments.get("partition", "test"),
            strategy=call.arguments.get("strategy", "rolling_one_step"),
        )
        return result, ()

    elif call.tool_name == "save_forecast_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_forecast_bundle requires a path argument.")
        result = session.save_forecast_bundle(path)
        state_changes.append(f"Saved forecast bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_forecast_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_forecast_bundle requires a path argument.")
        session.load_forecast_bundle(path)
        state_changes.append(f"Loaded forecast bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_anomaly":
        kwargs: dict[str, Any] = {
            "method": call.arguments.get("method", "isolation_forest"),
            "mode": call.arguments.get("mode", "unsupervised"),
            "random_state": 0,
        }
        if "contamination" in call.arguments and call.arguments["contamination"] is not None:
            kwargs["contamination"] = float(call.arguments["contamination"])
        if "threshold_policy" in call.arguments and call.arguments["threshold_policy"] is not None:
            kwargs["threshold_policy"] = call.arguments["threshold_policy"]
        if "normal_label_column" in call.arguments:
            kwargs["normal_label_column"] = call.arguments.get("normal_label_column")
        if "prefer_reduce_components" in call.arguments:
            kwargs["prefer_reduce_components"] = bool(
                call.arguments.get("prefer_reduce_components")
            )
        result = session.fit_anomaly(**kwargs)
        state_changes.append(
            f"Fitted anomaly method={result.method} mode={result.mode} "
            f"threshold={result.threshold}."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "score_anomalies":
        partition = call.arguments.get("partition", "test")
        attach = bool(call.arguments.get("attach", False))
        kwargs: dict[str, Any] = {
            "partition": partition,
            "attach": attach,
        }
        if "override_threshold" in call.arguments:
            kwargs["override_threshold"] = call.arguments.get("override_threshold")
        result = session.score_anomalies(**kwargs)
        if attach:
            state_changes.append(f"Attached anomaly score/flag columns (partition={partition})")
            return result, tuple(state_changes)
        return result, ()

    elif call.tool_name == "evaluate_anomaly":
        kwargs = {
            "partition": call.arguments.get("partition", "validation"),
        }
        if "label_column" in call.arguments:
            kwargs["label_column"] = call.arguments.get("label_column")
        if "k" in call.arguments and call.arguments["k"] is not None:
            kwargs["k"] = int(call.arguments["k"])
        result = session.evaluate_anomaly(**kwargs)
        return result, ()

    elif call.tool_name == "save_anomaly_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_anomaly_bundle requires a path argument.")
        result = session.save_anomaly_bundle(path)
        state_changes.append(f"Saved anomaly bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_anomaly_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_anomaly_bundle requires a path argument.")
        session.load_anomaly_bundle(path)
        state_changes.append(f"Loaded anomaly bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_semisupervised":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "base_estimator",
                "threshold",
                "n_neighbors",
                "epochs",
                "text_column",
                "prefer_reduce_components",
            )
            if key in call.arguments
        }
        result = session.fit_semisupervised(**kwargs)
        return (
            f"Fitted semi-supervised backend={result.backend} method={result.method} "
            f"n_labeled={result.n_labeled_train} n_unlabeled={result.n_unlabeled_train}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_semisupervised":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_semisupervised(**kwargs)
        return result, ()

    elif call.tool_name == "save_semisupervised_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_semisupervised_bundle requires a path argument.")
        result = session.save_semisupervised_bundle(path)
        state_changes.append(f"Saved semi-supervised bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_semisupervised_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_semisupervised_bundle requires a path argument.")
        session.load_semisupervised_bundle(path)
        state_changes.append(f"Loaded semi-supervised bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_ssl_pretext":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "method",
                "latent_dim",
                "mask_ratio",
                "prefer_reduce_components",
            )
            if key in call.arguments
        }
        result = session.fit_ssl_pretext(**kwargs)
        return (
            f"Fitted SSL pretext method={result.method} "
            f"latent_dim={result.latent_dim} mae={result.reconstruction_mae}",
            tuple(state_changes),
        )

    elif call.tool_name == "finetune_ssl_head":
        kwargs = {
            key: call.arguments[key]
            for key in ("estimator",)
            if key in call.arguments
        }
        result = session.finetune_ssl_head(**kwargs)
        return (
            f"Fitted SSL head estimator={result.estimator_name} "
            f"n_labeled={result.n_labeled_train}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_ssl":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_ssl(**kwargs)
        return result, ()

    elif call.tool_name == "save_ssl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_ssl_bundle requires a path argument.")
        result = session.save_ssl_bundle(path)
        state_changes.append(f"Saved SSL bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_ssl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_ssl_bundle requires a path argument.")
        session.load_ssl_bundle(path)
        state_changes.append(f"Loaded SSL bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_active_learner":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "strategy",
                "base_estimator",
                "batch_size",
                "label_budget",
                "prefer_reduce_components",
            )
            if key in call.arguments
        }
        result = session.fit_active_learner(**kwargs)
        return (
            f"Fitted active learner backend={result.backend} strategy={result.strategy} "
            f"n_labeled={result.n_labeled_train} "
            f"n_unlabeled_pool={result.n_unlabeled_pool}",
            tuple(state_changes),
        )

    elif call.tool_name == "suggest_query":
        kwargs = {
            key: call.arguments[key]
            for key in ("batch_size", "strategy")
            if key in call.arguments
        }
        result = session.suggest_query(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_active_learning":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_active_learning(**kwargs)
        return result, ()

    elif call.tool_name == "save_active_learning_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_active_learning_bundle requires a path argument.")
        result = session.save_active_learning_bundle(path)
        state_changes.append(f"Saved active-learning bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_active_learning_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_active_learning_bundle requires a path argument.")
        session.load_active_learning_bundle(path)
        state_changes.append(f"Loaded active-learning bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_online":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "estimator",
                "chunk_size",
                "n_init",
                "prefer_reduce_components",
                "allow_refit_fallback",
                "drift_disclose",
            )
            if key in call.arguments
        }
        result = session.fit_online(**kwargs)
        return (
            f"Fitted online learner estimator={result.estimator_name} "
            f"n_init={result.n_init_rows} "
            f"n_remaining={result.n_remaining_train}",
            tuple(state_changes),
        )

    elif call.tool_name == "partial_fit_online":
        kwargs = {
            key: call.arguments[key]
            for key in ("n_rows",)
            if key in call.arguments
        }
        result = session.partial_fit_online(**kwargs)
        return (
            f"Online update mode={result.update_mode} "
            f"n_chunk={result.n_chunk_rows} n_seen={result.n_seen_rows}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_online":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_online(**kwargs)
        return result, ()

    elif call.tool_name == "save_online_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_online_bundle requires a path argument.")
        result = session.save_online_bundle(path)
        state_changes.append(f"Saved online bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_online_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_online_bundle requires a path argument.")
        session.load_online_bundle(path)
        state_changes.append(f"Loaded online bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_multitask":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "task",
                "base_estimator",
                "prefer_reduce_components",
                "epochs",
                "batch_size",
                "learning_rate",
                "device",
            )
            if key in call.arguments
        }
        result = session.fit_multitask(**kwargs)
        return (
            f"Fitted multi-task backend={result.backend} method={result.method} "
            f"task={result.task} n_tasks={result.n_tasks} "
            f"targets={list(result.target_columns)}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_multitask":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_multitask(**kwargs)
        return result, ()

    elif call.tool_name == "save_multitask_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_multitask_bundle requires a path argument.")
        result = session.save_multitask_bundle(path)
        state_changes.append(f"Saved multi-task bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_multitask_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_multitask_bundle requires a path argument.")
        session.load_multitask_bundle(path)
        state_changes.append(f"Loaded multi-task bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_metalearning":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "task_column",
                "k_shot",
                "n_query",
                "n_episodes",
                "base_estimator",
                "prefer_reduce_components",
                "task_holdout_fraction",
                "meta_epochs",
                "inner_lr",
                "inner_steps",
            )
            if key in call.arguments
        }
        result = session.fit_metalearning(**kwargs)
        return (
            f"Fitted meta-learning backend={result.backend} method={result.method} "
            f"n_meta_train_tasks={result.n_meta_train_tasks} "
            f"k_shot={result.k_shot} "
            f"meta_train_accuracy={result.meta_train_accuracy}",
            tuple(state_changes),
        )

    elif call.tool_name == "adapt_to_task":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "task_id",
                "partition",
                "max_support_per_class",
            )
            if key in call.arguments
        }
        result = session.adapt_to_task(**kwargs)
        return (
            f"Adapted to task_id={result.task_id} "
            f"n_support={result.n_support} "
            f"n_classes={result.n_classes_adapted}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_metalearning":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k_shot", "prefer_novel_tasks")
            if key in call.arguments
        }
        result = session.evaluate_metalearning(**kwargs)
        return result, ()

    elif call.tool_name == "save_metalearning_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "save_metalearning_bundle requires a path argument."
            )
        result = session.save_metalearning_bundle(path)
        state_changes.append(f"Saved meta-learning bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_metalearning_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "load_metalearning_bundle requires a path argument."
            )
        session.load_metalearning_bundle(path)
        state_changes.append(f"Loaded meta-learning bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_federated":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "estimator",
                "client_column",
                "n_rounds",
                "local_epochs",
                "client_fraction",
                "mu",
                "prefer_reduce_components",
                "min_client_rows",
            )
            if key in call.arguments
        }
        result = session.fit_federated(**kwargs)
        return (
            f"Fitted federated backend={result.backend} method={result.method} "
            f"estimator={result.estimator_name} "
            f"n_clients={result.n_clients} "
            f"n_rounds={result.n_rounds} "
            f"final_train_metric={result.final_train_metric}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_federated":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "per_client")
            if key in call.arguments
        }
        result = session.evaluate_federated(**kwargs)
        return result, ()

    elif call.tool_name == "predict_federated":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.predict_federated(**kwargs)
        return result, ()

    elif call.tool_name == "save_federated_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "save_federated_bundle requires a path argument."
            )
        result = session.save_federated_bundle(path)
        state_changes.append(f"Saved federated bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_federated_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "load_federated_bundle requires a path argument."
            )
        session.load_federated_bundle(path)
        state_changes.append(f"Loaded federated bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_probabilistic":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "estimator",
                "task",
                "alpha",
                "conformal",
                "conformal_calibration_fraction",
                "interval_method",
                "prefer_reduce_components",
                "n_restarts_optimizer",
            )
            if key in call.arguments
        }
        result = session.fit_probabilistic(**kwargs)
        return (
            f"Fitted probabilistic estimator={result.estimator_name} "
            f"task={result.task} conformal={result.conformal} "
            f"n_fit_rows={result.n_fit_rows} "
            f"n_conformal_calib_rows={result.n_conformal_calib_rows}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_probabilistic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "alpha")
            if key in call.arguments
        }
        result = session.evaluate_probabilistic(**kwargs)
        return result, ()

    elif call.tool_name == "predict_probabilistic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "return_std", "return_proba")
            if key in call.arguments
        }
        result = session.predict_probabilistic(**kwargs)
        return result, ()

    elif call.tool_name == "predict_interval":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "alpha", "method")
            if key in call.arguments
        }
        result = session.predict_interval(**kwargs)
        return result, ()

    elif call.tool_name == "save_probabilistic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "save_probabilistic_bundle requires a path argument."
            )
        result = session.save_probabilistic_bundle(path)
        state_changes.append(f"Saved probabilistic bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_probabilistic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError(
                "load_probabilistic_bundle requires a path argument."
            )
        session.load_probabilistic_bundle(path)
        state_changes.append(f"Loaded probabilistic bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "declare_causal_assumptions":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "treatment",
                "outcome",
                "confounders",
                "estimand",
                "identification",
                "acknowledge_unconfoundedness",
                "acknowledge_positivity",
                "allow_empty_confounders",
            )
            if key in call.arguments
        }
        result = session.declare_causal_assumptions(**kwargs)
        state_changes.append(
            f"Declared CausalAssumptions treatment={result.treatment} "
            f"outcome={result.outcome} estimand={result.estimand}."
        )
        return result.to_dict(), tuple(state_changes)

    elif call.tool_name == "fit_causal":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "bootstrap_samples",
                "random_state",
                "outcome_model",
                "propensity_model",
            )
            if key in call.arguments
        }
        result = session.fit_causal(**kwargs)
        return (
            f"Fitted causal backend={result.backend} method={result.method} "
            f"ate={result.ate} n_train_rows={result.n_train_rows}",
            tuple(state_changes),
        )

    elif call.tool_name == "estimate_causal":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "bootstrap_samples")
            if key in call.arguments
        }
        result = session.estimate_causal(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_causal":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "bootstrap_samples")
            if key in call.arguments
        }
        result = session.evaluate_causal(**kwargs)
        return result, ()

    elif call.tool_name == "refute_causal":
        kwargs = {
            key: call.arguments[key]
            for key in ("kind", "random_state")
            if key in call.arguments
        }
        result = session.refute_causal(**kwargs)
        return result, ()

    elif call.tool_name == "save_causal_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_causal_bundle requires a path argument.")
        result = session.save_causal_bundle(path)
        state_changes.append(f"Saved causal bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_causal_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_causal_bundle requires a path argument.")
        session.load_causal_bundle(path)
        state_changes.append(f"Loaded causal bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "set_graph":
        edges = call.arguments.get("edges")
        if edges is None:
            raise ValidationError("set_graph requires an edges argument.")
        kwargs = {
            key: call.arguments[key]
            for key in ("source_col", "target_col", "node_id_col", "directed")
            if key in call.arguments
        }
        result = session.set_graph(edges, **kwargs)
        state_changes.append(
            f"Attached GraphSpec n_edges={result.n_edges} "
            f"node_id_col={result.node_id_col}."
        )
        return result.to_dict(), tuple(state_changes)

    elif call.tool_name == "fit_graph":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "method",
                "mode",
                "classical_estimator",
                "pyg_model",
                "epochs",
                "hidden_dim",
                "heads",
                "random_state",
                "include_graph_metrics",
            )
            if key in call.arguments
        }
        result = session.fit_graph(**kwargs)
        return (
            f"Fitted graph method={result.method} mode={result.mode} "
            f"train_accuracy={result.train_accuracy} "
            f"n_train_nodes={result.n_train_nodes}",
            tuple(state_changes),
        )

    elif call.tool_name == "predict_graph":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.predict_graph(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_graph":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_graph(**kwargs)
        return result, ()

    elif call.tool_name == "save_graph_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_graph_bundle requires a path argument.")
        result = session.save_graph_bundle(path)
        state_changes.append(f"Saved graph bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_graph_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_graph_bundle requires a path argument.")
        session.load_graph_bundle(path)
        state_changes.append(f"Loaded graph bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "source",
                "task",
                "max_depth",
                "min_samples_leaf",
                "max_rules",
                "random_state",
                "prefer_reduce_components",
            )
            if key in call.arguments
        }
        result = session.fit_symbolic(**kwargs)
        return (
            f"Fitted symbolic source={result.source} task={result.task} "
            f"n_rules={result.n_rules} provenance={result.provenance}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_symbolic(**kwargs)
        return result, ()

    elif call.tool_name == "predict_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "return_traces")
            if key in call.arguments
        }
        result = session.predict_symbolic(**kwargs)
        return result, ()

    elif call.tool_name == "fit_neuro_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "mode",
                "base_estimator",
                "task",
                "rule_source",
                "soft_strength",
                "max_depth",
                "max_rules",
                "random_state",
            )
            if key in call.arguments
        }
        result = session.fit_neuro_symbolic(**kwargs)
        return (
            f"Fitted neuro-symbolic mode={result.mode} "
            f"base={result.base_estimator_name} n_rules={result.n_rules} "
            f"rule_provenance={result.rule_provenance}",
            tuple(state_changes),
        )

    elif call.tool_name == "evaluate_neuro_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_neuro_symbolic(**kwargs)
        return result, ()

    elif call.tool_name == "predict_neuro_symbolic":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "return_traces")
            if key in call.arguments
        }
        result = session.predict_neuro_symbolic(**kwargs)
        return result, ()

    elif call.tool_name == "save_symbolic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_symbolic_bundle requires a path argument.")
        result = session.save_symbolic_bundle(path)
        state_changes.append(f"Saved symbolic bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_symbolic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_symbolic_bundle requires a path argument.")
        session.load_symbolic_bundle(path)
        state_changes.append(f"Loaded symbolic bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_cbr":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "task",
                "metric",
                "reuse",
                "adapt",
                "k",
                "text_columns",
                "text_model_name",
                "standardize",
                "distance_eps",
                "random_state",
                "prefer_reduce_components",
                "torch_epochs",
                "device",
            )
            if key in call.arguments
        }
        result = session.fit_cbr(**kwargs)
        return (
            f"Fitted CBR backend={result.backend} task={result.task} "
            f"metric={result.metric} reuse={result.reuse} k={result.k} "
            f"n_cases={result.n_cases}",
            tuple(state_changes),
        )

    elif call.tool_name == "retrieve_cases":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k", "backend")
            if key in call.arguments
        }
        result = session.retrieve_cases(**kwargs)
        return result, ()

    elif call.tool_name == "predict_cbr":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k", "return_traces", "backend")
            if key in call.arguments
        }
        result = session.predict_cbr(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_cbr":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k")
            if key in call.arguments
        }
        result = session.evaluate_cbr(**kwargs)
        return result, ()

    elif call.tool_name == "retain_cbr":
        disclosure = call.arguments.get("source_disclosure")
        if not disclosure:
            raise ValidationError(
                "retain_cbr requires source_disclosure (and typically "
                "row_indices/labeled_frame via Session API)."
            )
        kwargs = {
            key: call.arguments[key]
            for key in (
                "source_disclosure",
                "solution_column",
                "allow_overlap_with_train",
                "row_indices",
            )
            if key in call.arguments
        }
        result = session.retain_cbr(**kwargs)
        return (
            f"Retained n_added={result.n_added} n_skipped={result.n_skipped} "
            f"n_cases_after={result.n_cases_after}",
            tuple(state_changes),
        )

    elif call.tool_name == "save_cbr_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_cbr_bundle requires a path argument.")
        result = session.save_cbr_bundle(path)
        state_changes.append(f"Saved CBR bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_cbr_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_cbr_bundle requires a path argument.")
        session.load_cbr_bundle(path)
        state_changes.append(f"Loaded CBR bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_imitation":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "task",
                "estimator",
                "action_column",
                "random_state",
                "prefer_reduce_components",
            )
            if key in call.arguments
        }
        result = session.fit_imitation(**kwargs)
        return (
            f"Fitted imitation task={result.task} estimator={result.estimator} "
            f"n_train_rows={result.n_train_rows} train_score={result.train_score}",
            tuple(state_changes),
        )

    elif call.tool_name == "predict_imitation_action":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.predict_imitation_action(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_imitation":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_imitation(**kwargs)
        return result, ()

    elif call.tool_name == "save_imitation_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_imitation_bundle requires a path argument.")
        result = session.save_imitation_bundle(path)
        state_changes.append(f"Saved imitation bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_imitation_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_imitation_bundle requires a path argument.")
        session.load_imitation_bundle(path)
        state_changes.append(f"Loaded imitation bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_rl":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "mode",
                "algorithm",
                "action_column",
                "reward_column",
                "alpha",
                "epsilon",
                "temperature",
                "random_state",
                "prefer_reduce_components",
                "env_id",
                "n_episodes",
                "max_steps",
                "learning_rate",
                "gamma",
            )
            if key in call.arguments
        }
        result = session.fit_rl(**kwargs)
        return (
            f"Fitted RL mode={result.mode} algorithm={result.algorithm} "
            f"n_arms={result.n_arms} env_id={result.env_id}",
            tuple(state_changes),
        )

    elif call.tool_name == "act_rl":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "deterministic", "random_state")
            if key in call.arguments
        }
        result = session.act_rl(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_rl":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "partition",
                "n_episodes",
                "max_steps",
                "random_state",
                "deterministic",
            )
            if key in call.arguments
        }
        result = session.evaluate_rl(**kwargs)
        return result, ()

    elif call.tool_name == "save_rl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_rl_bundle requires a path argument.")
        result = session.save_rl_bundle(path)
        state_changes.append(f"Saved RL bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_rl_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_rl_bundle requires a path argument.")
        session.load_rl_bundle(path)
        state_changes.append(f"Loaded RL bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_tda":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "vectorization",
                "knn",
                "n_bins",
                "n_layers",
                "standardize",
                "head",
                "task",
                "random_state",
                "prefer_reduce_components",
                "max_points_guard",
                "subsample_strategy",
                "mapper",
            )
            if key in call.arguments
        }
        result = session.fit_tda(**kwargs)
        return (
            f"Fitted TDA backend={result.backend} vectorization={result.vectorization} "
            f"feature_dim={result.feature_dim} head={result.head} "
            f"n_train_rows={result.n_train_rows}",
            tuple(state_changes),
        )

    elif call.tool_name == "transform_tda":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.transform_tda(**kwargs)
        return result, ()

    elif call.tool_name == "predict_tda":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.predict_tda(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_tda":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "partition",
                "backend",
                "compare_diagram_distances",
                "diagram_distance_metric",
                "diagram_distance_dim",
            )
            if key in call.arguments
        }
        result = session.evaluate_tda(**kwargs)
        return result, ()

    elif call.tool_name == "save_tda_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_tda_bundle requires a path argument.")
        result = session.save_tda_bundle(path)
        state_changes.append(f"Saved TDA bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_tda_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_tda_bundle requires a path argument.")
        session.load_tda_bundle(path)
        state_changes.append(f"Loaded TDA bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_recommender":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "method",
                "user_column",
                "item_column",
                "rating_column",
                "feedback",
                "n_neighbors",
                "n_factors",
                "min_rating",
                "item_feature_columns",
                "cold_start",
                "random_state",
            )
            if key in call.arguments
        }
        result = session.fit_recommender(**kwargs)
        return (
            f"Fitted recommender method={result.method} "
            f"users={result.n_users} items={result.n_items} "
            f"interactions={result.n_train_interactions}",
            tuple(state_changes),
        )

    elif call.tool_name == "recommend":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k", "exclude_train_items")
            if key in call.arguments
        }
        result = session.recommend(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_recommender":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k")
            if key in call.arguments
        }
        result = session.evaluate_recommender(**kwargs)
        return result, ()

    elif call.tool_name == "save_recommender_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_recommender_bundle requires a path argument.")
        result = session.save_recommender_bundle(path)
        state_changes.append(f"Saved recommender bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_recommender_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_recommender_bundle requires a path argument.")
        session.load_recommender_bundle(path)
        state_changes.append(f"Loaded recommender bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_ranker":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "method",
                "query_column",
                "item_column",
                "relevance_column",
                "feature_columns",
                "pointwise_estimator",
                "pairwise_estimator",
                "max_pairs_per_query",
                "relevance_threshold",
                "alpha",
                "C",
                "random_state",
            )
            if key in call.arguments
        }
        result = session.fit_ranker(**kwargs)
        return (
            f"Fitted ranker method={result.method} "
            f"queries={result.n_train_queries} rows={result.n_train_rows} "
            f"features={result.n_features}",
            tuple(state_changes),
        )

    elif call.tool_name == "rank":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k")
            if key in call.arguments
        }
        result = session.rank(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_ranker":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k")
            if key in call.arguments
        }
        result = session.evaluate_ranker(**kwargs)
        return result, ()

    elif call.tool_name == "save_ranker_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_ranker_bundle requires a path argument.")
        result = session.save_ranker_bundle(path)
        state_changes.append(f"Saved ranker bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_ranker_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_ranker_bundle requires a path argument.")
        session.load_ranker_bundle(path)
        state_changes.append(f"Loaded ranker bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_kg":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "head_column",
                "relation_column",
                "tail_column",
                "embedding_dim",
                "epochs",
                "batch_size",
                "learning_rate",
                "margin",
                "neg_ratio",
                "norm",
                "random_state",
            )
            if key in call.arguments
        }
        result = session.fit_kg(**kwargs)
        return (
            f"Fitted KG backend={result.backend} method={result.method} "
            f"entities={result.n_entities} relations={result.n_relations} "
            f"triples={result.n_train_triples} dim={result.embedding_dim}",
            tuple(state_changes),
        )

    elif call.tool_name == "score_triples":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.score_triples(**kwargs)
        return result, ()

    elif call.tool_name == "predict_links":
        kwargs = {
            key: call.arguments[key]
            for key in ("mode", "heads", "relations", "tails", "k", "filtered")
            if key in call.arguments
        }
        result = session.predict_links(**kwargs)
        return result, ()

    elif call.tool_name == "query_kg":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "mode",
                "entity",
                "source",
                "target",
                "relation",
                "direction",
                "max_hops",
            )
            if key in call.arguments
        }
        result = session.query_kg(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_kg":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition", "k")
            if key in call.arguments
        }
        result = session.evaluate_kg(**kwargs)
        return result, ()

    elif call.tool_name == "save_kg_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_kg_bundle requires a path argument.")
        result = session.save_kg_bundle(path)
        state_changes.append(f"Saved KG bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_kg_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_kg_bundle requires a path argument.")
        session.load_kg_bundle(path)
        state_changes.append(f"Loaded KG bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_decision_policy":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "method",
                "backend",
                "partition",
                "allow_test_tuning",
                "fp_cost",
                "fn_cost",
                "tp_benefit",
                "tn_benefit",
                "cost_matrix",
                "class_labels",
                "capacity",
                "budget",
                "score_source",
                "score_column",
                "cost_column",
                "value_column",
                "id_column",
                "knapsack_solver",
                "objective",
                "min_score",
                "lp_max_fraction",
            )
            if key in call.arguments
        }
        result = session.fit_decision_policy(**kwargs)
        return (
            f"Fitted decision policy method={result.method} "
            f"partition={result.partition} n={result.n_rows} "
            f"threshold={result.threshold}",
            tuple(state_changes),
        )

    elif call.tool_name == "apply_decisions":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.apply_decisions(**kwargs)
        return result, ()

    elif call.tool_name == "evaluate_decisions":
        kwargs = {
            key: call.arguments[key]
            for key in ("partition",)
            if key in call.arguments
        }
        result = session.evaluate_decisions(**kwargs)
        return result, ()

    elif call.tool_name == "save_decision_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_decision_bundle requires a path argument.")
        result = session.save_decision_bundle(path)
        state_changes.append(f"Saved decision bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_decision_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_decision_bundle requires a path argument.")
        session.load_decision_bundle(path)
        state_changes.append(f"Loaded decision bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "fit_synthesizer":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "backend",
                "method",
                "columns",
                "random_state",
                "smooth_sigma",
                "correlation_ridge",
                "target_column",
                "k_neighbors",
                "sampling_strategy",
                "epochs",
                "batch_size",
            )
            if key in call.arguments
        }
        result = session.fit_synthesizer(**kwargs)
        return (
            f"Fitted synthesizer backend={getattr(result, 'backend', 'native')} "
            f"method={result.method} partition={result.partition} "
            f"n={result.n_rows} cols={result.n_columns}",
            tuple(state_changes),
        )

    elif call.tool_name == "sample_synthetic":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "n",
                "random_state",
                "merge_mode",
                "provenance_column",
                "validate",
            )
            if key in call.arguments
        }
        result = session.sample_synthetic(**kwargs)
        if result.merged:
            state_changes.append(
                f"Extended train with {result.n_rows} synthetic rows "
                f"(provenance={result.provenance_column})."
            )
        return result, tuple(state_changes)

    elif call.tool_name == "evaluate_synthetic":
        kwargs = {
            key: call.arguments[key]
            for key in (
                "mode",
                "eval_backend",
                "partition",
                "n_synthetic",
                "random_state",
                "estimator",
            )
            if key in call.arguments
        }
        result = session.evaluate_synthetic(**kwargs)
        return result, ()

    elif call.tool_name == "save_synthetic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("save_synthetic_bundle requires a path argument.")
        result = session.save_synthetic_bundle(path)
        state_changes.append(f"Saved synthetic bundle to: {path}")
        return {"path": str(result)}, tuple(state_changes)

    elif call.tool_name == "load_synthetic_bundle":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("load_synthetic_bundle requires a path argument.")
        session.load_synthetic_bundle(path)
        state_changes.append(f"Loaded synthetic bundle from: {path}")
        return {"loaded": True}, tuple(state_changes)

    elif call.tool_name == "rag_retrieve":
        query = call.arguments.get("query", "")
        if not query:
            raise ValidationError("rag_retrieve requires a non-empty query.")
        result = session.rag_retrieve(
            query,
            k=int(call.arguments.get("k", 5)),
            mode=call.arguments.get("mode"),
        )
        return result, ()

    elif call.tool_name == "rag_generate":
        query = call.arguments.get("query", "")
        if not query:
            raise ValidationError("rag_generate requires a non-empty query.")
        result = session.rag_generate(
            query,
            k=int(call.arguments.get("k", 5)),
        )
        return result, ()

    elif call.tool_name == "rag_ingest_corpus":
        documents = call.arguments.get("documents")
        text_column = call.arguments.get("text_column")
        if documents is None and text_column is None:
            raise ValidationError("rag_ingest_corpus requires documents= or text_column=.")
        session.rag_ingest_corpus(documents, text_column=text_column)
        state_changes.append("Ingested RAG corpus (prior index cleared).")
        return {"rag_corpus_ingested": True}, tuple(state_changes)

    elif call.tool_name == "rag_embed_and_index":
        embedder = call.arguments.get("embedder")
        session.rag_embed_and_index(embedder=embedder)
        state_changes.append("Built RAG index.")
        return {"rag_index_built": True}, tuple(state_changes)

    elif call.tool_name == "make_torch_loaders":
        session.make_torch_loaders(
            batch_size=int(call.arguments.get("batch_size", 32)),
            normalize=bool(call.arguments.get("normalize", True)),
            apply_plans=bool(call.arguments.get("apply_plans", False)),
        )
        state_changes.append("Built Torch DataLoaders.")
        return {"torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "make_text_torch_loaders":
        kwargs: dict[str, Any] = {
            "batch_size": int(call.arguments.get("batch_size", 16)),
        }
        if "text_column" in call.arguments and call.arguments["text_column"] is not None:
            kwargs["text_column"] = call.arguments["text_column"]
        if "max_len" in call.arguments and call.arguments["max_len"] is not None:
            kwargs["max_len"] = int(call.arguments["max_len"])
        session.make_text_torch_loaders(**kwargs)
        state_changes.append("Built text Torch DataLoaders (train-only vocab).")
        return {"text_torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "fit_torch":
        session.fit_torch(
            epochs=int(call.arguments.get("epochs", 5)),
            learning_rate=float(call.arguments.get("learning_rate", 1e-3)),
            device=call.arguments.get("device", "auto"),
        )
        state_changes.append("Fitted Torch module (dl_train_result updated).")
        return {"torch_fitted": True}, tuple(state_changes)

    elif call.tool_name == "evaluate_torch":
        partition = call.arguments.get("partition", "test")
        result = session.evaluate_torch(partition=partition)
        return result, ()

    elif call.tool_name == "cross_validate_torch":
        result = session.cross_validate_torch(
            n_folds=int(call.arguments.get("n_folds", 3)),
            epochs=int(call.arguments.get("epochs", 3)),
        )
        state_changes.append("Completed fold-local Torch CV.")
        return result, tuple(state_changes)

    elif call.tool_name == "make_multimodal_torch_loaders":
        kwargs: dict[str, Any] = {
            "batch_size": int(call.arguments.get("batch_size", 16)),
            "normalize": bool(call.arguments.get("normalize", True)),
            "normalize_images": bool(call.arguments.get("normalize_images", True)),
            "normalize_audio": bool(call.arguments.get("normalize_audio", True)),
        }
        if "text_column" in call.arguments and call.arguments["text_column"] is not None:
            kwargs["text_column"] = call.arguments["text_column"]
        if "image_column" in call.arguments and call.arguments["image_column"] is not None:
            kwargs["image_column"] = call.arguments["image_column"]
        if "audio_column" in call.arguments and call.arguments["audio_column"] is not None:
            kwargs["audio_column"] = call.arguments["audio_column"]
        if call.arguments.get("audio_sample_rate") is not None:
            kwargs["audio_sample_rate"] = int(call.arguments["audio_sample_rate"])
        if call.arguments.get("audio_max_samples") is not None:
            kwargs["audio_max_samples"] = int(call.arguments["audio_max_samples"])
        if call.arguments.get("audio_source_sample_rate") is not None:
            kwargs["audio_source_sample_rate"] = int(
                call.arguments["audio_source_sample_rate"]
            )
        session.make_multimodal_torch_loaders(**kwargs)
        state_changes.append(
            "Built multimodal Torch DataLoaders "
            "(train-only vocab/normalize/image/audio stats)."
        )
        return {"multimodal_torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "make_image_multimodal_torch_loaders":
        image_column = call.arguments.get("image_column")
        if not image_column:
            raise ValidationError(
                "make_image_multimodal_torch_loaders requires image_column."
            )
        img_kwargs: dict[str, Any] = {
            "image_column": str(image_column),
            "batch_size": int(call.arguments.get("batch_size", 16)),
            "normalize_images": bool(call.arguments.get("normalize_images", True)),
            "normalize_audio": bool(call.arguments.get("normalize_audio", True)),
        }
        if "text_column" in call.arguments and call.arguments["text_column"] is not None:
            img_kwargs["text_column"] = call.arguments["text_column"]
        if "audio_column" in call.arguments and call.arguments["audio_column"] is not None:
            img_kwargs["audio_column"] = call.arguments["audio_column"]
        if call.arguments.get("audio_sample_rate") is not None:
            img_kwargs["audio_sample_rate"] = int(call.arguments["audio_sample_rate"])
        if call.arguments.get("audio_max_samples") is not None:
            img_kwargs["audio_max_samples"] = int(call.arguments["audio_max_samples"])
        session.make_image_multimodal_torch_loaders(**img_kwargs)
        state_changes.append(
            "Built image multimodal Torch DataLoaders (train-only image/channel stats)."
        )
        return {"image_multimodal_torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "make_audio_multimodal_torch_loaders":
        audio_column = call.arguments.get("audio_column")
        if not audio_column:
            raise ValidationError(
                "make_audio_multimodal_torch_loaders requires audio_column."
            )
        aud_kwargs: dict[str, Any] = {
            "audio_column": str(audio_column),
            "batch_size": int(call.arguments.get("batch_size", 16)),
            "normalize_audio": bool(call.arguments.get("normalize_audio", True)),
        }
        if "text_column" in call.arguments and call.arguments["text_column"] is not None:
            aud_kwargs["text_column"] = call.arguments["text_column"]
        if "image_column" in call.arguments and call.arguments["image_column"] is not None:
            aud_kwargs["image_column"] = call.arguments["image_column"]
        if call.arguments.get("audio_sample_rate") is not None:
            aud_kwargs["audio_sample_rate"] = int(call.arguments["audio_sample_rate"])
        if call.arguments.get("audio_max_samples") is not None:
            aud_kwargs["audio_max_samples"] = int(call.arguments["audio_max_samples"])
        if call.arguments.get("audio_source_sample_rate") is not None:
            aud_kwargs["audio_source_sample_rate"] = int(
                call.arguments["audio_source_sample_rate"]
            )
        session.make_audio_multimodal_torch_loaders(**aud_kwargs)
        state_changes.append(
            "Built audio multimodal Torch DataLoaders (train-only audio amplitude stats)."
        )
        return {"audio_multimodal_torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "search_torch":
        search_kwargs: dict[str, Any] = {
            "n_folds": int(call.arguments.get("n_folds", 3)),
            "epochs": int(call.arguments.get("epochs", 2)),
            "n_iter": int(call.arguments.get("n_iter", 5)),
        }
        if call.arguments.get("param_grid") is not None:
            search_kwargs["param_grid"] = call.arguments["param_grid"]
        if call.arguments.get("param_distributions") is not None:
            search_kwargs["param_distributions"] = call.arguments["param_distributions"]
        result = session.search_torch(**search_kwargs)
        state_changes.append("Completed inner-fold Torch hyperparameter search.")
        return result, tuple(state_changes)

    elif call.tool_name == "nested_cv_torch":
        nested_kwargs: dict[str, Any] = {
            "outer_cv": int(call.arguments.get("outer_cv", 3)),
            "inner_cv": int(call.arguments.get("inner_cv", 2)),
            "epochs": int(call.arguments.get("epochs", 2)),
            "n_iter": int(call.arguments.get("n_iter", 5)),
        }
        if call.arguments.get("param_grid") is not None:
            nested_kwargs["param_grid"] = call.arguments["param_grid"]
        if call.arguments.get("param_distributions") is not None:
            nested_kwargs["param_distributions"] = call.arguments["param_distributions"]
        result = session.nested_cv_torch(**nested_kwargs)
        state_changes.append("Completed nested Torch CV (outer after inner search).")
        return result, tuple(state_changes)

    elif call.tool_name == "export_torch":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("export_torch requires a path argument.")
        result = session.export_torch(
            path,
            format=call.arguments.get("format", "torchscript"),
        )
        state_changes.append(f"Exported Torch trainer to {path}.")
        return result, tuple(state_changes)

    elif call.tool_name == "make_speech_torch_loaders":
        speech_kwargs: dict[str, Any] = {
            "batch_size": int(call.arguments.get("batch_size", 8)),
            "normalize_audio": bool(call.arguments.get("normalize_audio", True)),
        }
        if call.arguments.get("audio_column") is not None:
            speech_kwargs["audio_column"] = str(call.arguments["audio_column"])
        if call.arguments.get("sample_rate") is not None:
            speech_kwargs["sample_rate"] = int(call.arguments["sample_rate"])
        if call.arguments.get("max_samples") is not None:
            speech_kwargs["max_samples"] = int(call.arguments["max_samples"])
        session.make_speech_torch_loaders(**speech_kwargs)
        state_changes.append(
            "Built speech classification Torch DataLoaders (finetune-lite)."
        )
        return {"speech_torch_loaders_built": True}, tuple(state_changes)

    elif call.tool_name == "fit_speech_torch":
        fit_speech_kwargs: dict[str, Any] = {
            "epochs": int(call.arguments.get("epochs", 5)),
            "freeze_encoder": bool(call.arguments.get("freeze_encoder", False)),
        }
        if call.arguments.get("audio_column") is not None:
            fit_speech_kwargs["audio_column"] = str(call.arguments["audio_column"])
        session.fit_speech_torch(**fit_speech_kwargs)
        state_changes.append("Fine-tuned tiny speech classifier (finetune-lite).")
        return {"speech_torch_fitted": True}, tuple(state_changes)

    elif call.tool_name == "transcribe_speech":
        audio_column = call.arguments.get("audio_column")
        if not audio_column:
            raise ValidationError("transcribe_speech requires audio_column.")
        result = session.transcribe_speech(
            audio_column=str(audio_column),
            backend=call.arguments.get("backend", "stub"),
            model_id=call.arguments.get("model_id"),
            partition=call.arguments.get("partition", "all"),
        )
        state_changes.append("Transcribed speech audio column (ASR integration path).")
        return result, tuple(state_changes)

    elif call.tool_name == "load_pretrained_backbone":
        modality = call.arguments.get("modality")
        if not modality:
            raise ValidationError("load_pretrained_backbone requires modality.")
        bb_kwargs: dict[str, Any] = {
            "weights": call.arguments.get("weights", "mock"),
            "freeze": bool(call.arguments.get("freeze", True)),
            "seed": int(call.arguments.get("seed", 0)),
        }
        if call.arguments.get("architecture") is not None:
            bb_kwargs["architecture"] = str(call.arguments["architecture"])
        if call.arguments.get("model_id") is not None:
            bb_kwargs["model_id"] = str(call.arguments["model_id"])
        result = session.load_pretrained_backbone(str(modality), **bb_kwargs)
        state_changes.append(f"Loaded pretrained {modality} backbone hook.")
        return result, tuple(state_changes)

    elif call.tool_name == "pack_torchserve":
        output_dir = call.arguments.get("output_dir")
        if not output_dir:
            raise ValidationError("pack_torchserve requires output_dir.")
        pack_kwargs: dict[str, Any] = {}
        if call.arguments.get("torchscript_path") is not None:
            pack_kwargs["torchscript_path"] = str(call.arguments["torchscript_path"])
        if call.arguments.get("model_name") is not None:
            pack_kwargs["model_name"] = str(call.arguments["model_name"])
        result = session.pack_torchserve(str(output_dir), **pack_kwargs)
        state_changes.append(f"Packed TorchServe directory at {output_dir}.")
        return result, tuple(state_changes)

    elif call.tool_name == "prepare_tensorrt_export":
        output_dir = call.arguments.get("output_dir")
        if not output_dir:
            raise ValidationError("prepare_tensorrt_export requires output_dir.")
        trt_kwargs: dict[str, Any] = {
            "fp16": bool(call.arguments.get("fp16", True)),
        }
        if call.arguments.get("onnx_path") is not None:
            trt_kwargs["onnx_path"] = str(call.arguments["onnx_path"])
        if call.arguments.get("engine_name") is not None:
            trt_kwargs["engine_name"] = str(call.arguments["engine_name"])
        result = session.prepare_tensorrt_export(str(output_dir), **trt_kwargs)
        state_changes.append(f"Wrote TensorRT plan under {output_dir}.")
        return result, tuple(state_changes)

    elif call.tool_name == "emit_k8s_ddp_job":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("emit_k8s_ddp_job requires path.")
        k8s_kwargs: dict[str, Any] = {
            "nnodes": int(call.arguments.get("nnodes", 2)),
            "nproc_per_node": int(call.arguments.get("nproc_per_node", 1)),
        }
        if call.arguments.get("script_path") is not None:
            k8s_kwargs["script_path"] = str(call.arguments["script_path"])
        result = session.emit_k8s_ddp_job(str(path), **k8s_kwargs)
        state_changes.append(f"Emitted K8s torchrun Job YAML at {path}.")
        return result, tuple(state_changes)

    elif call.tool_name == "domain_adapt_speech_torch":
        adapt_kwargs: dict[str, Any] = {
            "epochs": int(call.arguments.get("epochs", 5)),
            "freeze_encoder": bool(call.arguments.get("freeze_encoder", True)),
            "batch_size": int(call.arguments.get("batch_size", 8)),
        }
        if call.arguments.get("audio_column") is not None:
            adapt_kwargs["audio_column"] = str(call.arguments["audio_column"])
        result = session.domain_adapt_speech_torch(**adapt_kwargs)
        state_changes.append(
            "Domain-adapted speech classifier (finetune-lite; not FM pretrain)."
        )
        return result, tuple(state_changes)

    elif call.tool_name == "attach_backbone_head":
        n_classes = call.arguments.get("n_classes")
        if n_classes is None:
            raise ValidationError("attach_backbone_head requires n_classes.")
        head_kwargs: dict[str, Any] = {}
        if "freeze_backbone" in call.arguments:
            head_kwargs["freeze_backbone"] = call.arguments.get("freeze_backbone")
        result = session.attach_backbone_head(int(n_classes), **head_kwargs)
        state_changes.append("Attached classification head to pretrained backbone.")
        return result, tuple(state_changes)

    elif call.tool_name == "evaluate_asr":
        references = call.arguments.get("references")
        if references is None:
            raise ValidationError("evaluate_asr requires references.")
        asr_kwargs: dict[str, Any] = {
            "references": list(references),
            "lowercase": bool(call.arguments.get("lowercase", True)),
        }
        if call.arguments.get("hypotheses") is not None:
            asr_kwargs["hypotheses"] = list(call.arguments["hypotheses"])
        result = session.evaluate_asr(**asr_kwargs)
        state_changes.append("Scored ASR hypotheses (WER/CER).")
        return result, tuple(state_changes)

    elif call.tool_name == "emit_k8s_serve_deployment":
        path = call.arguments.get("path")
        if not path:
            raise ValidationError("emit_k8s_serve_deployment requires path.")
        serve_k8s_kwargs: dict[str, Any] = {}
        if call.arguments.get("name") is not None:
            serve_k8s_kwargs["name"] = str(call.arguments["name"])
        if call.arguments.get("replicas") is not None:
            serve_k8s_kwargs["replicas"] = int(call.arguments["replicas"])
        if call.arguments.get("port") is not None:
            serve_k8s_kwargs["port"] = int(call.arguments["port"])
        result = session.emit_k8s_serve_deployment(str(path), **serve_k8s_kwargs)
        state_changes.append(f"Emitted K8s serve Deployment YAML at {path}.")
        return result, tuple(state_changes)

    else:
        raise ValidationError(f"No dispatch handler for tool: {call.tool_name}")


def _infer_expected_changes(tool_name: str, arguments: dict[str, Any]) -> tuple[str, ...]:
    """Infer expected state changes from a tool call."""
    changes: list[str] = []

    if tool_name == "set_roles":
        mapping = arguments.get("mapping", {})
        for col, role in mapping.items():
            changes.append(f"Column '{col}' will be assigned role '{role}'.")

    elif tool_name in (
        "describe_dataset",
        "explain_operation",
        "workflow_status",
        "eda_summary",
        "dry_run_plan",
        "evaluate",
        "evaluate_clusters",
        "evaluate_anomaly",
        "evaluate_semisupervised",
        "evaluate_ssl",
        "evaluate_active_learning",
        "evaluate_online",
        "evaluate_multitask",
        "evaluate_metalearning",
        "evaluate_federated",
        "predict_federated",
        "suggest_query",
        "walkthrough",
        "head",
        "ai_status",
        "rag_retrieve",
        "rag_generate",
        "evaluate_torch",
        "evaluate_asr",
    ):
        changes.append("No state changes (read-only operation).")

    elif tool_name == "assign_clusters":
        partition = arguments.get("partition", "test")
        if arguments.get("attach"):
            changes.append(
                f"Will assign cluster labels and attach them to the frame (partition={partition})."
            )
        else:
            changes.append("No state changes (read-only assign; labels returned only).")

    elif tool_name == "split":
        test_size = arguments.get("test_size", 0.2)
        changes.append(f"Will create train/test split with test_size={test_size}.")

    elif tool_name == "impute":
        changes.append("Will impute missing values in numeric and categorical columns.")

    elif tool_name == "encode":
        method = arguments.get("method", "onehot")
        changes.append(f"Will encode categorical columns using {method} encoding.")

    elif tool_name == "scale":
        method = arguments.get("method", "standard")
        changes.append(f"Will scale numeric features using {method} scaling.")

    elif tool_name == "fit":
        estimator = arguments.get("estimator", "auto")
        changes.append(f"Will fit model with estimator={estimator}.")

    elif tool_name == "evolutionary_search":
        estimator = arguments.get("estimator", "?")
        changes.append(
            f"Will run evolutionary GA HPO on estimator={estimator} "
            "(train-fold CV only; may refit winner)."
        )

    elif tool_name == "fit_clusters":
        method = arguments.get("method", "kmeans")
        changes.append(f"Will fit unsupervised clusters with method={method}.")

    elif tool_name == "save_unsupervised_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save unsupervised bundle to {path}.")

    elif tool_name == "load_unsupervised_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load unsupervised bundle from {path}.")

    elif tool_name == "fit_voting":
        estimators = arguments.get("estimators", [])
        changes.append(f"Will fit a voting ensemble with bases={estimators}.")

    elif tool_name == "fit_stacking":
        estimators = arguments.get("estimators", [])
        changes.append(f"Will fit a stacking ensemble with bases={estimators}.")

    elif tool_name == "fit_blending":
        estimators = arguments.get("estimators", [])
        changes.append(f"Will fit a blending ensemble with bases={estimators}.")

    elif tool_name == "save_ensemble_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save ensemble bundle to {path}.")

    elif tool_name == "load_ensemble_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load ensemble bundle from {path}.")

    elif tool_name == "fit_forecast":
        method = arguments.get("method", "lag_ridge")
        changes.append(f"Will fit classical forecast with method={method}.")

    elif tool_name == "generate_forecast":
        changes.append("No Session mutation beyond history (forecast values returned).")

    elif tool_name == "evaluate_forecast":
        changes.append("No Session mutation beyond history (metrics returned).")

    elif tool_name == "save_forecast_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save forecast bundle to {path}.")

    elif tool_name == "load_forecast_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load forecast bundle from {path}.")

    elif tool_name == "fit_anomaly":
        method = arguments.get("method", "isolation_forest")
        mode = arguments.get("mode", "unsupervised")
        changes.append(f"Will fit anomaly detector method={method} mode={mode}.")

    elif tool_name == "score_anomalies":
        changes.append("No Session mutation beyond history (scores/flags returned).")

    elif tool_name == "evaluate_anomaly":
        changes.append("No Session mutation beyond history (metrics returned).")

    elif tool_name == "save_anomaly_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save anomaly bundle to {path}.")

    elif tool_name == "load_anomaly_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load anomaly bundle from {path}.")

    elif tool_name == "fit_semisupervised":
        backend = arguments.get("backend", "sklearn")
        method = arguments.get("method", "label_propagation")
        changes.append(
            f"Will fit semi-supervised backend={backend} method={method} on train only."
        )

    elif tool_name == "save_semisupervised_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save semi-supervised bundle to {path}.")

    elif tool_name == "load_semisupervised_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load semi-supervised bundle from {path}.")

    elif tool_name == "fit_ssl_pretext":
        method = arguments.get("method", "masked_tabular")
        changes.append(f"Will fit SSL pretext method={method} on train only.")

    elif tool_name == "finetune_ssl_head":
        estimator = arguments.get("estimator", "logistic_regression")
        changes.append(f"Will fit SSL head estimator={estimator} on labeled train.")

    elif tool_name == "save_ssl_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save SSL bundle to {path}.")

    elif tool_name == "load_ssl_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load SSL bundle from {path}.")

    elif tool_name == "fit_active_learner":
        strategy = arguments.get("strategy", "margin")
        changes.append(
            f"Will fit active learner strategy={strategy} on labeled train only."
        )

    elif tool_name == "save_active_learning_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save active-learning bundle to {path}.")

    elif tool_name == "load_active_learning_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load active-learning bundle from {path}.")

    elif tool_name == "fit_online":
        estimator = arguments.get("estimator", "sgd_classifier")
        changes.append(
            f"Will warm-start online learner estimator={estimator} on a train chunk."
        )

    elif tool_name == "partial_fit_online":
        n_rows = arguments.get("n_rows")
        changes.append(
            "Will apply partial_fit_online on the next train chunk"
            + (f" (n_rows={n_rows})" if n_rows is not None else "")
            + "."
        )

    elif tool_name == "save_online_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save online bundle to {path}.")

    elif tool_name == "load_online_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load online bundle from {path}.")

    elif tool_name == "fit_multitask":
        backend = arguments.get("backend")
        method = arguments.get("method", "multi_output")
        changes.append(
            f"Will fit multi-task learner backend={backend} method={method} "
            "on train targets."
        )

    elif tool_name == "save_multitask_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save multi-task bundle to {path}.")

    elif tool_name == "load_multitask_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load multi-task bundle from {path}.")

    elif tool_name == "fit_metalearning":
        method = arguments.get("method", "prototypical")
        changes.append(
            f"Will meta-train few-shot learner method={method} on train tasks."
        )

    elif tool_name == "adapt_to_task":
        task_id = arguments.get("task_id", "")
        changes.append(f"Will adapt meta-learner to task_id={task_id}.")

    elif tool_name == "save_metalearning_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save meta-learning bundle to {path}.")

    elif tool_name == "load_metalearning_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load meta-learning bundle from {path}.")

    elif tool_name == "fit_federated":
        backend = arguments.get("backend", "native")
        method = arguments.get("method", "fedavg")
        estimator = arguments.get("estimator", "sgd_classifier")
        changes.append(
            f"Will simulate federated backend={backend} method={method} "
            f"estimator={estimator} on train clients (local FL simulation)."
        )

    elif tool_name == "save_federated_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save federated bundle to {path}.")

    elif tool_name == "load_federated_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load federated bundle from {path}.")

    elif tool_name == "fit_probabilistic":
        estimator = arguments.get("estimator", "bayesian_ridge")
        conformal = arguments.get("conformal", True)
        changes.append(
            f"Will fit probabilistic estimator={estimator} "
            f"conformal={conformal} (train-only; not PyMC/Stan)."
        )

    elif tool_name == "save_probabilistic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save probabilistic bundle to {path}.")

    elif tool_name == "load_probabilistic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load probabilistic bundle from {path}.")

    elif tool_name == "declare_causal_assumptions":
        changes.append(
            "Will declare CausalAssumptions (required before fit_causal; "
            "EDA is not a substitute)."
        )

    elif tool_name == "fit_causal":
        method = arguments.get("method", "aipw")
        changes.append(
            f"Will fit causal method={method} under declared assumptions "
            "(train-only nuisances; not DoWhy/EconML)."
        )

    elif tool_name == "save_causal_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save causal bundle to {path}.")

    elif tool_name == "load_causal_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load causal bundle from {path}.")

    elif tool_name == "set_graph":
        changes.append(
            "Will attach GraphSpec edge list (Session rows = nodes; "
            "not a Neo4j/KG product)."
        )

    elif tool_name == "fit_graph":
        method = arguments.get("method", "classical")
        mode = arguments.get("mode", "inductive")
        changes.append(
            f"Will fit graph method={method} mode={mode} "
            "(train labels only; classical needs buildml[graph], "
            "gcn needs buildml[torch], pyg needs buildml[graph-pyg])."
        )

    elif tool_name == "save_graph_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save graph bundle to {path}.")

    elif tool_name == "load_graph_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load graph bundle from {path}.")

    elif tool_name == "fit_symbolic":
        source = arguments.get("source", "decision_tree")
        changes.append(
            f"Will fit symbolic source={source} on train "
            "(tabular rules; not Prolog/Z3/AGI)."
        )

    elif tool_name == "fit_neuro_symbolic":
        mode = arguments.get("mode", "constraint_overlay")
        changes.append(
            f"Will fit neuro-symbolic mode={mode} "
            "(sklearn + rules hybrid; train-only)."
        )

    elif tool_name == "save_symbolic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save symbolic bundle to {path}.")

    elif tool_name == "load_symbolic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load symbolic bundle from {path}.")

    elif tool_name == "fit_cbr":
        metric = arguments.get("metric", "euclidean")
        changes.append(
            f"Will fit CBR metric={metric} on train "
            "(tabular case memory; not RAG)."
        )

    elif tool_name == "retain_cbr":
        changes.append(
            "Will retain labeled cases into case memory "
            "(refuses Session holdout indices)."
        )

    elif tool_name == "save_cbr_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save CBR bundle to {path}.")

    elif tool_name == "load_cbr_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load CBR bundle from {path}.")

    elif tool_name == "fit_imitation":
        changes.append(
            "Will fit behavioral cloning on train demonstrations "
            "(state→action; not inverse RL / robotics)."
        )

    elif tool_name == "save_imitation_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save imitation bundle to {path}.")

    elif tool_name == "load_imitation_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load imitation bundle from {path}.")

    elif tool_name == "fit_rl":
        mode = arguments.get("mode", "contextual_bandit")
        changes.append(
            f"Will fit RL mode={mode} "
            "(bandit train-only or optional gym_reinforce; not MuJoCo/robotics)."
        )

    elif tool_name == "save_rl_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save RL bundle to {path}.")

    elif tool_name == "load_rl_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load RL bundle from {path}.")

    elif tool_name == "fit_tda":
        changes.append(
            "Will fit TDA on train (local Vietoris–Rips + vectorization; "
            "requires buildml[tda]; not a Mapper suite)."
        )

    elif tool_name == "save_tda_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save TDA bundle to {path}.")

    elif tool_name == "load_tda_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load TDA bundle from {path}.")

    elif tool_name == "fit_recommender":
        changes.append(
            "Will fit a recommender on train interactions "
            "(CF / SVD-NMF / content; not RAG; not EDA Recommendation Findings)."
        )

    elif tool_name == "save_recommender_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save recommender bundle to {path}.")

    elif tool_name == "load_recommender_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load recommender bundle from {path}.")

    elif tool_name == "fit_ranker":
        changes.append(
            "Will fit a tabular LTR ranker on train query–item rows "
            "(pointwise or pairwise RankSVM-lite; not RAG; not recommender CF)."
        )

    elif tool_name == "save_ranker_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save ranker bundle to {path}.")

    elif tool_name == "load_ranker_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load ranker bundle from {path}.")

    elif tool_name == "fit_kg":
        backend = arguments.get("backend", "native")
        method = arguments.get("method", "transe")
        changes.append(
            f"Will fit KG embeddings on train triples "
            f"(backend={backend}, method={method}; not Graph ML node-classify; "
            "not Neo4j; not RAG)."
        )

    elif tool_name == "save_kg_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save KG bundle to {path}.")

    elif tool_name == "load_kg_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load KG bundle from {path}.")

    elif tool_name == "fit_decision_policy":
        method = arguments.get("method", "threshold")
        backend = arguments.get("backend")
        partition = arguments.get("partition", "validation")
        backend_note = f", backend={backend}" if backend else ""
        changes.append(
            f"Will fit decision policy method={method}{backend_note} on partition={partition} "
            "(ML score/cost/allocation helpers — not a general OR platform; "
            "test tuning requires allow_test_tuning=True)."
        )

    elif tool_name == "save_decision_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save decision bundle to {path}.")

    elif tool_name == "load_decision_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load decision bundle from {path}.")

    elif tool_name == "fit_synthesizer":
        method = arguments.get("method", "gaussian_copula")
        changes.append(
            f"Will fit synthesizer method={method} on train only "
            "(not differential privacy; distinct from Session.resample)."
        )

    elif tool_name == "sample_synthetic":
        merge_mode = arguments.get("merge_mode", "none")
        changes.append(
            f"Will sample synthetic rows (merge_mode={merge_mode})."
        )

    elif tool_name == "save_synthetic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will save synthetic bundle to {path}.")

    elif tool_name == "load_synthetic_bundle":
        path = arguments.get("path", "")
        changes.append(f"Will load synthetic bundle from {path}.")

    elif tool_name == "drop_columns":
        columns = arguments.get("columns", [])
        changes.append(f"DESTRUCTIVE: Will permanently drop columns {columns}.")

    elif tool_name == "checkpoint_save":
        path = arguments.get("path", "")
        changes.append(f"Will save checkpoint to {path}.")

    elif tool_name == "rag_ingest_corpus":
        changes.append("Will ingest a RAG corpus and clear any prior index.")

    elif tool_name == "rag_embed_and_index":
        changes.append("Will embed chunks and build the RAG vector index.")

    elif tool_name == "make_torch_loaders":
        changes.append("Will build Torch DataLoaders on the Session.")

    elif tool_name == "make_text_torch_loaders":
        changes.append("Will build text Torch DataLoaders with a train-only vocabulary.")

    elif tool_name == "fit_torch":
        changes.append("Will train a Torch module and store dl_train_result.")

    elif tool_name == "cross_validate_torch":
        changes.append("Will run fold-local Torch CV and store dl_cv_result.")

    elif tool_name == "make_multimodal_torch_loaders":
        changes.append(
            "Will build multimodal Torch DataLoaders "
            "(train-only vocab/normalize/image/audio stats)."
        )

    elif tool_name == "make_image_multimodal_torch_loaders":
        changes.append(
            "Will build image multimodal Torch DataLoaders "
            "(image ⊕ tabular and/or text and/or audio)."
        )

    elif tool_name == "make_audio_multimodal_torch_loaders":
        changes.append(
            "Will build audio multimodal Torch DataLoaders "
            "(audio ⊕ tabular and/or text and/or image)."
        )

    elif tool_name == "search_torch":
        changes.append("Will run inner-fold Torch hyperparameter search (not nested outer).")

    elif tool_name == "nested_cv_torch":
        changes.append("Will run nested Torch CV and store dl_nested_cv_result.")

    elif tool_name == "export_torch":
        path = arguments.get("path", "")
        changes.append(f"Will export the last Torch trainer to {path}.")

    elif tool_name == "make_speech_torch_loaders":
        changes.append("Will build speech classification Torch DataLoaders (finetune-lite).")

    elif tool_name == "fit_speech_torch":
        changes.append("Will fine-tune a tiny speech classifier (not FM-from-scratch).")

    elif tool_name == "transcribe_speech":
        changes.append("Will transcribe an audio column via stub or transformers ASR.")

    elif tool_name == "load_pretrained_backbone":
        modality = arguments.get("modality", "vision")
        changes.append(f"Will load a curated {modality} pretrained backbone hook.")

    elif tool_name == "pack_torchserve":
        path = arguments.get("output_dir", "")
        changes.append(f"Will pack a TorchServe directory at {path}.")

    elif tool_name == "prepare_tensorrt_export":
        path = arguments.get("output_dir", "")
        changes.append(f"Will write a TensorRT trtexec plan under {path}.")

    elif tool_name == "emit_k8s_ddp_job":
        path = arguments.get("path", "")
        changes.append(f"Will emit a K8s torchrun Job YAML at {path}.")

    elif tool_name == "domain_adapt_speech_torch":
        changes.append("Will domain-adapt speech classify (finetune-lite; not FM pretrain).")

    elif tool_name == "attach_backbone_head":
        changes.append("Will attach a classification head to the Session pretrained backbone.")

    elif tool_name == "emit_k8s_serve_deployment":
        path = arguments.get("path", "")
        changes.append(f"Will emit a K8s serve Deployment YAML at {path}.")

    return tuple(changes) if changes else ("Unknown state changes.",)


def _summarize_result(result: Any) -> str:
    """Create a brief summary of a tool result."""
    if result is None:
        return "No result returned."

    if isinstance(result, dict):
        keys = list(result.keys())[:5]
        return f"Result with keys: {keys}"

    if hasattr(result, "to_dict"):
        return f"Result: {type(result).__name__}"

    text = str(result)
    if len(text) > 200:
        return text[:200] + "..."
    return text


class IterationLimitExceeded(ValidationError):
    """Raised when max tool iterations is exceeded."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"Maximum tool iterations ({limit}) exceeded. "
            "This limit prevents runaway loops."
        )
