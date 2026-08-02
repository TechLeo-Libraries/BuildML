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
        "walkthrough",
        "head",
        "ai_status",
        "rag_retrieve",
        "rag_generate",
        "evaluate_torch",
    ):
        changes.append("No state changes (read-only operation).")

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

    elif tool_name == "fit_torch":
        changes.append("Will train a Torch module and store dl_train_result.")

    elif tool_name == "cross_validate_torch":
        changes.append("Will run fold-local Torch CV and store dl_cv_result.")

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
