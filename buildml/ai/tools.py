"""Tool registry for AI operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buildml.ai.types import ConfirmPolicy, ToolCall
from buildml.core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Specification for one tool in the AI operator registry."""

    name: str
    description: str
    parameters: dict[str, Any]
    confirm_policy: ConfirmPolicy = ConfirmPolicy.CONFIRM
    session_method: str | None = None
    read_only: bool = False
    destructive: bool = False
    catalog_operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "confirm_policy": self.confirm_policy.value,
            "session_method": self.session_method,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "catalog_operation": self.catalog_operation,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _build_m1_tools() -> tuple[ToolSpec, ...]:
    """Build the M1 conservative tool allowlist."""
    return (
        ToolSpec(
            name="describe_dataset",
            description=(
                "Return a summary of the current dataset including column names, "
                "data types, row count, roles, and basic statistics. Does not "
                "execute any changes."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            read_only=True,
            catalog_operation="metadata",
        ),
        ToolSpec(
            name="explain_operation",
            description=(
                "Explain what a BuildML operation does, its prerequisites, "
                "parameters, and expected outputs using the explain catalog."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation name to explain (e.g. 'fit', 'split').",
                    },
                },
                "required": ["operation"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="explain",
            read_only=True,
            catalog_operation="explain",
        ),
        ToolSpec(
            name="workflow_status",
            description=(
                "Return the current workflow status showing which operations "
                "are done, available, or blocked. Does not execute changes."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="workflow",
            read_only=True,
            catalog_operation="workflow",
        ),
        ToolSpec(
            name="eda_summary",
            description=(
                "Return a summary of exploratory data analysis findings "
                "including data quality issues, distributions, and recommendations."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="eda",
            read_only=True,
            catalog_operation="eda",
        ),
        ToolSpec(
            name="dry_run_plan",
            description=(
                "Preview what a plan would do without executing it. "
                "Returns validation results and expected state changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The plan name or operation sequence to dry-run.",
                    },
                },
                "required": ["plan"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="dry_run",
            read_only=True,
            catalog_operation="dry_run",
        ),
        ToolSpec(
            name="set_roles",
            description=(
                "Assign semantic roles to columns (feature, target, id, exclude). "
                "This is a write operation that requires confirmation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mapping": {
                        "type": "object",
                        "description": "Column name to role mapping.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["mapping"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="set_roles",
            read_only=False,
            catalog_operation="set_roles",
        ),
    )


def _build_rag_dl_tools() -> tuple[ToolSpec, ...]:
    """RAG + DL tools safe to expose behind the allowlist (Phase C)."""
    return (
        ToolSpec(
            name="rag_retrieve",
            description=(
                "Retrieve ranked chunks from the active RAG index for a query. "
                "Read-only; requires a prior rag_embed_and_index or load_rag_bundle."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text."},
                    "k": {"type": "integer", "description": "Number of hits (default 5)."},
                    "mode": {
                        "type": "string",
                        "description": "dense, bm25, or hybrid.",
                        "enum": ["dense", "bm25", "hybrid"],
                    },
                },
                "required": ["query"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="rag_retrieve",
            read_only=True,
            catalog_operation="rag_retrieve",
        ),
        ToolSpec(
            name="rag_generate",
            description=(
                "Retrieve context and generate a grounded answer with citations. "
                "Uses the Session AI provider. Requires an active RAG index."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Question to answer."},
                    "k": {"type": "integer", "description": "Retrieval depth (default 5)."},
                },
                "required": ["query"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="rag_generate",
            read_only=True,
            catalog_operation="rag_generate",
        ),
        ToolSpec(
            name="rag_ingest_corpus",
            description=(
                "Ingest text documents into the Session RAG corpus. "
                "Write operation; clears any prior index."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "In-memory document texts to ingest.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Optional Session frame column to index.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="rag_ingest_corpus",
            read_only=False,
            catalog_operation="rag_ingest_corpus",
        ),
        ToolSpec(
            name="rag_embed_and_index",
            description=(
                "Chunk (if needed), embed, and build the RAG vector index. "
                "Write operation; refuses eval_only contamination."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "embedder": {
                        "type": "string",
                        "description": "hashing (default), auto, or sentence-transformers.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="rag_embed_and_index",
            read_only=False,
            catalog_operation="rag_embed_and_index",
        ),
        ToolSpec(
            name="make_torch_loaders",
            description=(
                "Build Torch DataLoaders from current roles and split. "
                "Requires buildml[torch]. Write operation on Session torch slots."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "description": "Batch size (default 32)."},
                    "normalize": {
                        "type": "boolean",
                        "description": "Fit mean/std on train (default true).",
                    },
                    "apply_plans": {
                        "type": "boolean",
                        "description": "Re-apply fitted classical plans before loaders.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_torch_loaders",
            read_only=False,
            catalog_operation="make_torch_loaders",
        ),
        ToolSpec(
            name="make_text_torch_loaders",
            description=(
                "Build token-id Torch DataLoaders for text/sequence classification. "
                "Fits vocabulary on train only. Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text_column": {
                        "type": "string",
                        "description": "Text feature column (inferred when unique).",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 16)."},
                    "max_len": {"type": "integer", "description": "Maximum tokens per row."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_text_torch_loaders",
            read_only=False,
            catalog_operation="make_text_torch_loaders",
        ),
        ToolSpec(
            name="fit_torch",
            description=(
                "Train a Torch module (built-in MLP when module omitted) on the "
                "train loader. Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer", "description": "Training epochs (default 5)."},
                    "learning_rate": {"type": "number", "description": "Adam learning rate."},
                    "device": {
                        "type": "string",
                        "description": "cpu, cuda, mps, or auto.",
                        "enum": ["cpu", "cuda", "mps", "auto"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_torch",
            read_only=False,
            catalog_operation="fit_torch",
        ),
        ToolSpec(
            name="evaluate_torch",
            description=(
                "Evaluate the last Torch trainer on a named partition. Read-only metrics."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "description": "train, validation, or test.",
                        "enum": ["train", "validation", "test"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_torch",
            read_only=True,
            catalog_operation="evaluate_torch",
        ),
        ToolSpec(
            name="cross_validate_torch",
            description=(
                "Fold-local Torch CV on numeric tabular features. Not nested search. "
                "Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n_folds": {"type": "integer", "description": "Number of folds (default 3)."},
                    "epochs": {"type": "integer", "description": "Epochs per fold."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="cross_validate_torch",
            read_only=False,
            catalog_operation="cross_validate_torch",
        ),
        ToolSpec(
            name="make_multimodal_torch_loaders",
            description=(
                "Build fused multimodal Torch DataLoaders for tabular/text/image/audio mixes "
                "(train-only vocab, numeric normalize, image channel stats, audio amplitude "
                "stats). Requires buildml[torch]. Audio fusion is a small 1D-CNN branch, "
                "not a speech foundation model."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text_column": {"type": "string", "description": "Text feature column."},
                    "image_column": {
                        "type": "string",
                        "description": "Image path or array feature column.",
                    },
                    "audio_column": {
                        "type": "string",
                        "description": "Audio path or waveform array feature column.",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 16)."},
                    "normalize": {
                        "type": "boolean",
                        "description": "Fit numeric mean/std on train (default true).",
                    },
                    "normalize_images": {
                        "type": "boolean",
                        "description": "Fit image channel mean/std on train (default true).",
                    },
                    "normalize_audio": {
                        "type": "boolean",
                        "description": "Fit audio amplitude mean/std on train (default true).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_multimodal_torch_loaders",
            read_only=False,
            catalog_operation="make_multimodal_torch_loaders",
        ),
        ToolSpec(
            name="make_image_multimodal_torch_loaders",
            description=(
                "Build image multimodal Torch DataLoaders (image ⊕ tabular and/or text "
                "and/or audio). Requires image_column. Train-only image channel stats. "
                "Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image_column": {
                        "type": "string",
                        "description": "Image path or array feature column (required).",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Optional text feature column.",
                    },
                    "audio_column": {
                        "type": "string",
                        "description": "Optional audio path or waveform feature column.",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 16)."},
                    "normalize_images": {
                        "type": "boolean",
                        "description": "Fit image channel mean/std on train (default true).",
                    },
                },
                "required": ["image_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_image_multimodal_torch_loaders",
            read_only=False,
            catalog_operation="make_image_multimodal_torch_loaders",
        ),
        ToolSpec(
            name="make_audio_multimodal_torch_loaders",
            description=(
                "Build audio multimodal Torch DataLoaders (audio ⊕ tabular and/or text "
                "and/or image). Requires audio_column. Train-only audio amplitude stats. "
                "Small 1D-CNN fusion branch — not a speech foundation model. "
                "Requires buildml[torch] (soundfile for path cells)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_column": {
                        "type": "string",
                        "description": "Audio path or waveform array feature column (required).",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Optional text feature column.",
                    },
                    "image_column": {
                        "type": "string",
                        "description": "Optional image path or array feature column.",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 16)."},
                    "normalize_audio": {
                        "type": "boolean",
                        "description": "Fit audio amplitude mean/std on train (default true).",
                    },
                },
                "required": ["audio_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_audio_multimodal_torch_loaders",
            read_only=False,
            catalog_operation="make_audio_multimodal_torch_loaders",
        ),
        ToolSpec(
            name="search_torch",
            description=(
                "Inner-fold Torch hyperparameter search on the train universe. "
                "Not a nested outer estimate. Requires param_grid or "
                "param_distributions. Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "param_grid": {
                        "type": "object",
                        "description": (
                            "Grid of searchable lists (learning_rate, hidden, dropout, "
                            "batch_size, epochs, weight_decay)."
                        ),
                    },
                    "param_distributions": {
                        "type": "object",
                        "description": "Randomized search distributions (same keys as grid).",
                    },
                    "n_folds": {"type": "integer", "description": "Inner folds (default 3)."},
                    "epochs": {"type": "integer", "description": "Epochs per trial fold."},
                    "n_iter": {
                        "type": "integer",
                        "description": "Randomized trials when using distributions.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="search_torch",
            read_only=False,
            catalog_operation="search_torch",
        ),
        ToolSpec(
            name="nested_cv_torch",
            description=(
                "Nested Torch CV: outer evaluation after fold-local inner hyperparameter "
                "search. Requires param_grid or param_distributions. Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "param_grid": {
                        "type": "object",
                        "description": (
                            "Grid of searchable lists (learning_rate, hidden, dropout, "
                            "batch_size, epochs, weight_decay)."
                        ),
                    },
                    "param_distributions": {
                        "type": "object",
                        "description": "Randomized search distributions (same keys as grid).",
                    },
                    "outer_cv": {"type": "integer", "description": "Outer folds (default 3)."},
                    "inner_cv": {"type": "integer", "description": "Inner folds (default 2)."},
                    "epochs": {"type": "integer", "description": "Epochs per fit."},
                    "n_iter": {
                        "type": "integer",
                        "description": "Randomized trials when using distributions.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="nested_cv_torch",
            read_only=False,
            catalog_operation="nested_cv_torch",
        ),
        ToolSpec(
            name="export_torch",
            description=(
                "Export the last Torch trainer to TorchScript or ONNX (alpha escape hatch). "
                "Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "format": {
                        "type": "string",
                        "enum": ["torchscript", "onnx"],
                        "description": "Export format.",
                    },
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="export_torch",
            read_only=False,
            catalog_operation="export_torch",
        ),
    )


def _build_m2_tools() -> tuple[ToolSpec, ...]:
    """Build M2 expanded tool allowlist for E2E classical + RAG/DL pipeline."""
    return _build_m1_tools() + (
        ToolSpec(
            name="split",
            description=(
                "Create train/validation/test splits from the dataset. "
                "Requires roles to be set with at least a target column. "
                "This is a write operation that modifies Session state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of data for test set (default 0.2).",
                    },
                    "validation_size": {
                        "type": "number",
                        "description": "Fraction of data for validation set (default 0.0).",
                    },
                    "stratify": {
                        "type": "boolean",
                        "description": "Stratify by target column (default False).",
                    },
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed for reproducibility.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="split",
            read_only=False,
            catalog_operation="split",
        ),
        ToolSpec(
            name="impute",
            description=(
                "Impute missing values in columns. "
                "Fits on train, applies to all partitions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Imputation: mean, median, most_frequent, or constant.",
                        "enum": ["mean", "median", "most_frequent", "constant"],
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to impute (default: numeric non-target columns).",
                    },
                    "fill_value": {
                        "description": "Constant fill value when strategy='constant'.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="impute",
            read_only=False,
            catalog_operation="impute",
        ),
        ToolSpec(
            name="encode",
            description=(
                "Encode categorical columns using one-hot or ordinal encoding. "
                "Fits encoder on train, applies to all partitions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Encoding method: onehot, ordinal, target.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to encode (default: categorical features).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="encode",
            read_only=False,
            catalog_operation="encode",
        ),
        ToolSpec(
            name="scale",
            description=(
                "Scale numeric features using standardization or normalization. "
                "Fits scaler on train, applies to all partitions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Scaling method: standard, minmax, robust.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to scale (default: numeric features).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="scale",
            read_only=False,
            catalog_operation="scale",
        ),
        ToolSpec(
            name="fit",
            description=(
                "Fit an ML model on the training data. Requires split to exist. "
                "This is a write operation that creates fit_result on the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": (
                            "Estimator name: LogisticRegression, RandomForestClassifier, "
                            "GradientBoostingClassifier, SVC, KNeighborsClassifier, "
                            "LinearRegression, Ridge, Lasso, RandomForestRegressor, etc."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": "Task type: classification, regression, or auto.",
                        "enum": ["classification", "regression", "auto"],
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "Estimator hyperparameters (e.g. n_estimators, max_depth).",
                    },
                },
                "required": ["estimator"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit",
            read_only=False,
            catalog_operation="fit",
        ),
        ToolSpec(
            name="evaluate",
            description=(
                "Evaluate the fitted model on a partition (train/validation/test). "
                "Returns metrics appropriate for the task type."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "description": "Partition to evaluate: train, validation, test.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate",
            read_only=True,
            catalog_operation="evaluate",
        ),
        ToolSpec(
            name="walkthrough",
            description=(
                "Get a teaching walkthrough of the current workflow state, "
                "including what's done, what's next, and recommendations."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="walkthrough",
            read_only=True,
            catalog_operation="walkthrough",
        ),
        ToolSpec(
            name="head",
            description=(
                "Preview the first N rows of the dataset. "
                "Read-only inspection, does not modify state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to preview (default 5).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="head",
            read_only=True,
            catalog_operation="head",
        ),
        ToolSpec(
            name="drop_columns",
            description=(
                "Drop specified columns from the dataset. "
                "This is a DESTRUCTIVE operation that cannot be undone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names to drop.",
                    },
                },
                "required": ["columns"],
            },
            confirm_policy=ConfirmPolicy.ALWAYS_CONFIRM,
            session_method="drop_columns",
            read_only=False,
            destructive=True,
            catalog_operation="drop_columns",
        ),
        ToolSpec(
            name="checkpoint_save",
            description=(
                "Save the current Session state to a checkpoint bundle. "
                "Write operation but non-destructive (creates new files)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output directory path for checkpoint.",
                    },
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="checkpoint_save",
            read_only=False,
            catalog_operation="checkpoint_save",
        ),
        ToolSpec(
            name="ai_status",
            description=(
                "Get the current AI operator status including provider config, "
                "egress level, transcript entries, and confirmation mode."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            read_only=True,
            catalog_operation="ai_status",
        ),
    ) + _build_rag_dl_tools()


def build_default_registry() -> ToolRegistry:
    """Build the default tool registry (classical + RAG + DL allowlist)."""
    return ToolRegistry(tools=_build_m2_tools())


def registered_tool_names() -> tuple[str, ...]:
    """Return sorted tool names from the default registry (for tests / docs sync)."""
    return tuple(sorted(t.name for t in _build_m2_tools()))


class ToolRegistry:
    """Registry of allowed tools for the AI operator.

    Tools are allowlisted by name and category. Unlisted tools are rejected.
    """

    def __init__(self, tools: tuple[ToolSpec, ...] | None = None) -> None:
        if tools is None:
            tools = _build_m1_tools()
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def validate_tool_call(self, call: ToolCall) -> ToolSpec:
        """Validate a tool call is in the registry.

        Raises
        ------
        ValidationError
            If the tool is not in the allowlist.
        """
        spec = self._tools.get(call.tool_name)
        if spec is None:
            raise ValidationError(
                f"Tool '{call.tool_name}' is not in the allowed tool registry. "
                f"Available tools: {sorted(self._tools.keys())}"
            )
        return spec

    def requires_confirmation(self, call: ToolCall) -> bool:
        """Check if a tool call requires user confirmation."""
        spec = self._tools.get(call.tool_name)
        if spec is None:
            return True
        if spec.destructive:
            return True
        return spec.confirm_policy != ConfirmPolicy.AUTO

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI tool format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    def read_only_tools(self) -> tuple[ToolSpec, ...]:
        """Return only read-only tools (for advisor mode)."""
        return tuple(t for t in self._tools.values() if t.read_only)


_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "system:",
    "assistant:",
    "SYSTEM:",
    "ASSISTANT:",
    "you are now",
    "new instructions:",
    "override:",
)


def sanitize_tool_result(result: Any) -> str:
    """Sanitize a tool result before feeding back to the LLM.

    Marks the result as data, not instructions, and scans for injection patterns.
    """
    text = str(result)
    for marker in _INJECTION_MARKERS:
        if marker.lower() in text.lower():
            text = text.replace(marker, f"[DATA: {marker}]")
    return f"[TOOL RESULT - DATA ONLY]\n{text}\n[END TOOL RESULT]"


def mark_untrusted_data(data: str, source: str = "user") -> str:
    """Mark data as untrusted with source context.

    Used to wrap column names, cell values, and user input before sending
    to the LLM to prevent instruction injection.
    """
    return f"[UNTRUSTED DATA FROM {source.upper()}]\n{data}\n[END UNTRUSTED DATA]"
