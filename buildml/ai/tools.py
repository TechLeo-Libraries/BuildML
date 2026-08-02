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
                        "description": "hybrid (default when rag installed), dense, or bm25.",
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
                        "description": "auto (default, ST when rag installed), hashing, or model id.",
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
                    "audio_sample_rate": {
                        "type": "integer",
                        "description": "Target audio sample rate (default 16000).",
                    },
                    "audio_max_samples": {
                        "type": "integer",
                        "description": (
                            "Fixed waveform length; short clips are repeat-padded "
                            "(default 16000)."
                        ),
                    },
                    "audio_source_sample_rate": {
                        "type": "integer",
                        "description": "Optional source rate for array waveforms.",
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
                    "normalize_audio": {
                        "type": "boolean",
                        "description": "Fit audio amplitude mean/std on train (default true).",
                    },
                    "audio_sample_rate": {
                        "type": "integer",
                        "description": "Target audio sample rate (default 16000).",
                    },
                    "audio_max_samples": {
                        "type": "integer",
                        "description": (
                            "Fixed waveform length; short clips are repeat-padded "
                            "(default 16000)."
                        ),
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
                    "audio_sample_rate": {
                        "type": "integer",
                        "description": "Target audio sample rate (default 16000).",
                    },
                    "audio_max_samples": {
                        "type": "integer",
                        "description": (
                            "Fixed waveform length; short clips are repeat-padded "
                            "(default 16000)."
                        ),
                    },
                    "audio_source_sample_rate": {
                        "type": "integer",
                        "description": "Optional source rate for array waveforms.",
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
        ToolSpec(
            name="make_speech_torch_loaders",
            description=(
                "Build speech classification Torch DataLoaders from an audio column. "
                "Finetune-lite path — not training a foundation model from scratch. "
                "Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_column": {
                        "type": "string",
                        "description": "Audio path or waveform feature column.",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 8)."},
                    "sample_rate": {
                        "type": "integer",
                        "description": "Target sample rate (default 16000).",
                    },
                    "max_samples": {
                        "type": "integer",
                        "description": "Fixed waveform length (default 16000).",
                    },
                    "normalize_audio": {
                        "type": "boolean",
                        "description": "Fit amplitude mean/std on train (default true).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="make_speech_torch_loaders",
            read_only=False,
            catalog_operation="make_speech_torch_loaders",
        ),
        ToolSpec(
            name="fit_speech_torch",
            description=(
                "Fine-tune a tiny speech encoder + classifier head (finetune-lite). "
                "Not Whisper-scale FM training from scratch. Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_column": {
                        "type": "string",
                        "description": "Audio column when loaders must be built.",
                    },
                    "epochs": {"type": "integer", "description": "Training epochs (default 5)."},
                    "freeze_encoder": {
                        "type": "boolean",
                        "description": "Train head only when true.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_speech_torch",
            read_only=False,
            catalog_operation="fit_speech_torch",
        ),
        ToolSpec(
            name="transcribe_speech",
            description=(
                "ASR transcription for an audio column. backend=stub is CI-safe; "
                "backend=transformers requires buildml[speech] and may download weights. "
                "Integration path — not FM training from scratch."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_column": {
                        "type": "string",
                        "description": "Audio path or waveform feature column (required).",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["stub", "transformers"],
                        "description": "ASR backend (default stub).",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Optional Hugging Face model id for transformers.",
                    },
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                        "description": "Rows to transcribe (default all).",
                    },
                },
                "required": ["audio_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="transcribe_speech",
            read_only=True,
            catalog_operation="transcribe_speech",
        ),
        ToolSpec(
            name="load_pretrained_backbone",
            description=(
                "Load a curated vision/audio/speech pretrained backbone hook "
                "(weights=none|mock|pretrained). mock is CI-safe. Not a full zoo product. "
                "Requires buildml[torch] (+ vision/speech extras for some modalities)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "enum": ["vision", "audio", "speech"],
                        "description": "Backbone modality.",
                    },
                    "architecture": {
                        "type": "string",
                        "description": "Optional architecture id (defaults per modality).",
                    },
                    "weights": {
                        "type": "string",
                        "enum": ["none", "mock", "pretrained"],
                        "description": "Weight mode (default mock).",
                    },
                    "freeze": {
                        "type": "boolean",
                        "description": "Freeze backbone parameters (default true).",
                    },
                    "seed": {"type": "integer", "description": "Seed for mock init."},
                    "model_id": {
                        "type": "string",
                        "description": "Optional Hugging Face model id for audio/speech.",
                    },
                },
                "required": ["modality"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_pretrained_backbone",
            read_only=False,
            catalog_operation="load_pretrained_backbone",
        ),
        ToolSpec(
            name="pack_torchserve",
            description=(
                "Pack a TorchScript file into a TorchServe-ready directory "
                "(model.pt, handler, config, manifest). Does not run TorchServe."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Destination directory.",
                    },
                    "torchscript_path": {
                        "type": "string",
                        "description": "TorchScript file (or last export_torch).",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "TorchServe model name (default buildml_model).",
                    },
                },
                "required": ["output_dir"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="pack_torchserve",
            read_only=False,
            catalog_operation="pack_torchserve",
        ),
        ToolSpec(
            name="prepare_tensorrt_export",
            description=(
                "Write a TensorRT trtexec plan next to an ONNX artifact. "
                "Does not build TensorRT engines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Destination directory.",
                    },
                    "onnx_path": {
                        "type": "string",
                        "description": "ONNX file (or last export_torch).",
                    },
                    "engine_name": {
                        "type": "string",
                        "description": "Suggested engine filename (default model.engine).",
                    },
                    "fp16": {
                        "type": "boolean",
                        "description": "Include --fp16 in the example trtexec command.",
                    },
                },
                "required": ["output_dir"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="prepare_tensorrt_export",
            read_only=False,
            catalog_operation="prepare_tensorrt_export",
        ),
        ToolSpec(
            name="emit_k8s_ddp_job",
            description=(
                "Emit a Kubernetes Job YAML template for torchrun multi-node DDP. "
                "Not live multi-cluster orchestration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output YAML path.",
                    },
                    "nnodes": {
                        "type": "integer",
                        "description": "Number of nodes (default 2).",
                    },
                    "nproc_per_node": {
                        "type": "integer",
                        "description": "Processes per node (default 1).",
                    },
                    "script_path": {
                        "type": "string",
                        "description": "Training script path inside the container.",
                    },
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="emit_k8s_ddp_job",
            read_only=False,
            catalog_operation="emit_k8s_ddp_job",
        ),
        ToolSpec(
            name="domain_adapt_speech_torch",
            description=(
                "Domain-adapt / finetune-lite speech classify (alias path of fit_speech_torch "
                "with stronger disclosures). Not FM continued pretrain from scratch. "
                "Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "audio_column": {
                        "type": "string",
                        "description": "Audio column when loaders must be built.",
                    },
                    "epochs": {"type": "integer", "description": "Training epochs (default 5)."},
                    "freeze_encoder": {
                        "type": "boolean",
                        "description": "Train head only when true (default true).",
                    },
                    "batch_size": {"type": "integer", "description": "Batch size (default 8)."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="domain_adapt_speech_torch",
            read_only=False,
            catalog_operation="domain_adapt_speech_torch",
        ),
        ToolSpec(
            name="attach_backbone_head",
            description=(
                "Attach a classification head to the Session pretrained backbone "
                "(requires prior load_pretrained_backbone). Requires buildml[torch]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n_classes": {
                        "type": "integer",
                        "description": "Number of classification classes (>= 2).",
                    },
                    "freeze_backbone": {
                        "type": "boolean",
                        "description": "Optional freeze override; omit to keep loaded state.",
                    },
                },
                "required": ["n_classes"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="attach_backbone_head",
            read_only=False,
            catalog_operation="attach_backbone_head",
        ),
        ToolSpec(
            name="evaluate_asr",
            description=(
                "Score ASR hypotheses vs references with WER/CER. "
                "When hypotheses omitted, reuses last transcribe_speech texts."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Hypothesis transcripts (optional if Session has ASR).",
                    },
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Reference transcripts.",
                    },
                    "lowercase": {
                        "type": "boolean",
                        "description": "Lowercase before scoring (default true).",
                    },
                },
                "required": ["references"],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_asr",
            read_only=True,
            catalog_operation="evaluate_asr",
        ),
        ToolSpec(
            name="emit_k8s_serve_deployment",
            description=(
                "Emit a Kubernetes Deployment+Service YAML template for managed BuildML serve. "
                "Not live multi-cluster orchestration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output YAML path.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Deployment/Service name (default buildml-serve).",
                    },
                    "replicas": {
                        "type": "integer",
                        "description": "Replica count (default 1).",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Container/Service port (default 8080).",
                    },
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="emit_k8s_serve_deployment",
            read_only=False,
            catalog_operation="emit_k8s_serve_deployment",
        ),
        # serve_bundle intentionally omitted: network listener is CLI/Session-primary
        # (see EXPLICITLY_NON_AI_SESSION_METHODS in buildml.explain.sync).
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
                        "description": (
                            "Columns to impute (default: numeric feature-role columns; "
                            "skips ignore/id/target/group/time/weight)."
                        ),
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
                        "description": (
                            "Columns to encode (default: categorical feature-role columns; "
                            "skips ignore/id; pass explicitly to force-include)."
                        ),
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
                        "description": (
                            "Columns to scale (default: numeric feature-role columns; "
                            "skips ignore/id so costs/ids stay unmutated; "
                            "pass explicitly to force-include)."
                        ),
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
        ToolSpec(
            name="fit_clusters",
            description=(
                "Fit an unsupervised clusterer on the train partition only "
                "(kmeans, agglomerative, or dbscan). Optionally prefers "
                "Session.reduce_dimensions PCA components. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": [
                            "kmeans",
                            "agglomerative",
                            "dbscan",
                            "gmm",
                            "hdbscan",
                            "spectral",
                            "optics",
                            "mean_shift",
                            "dec",
                            "idec",
                        ],
                        "description": "Clustering algorithm.",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Requested k for kmeans/agglomerative.",
                    },
                    "prefer_reduce_components": {
                        "type": "boolean",
                        "description": "Prefer PCA component columns when present.",
                    },
                    "eps": {"type": "number", "description": "DBSCAN eps."},
                    "min_samples": {"type": "integer", "description": "DBSCAN min_samples."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_clusters",
            read_only=False,
            catalog_operation="fit_clusters",
        ),
        ToolSpec(
            name="assign_clusters",
            description=(
                "Assign cluster labels with the train-fitted ClusterPlan (no refit). "
                "Read-only unless attach=true with partition=all."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                        "description": "Rows to label.",
                    },
                    "attach": {
                        "type": "boolean",
                        "description": "Write label_column onto the frame (requires partition=all).",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="assign_clusters",
            read_only=True,
            catalog_operation="assign_clusters",
        ),
        ToolSpec(
            name="evaluate_clusters",
            description=(
                "Evaluate train-fitted clusters with geometric validity metrics "
                "(optional external ARI/NMI). Not supervised accuracy."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "external_label_column": {
                        "type": "string",
                        "description": "Optional reference labels for ARI/NMI only.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_clusters",
            read_only=True,
            catalog_operation="evaluate_clusters",
        ),
        ToolSpec(
            name="save_unsupervised_bundle",
            description=(
                "Persist the active ClusterPlan as buildml.unsupervised_bundle.v2. "
                "Distinct from Session checkpoints and Torch/RAG bundles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_unsupervised_bundle",
            read_only=False,
            catalog_operation="save_unsupervised_bundle",
        ),
        ToolSpec(
            name="load_unsupervised_bundle",
            description="Load a buildml.unsupervised_bundle.v2 (or v1) ClusterPlan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_unsupervised_bundle",
            read_only=False,
            catalog_operation="load_unsupervised_bundle",
        ),
        ToolSpec(
            name="fit_voting",
            description=(
                "Fit a native VotingClassifier/VotingRegressor on train only from "
                "two or more allowlisted estimator names. Distinct from fitting a "
                "single RandomForest via fit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "At least two estimator names, e.g. "
                            "['LogisticRegression', 'RandomForestClassifier']."
                        ),
                    },
                    "voting": {
                        "type": "string",
                        "enum": ["hard", "soft"],
                        "description": "Voting mode for classification.",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                },
                "required": ["estimators"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_voting",
            read_only=False,
            catalog_operation="fit_voting",
        ),
        ToolSpec(
            name="fit_stacking",
            description=(
                "Fit a native StackingClassifier/StackingRegressor on train only. "
                "Stacking CV folds stay inside train; Session test is held out."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "At least two allowlisted estimator names.",
                    },
                    "cv": {
                        "type": "integer",
                        "description": "Out-of-fold folds inside train.",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                },
                "required": ["estimators"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_stacking",
            read_only=False,
            catalog_operation="fit_stacking",
        ),
        ToolSpec(
            name="fit_blending",
            description=(
                "Fit a holdout-blend ensemble on train only. The blend holdout is "
                "carved from train; Session validation/test never enter meta fit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "At least two allowlisted estimator names.",
                    },
                    "holdout_fraction": {
                        "type": "number",
                        "description": "Fraction of train reserved for meta-learner fit.",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                },
                "required": ["estimators"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_blending",
            read_only=False,
            catalog_operation="fit_blending",
        ),
        ToolSpec(
            name="evaluate_ensemble",
            description=(
                "Evaluate the last native ensemble with classical supervised metrics "
                "plus strategy disclosures."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_ensemble",
            read_only=True,
            catalog_operation="evaluate_ensemble",
        ),
        ToolSpec(
            name="save_ensemble_bundle",
            description=(
                "Persist the active EnsemblePlan as buildml.ensemble_bundle.v1. "
                "Distinct from Session checkpoints and classical pipelines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_ensemble_bundle",
            read_only=False,
            catalog_operation="save_ensemble_bundle",
        ),
        ToolSpec(
            name="load_ensemble_bundle",
            description="Load a buildml.ensemble_bundle.v1 EnsemblePlan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_ensemble_bundle",
            read_only=False,
            catalog_operation="load_ensemble_bundle",
        ),
        ToolSpec(
            name="evolutionary_search",
            description=(
                "Genetic-algorithm hyperparameter search on train-fold CV only. "
                "In-tree NumPy GA (population, selection, crossover/mutation, elitism) "
                "— not random search renamed, not NAS, not a swarm zoo. Same leakage "
                "refusal as grid_search for Session-global preprocess."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": (
                            "Base estimator name to evolve hyperparameters for "
                            "(e.g. DecisionTreeClassifier, LogisticRegression)."
                        ),
                    },
                    "param_space": {
                        "type": "object",
                        "description": (
                            "Declare-style space: "
                            "{name: {type: float|int|categorical, ...}} or list choices."
                        ),
                    },
                    "recipe_space": {
                        "type": "object",
                        "description": "Optional fold-local recipe knobs (requires preprocess).",
                    },
                    "population_size": {
                        "type": "integer",
                        "description": "GA population size (default 12).",
                    },
                    "n_generations": {
                        "type": "integer",
                        "description": "GA generations (default 5).",
                    },
                    "max_evaluations": {
                        "type": "integer",
                        "description": "Hard cap on unique CV evaluations.",
                    },
                    "cv": {
                        "type": "integer",
                        "description": "Inner CV folds (default 3 for AI path).",
                    },
                    "ranking_metric": {
                        "type": "string",
                        "description": "Metric used to rank genomes.",
                    },
                    "refit": {
                        "type": "boolean",
                        "description": "Refit winner on full train (default true).",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                },
                "required": ["estimator", "param_space"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="evolutionary_search",
            read_only=False,
            catalog_operation="evolutionary_search",
        ),
        ToolSpec(
            name="run_automl",
            description=(
                "Search model families and fold-local preprocess strategies on train "
                "(beyond single-estimator HPO). Not NAS and not causal. Session test "
                "never enters selection. Prefer unpoisoned data (no Session-global prep)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "optuna", "flaml", "autogluon"],
                    },
                    "method": {
                        "type": "string",
                        "enum": ["randomized", "grid", "optuna", "evolutionary"],
                    },
                    "selection": {
                        "type": "string",
                        "enum": ["cv", "nested", "validation"],
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Trial budget (native backends).",
                    },
                    "time_budget": {
                        "type": "number",
                        "description": "Optional wall-clock cap in seconds.",
                    },
                    "include_recipe_search": {"type": "boolean"},
                    "include_industry_families": {"type": "boolean"},
                    "include_ensembles": {"type": "boolean"},
                    "ensemble_mode": {
                        "type": "string",
                        "enum": ["voting", "stacking", "both"],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                    "families": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of catalog family names.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="run_automl",
            read_only=False,
            catalog_operation="run_automl",
        ),
        ToolSpec(
            name="evaluate_automl",
            description=(
                "Evaluate the last AutoML winner with classical supervised metrics "
                "plus AutoML disclosures."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_automl",
            read_only=True,
            catalog_operation="evaluate_automl",
        ),
        ToolSpec(
            name="save_automl_bundle",
            description=(
                "Persist the active AutoMLPlan as buildml.automl_bundle.v1. "
                "Distinct from Session checkpoints and classical pipelines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_automl_bundle",
            read_only=False,
            catalog_operation="save_automl_bundle",
        ),
        ToolSpec(
            name="load_automl_bundle",
            description="Load a buildml.automl_bundle.v1 AutoMLPlan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_automl_bundle",
            read_only=False,
            catalog_operation="load_automl_bundle",
        ),
        ToolSpec(
            name="fit_forecast",
            description=(
                "Fit a forecaster on the train partition only. "
                "auto=ETS when statsmodels installed else lag_ridge. "
                "Baselines, lag models, ARIMA/ETS/SARIMAX (timeseries extra), "
                "Prophet (timeseries-prophet), N-BEATS (timeseries-ml). "
                "Requires time_split. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "naive",
                            "seasonal_naive",
                            "drift",
                            "mean",
                            "lag_ridge",
                            "lag_hgb",
                            "arima",
                            "auto_arima",
                            "ets",
                            "sarimax",
                            "prophet",
                            "nbeats",
                        ],
                        "description": "Forecast algorithm.",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Default generate horizon.",
                    },
                    "lags": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Positive lag orders.",
                    },
                    "seasonal_period": {
                        "type": "integer",
                        "description": "Season length for seasonal_naive.",
                    },
                    "exog_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional numeric exogenous columns.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_forecast",
            read_only=False,
            catalog_operation="fit_forecast",
        ),
        ToolSpec(
            name="generate_forecast",
            description=(
                "Generate an H-step forecast from the train-fitted ForecastPlan "
                "(no refit). Recursive multi-step; future_exog required for exog plans."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "horizon": {"type": "integer", "description": "Steps ahead."},
                    "origin": {
                        "type": "string",
                        "enum": ["train_end", "validation_end", "test_end"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="generate_forecast",
            read_only=True,
            catalog_operation="generate_forecast",
        ),
        ToolSpec(
            name="evaluate_forecast",
            description=(
                "Evaluate the train-fitted ForecastPlan with MAE/RMSE/MAPE "
                "(rolling_one_step, origin, or rolling_origin)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["validation", "test"],
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["rolling_one_step", "origin", "rolling_origin"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_forecast",
            read_only=True,
            catalog_operation="evaluate_forecast",
        ),
        ToolSpec(
            name="save_forecast_bundle",
            description=(
                "Persist the active ForecastPlan as buildml.forecast_bundle.v2. "
                "Distinct from Session checkpoints and Torch/RAG bundles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_forecast_bundle",
            read_only=False,
            catalog_operation="save_forecast_bundle",
        ),
        ToolSpec(
            name="load_forecast_bundle",
            description="Load a buildml.forecast_bundle.v2 (or v1) ForecastPlan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_forecast_bundle",
            read_only=False,
            catalog_operation="load_forecast_bundle",
        ),
        ToolSpec(
            name="analyze_timeseries",
            description=(
                "Run time-series analysis on train scope: STL decomposition, "
                "ACF/PACF, ADF/KPSS, changepoints, rolling/spectral features. "
                "Requires time_split. Industry defaults with buildml[timeseries]."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["train", "all"]},
                    "decompose_method": {
                        "type": "string",
                        "enum": ["stl", "classical", "moving_average"],
                    },
                    "seasonal_period": {"type": "integer"},
                    "include_decompose": {"type": "boolean"},
                    "include_diagnostics": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="analyze_timeseries",
            read_only=True,
            catalog_operation="analyze_timeseries",
        ),
        ToolSpec(
            name="ts_decompose",
            description="STL/classical seasonal decomposition (train-only default).",
            parameters={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["train", "all"]},
                    "decompose_method": {
                        "type": "string",
                        "enum": ["stl", "classical", "moving_average"],
                    },
                    "seasonal_period": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="ts_decompose",
            read_only=True,
            catalog_operation="ts_decompose",
        ),
        ToolSpec(
            name="ts_diagnostics",
            description="ACF/PACF and ADF/KPSS stationarity diagnostics.",
            parameters={
                "type": "object",
                "properties": {
                    "acf_lags": {"type": "integer"},
                    "pacf_lags": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="ts_diagnostics",
            read_only=True,
            catalog_operation="ts_diagnostics",
        ),
        ToolSpec(
            name="fit_anomaly",
            description=(
                "Fit an anomaly/fraud detector on the train partition only. "
                "Backends: sklearn (core), pyod (anomaly-industry), torch (autoencoder). "
                "Distinct from EDA IsolationForest screens. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "pyod", "torch"],
                        "description": "Detector backend (see anomaly_capability_matrix).",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "isolation_forest",
                            "lof",
                            "one_class_svm",
                            "hbos",
                            "copod",
                            "ecod",
                            "deepsvdd",
                            "autoencoder",
                            "supervised_hgb",
                            "supervised_xgb",
                            "supervised_lgbm",
                        ],
                        "description": "Detector / scorer algorithm.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["unsupervised", "novelty", "supervised"],
                        "description": "Fit regime.",
                    },
                    "contamination": {
                        "type": "number",
                        "description": "Prior alert fraction / IF-LOF contamination.",
                    },
                    "threshold_policy": {
                        "type": "string",
                        "enum": [
                            "contamination",
                            "quantile",
                            "score_threshold",
                            "decision_zero",
                        ],
                    },
                    "normal_label_column": {
                        "type": "string",
                        "description": "Novelty: column defining normal-only fit rows.",
                    },
                    "prefer_reduce_components": {
                        "type": "boolean",
                        "description": "Prefer PCA component columns when present.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_anomaly",
            read_only=False,
            catalog_operation="fit_anomaly",
        ),
        ToolSpec(
            name="score_anomalies",
            description=(
                "Score and flag rows with the train-fitted AnomalyPlan (no refit). "
                "Reports threshold and alert_rate. Read-only unless attach=true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "attach": {
                        "type": "boolean",
                        "description": "Write score/flag columns (requires partition=all).",
                    },
                    "override_threshold": {
                        "type": "number",
                        "description": "Optional absolute threshold for this call only.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="score_anomalies",
            read_only=True,
            catalog_operation="score_anomalies",
        ),
        ToolSpec(
            name="evaluate_anomaly",
            description=(
                "Evaluate train-fitted anomaly scores (alert_rate + optional "
                "PR-AUC / precision@k under labels). Not a full fraud platform."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Optional labels for ranking metrics.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Budget for precision@k / recall@k.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_anomaly",
            read_only=True,
            catalog_operation="evaluate_anomaly",
        ),
        ToolSpec(
            name="tune_anomaly_threshold",
            description=(
                "Tune anomaly threshold on validation labels (leakage-safe). "
                "Same discipline as tune_threshold; never test unless allow_test_tuning."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["f1", "fbeta", "precision_at_contamination", "youden"],
                    },
                    "k": {"type": "integer", "description": "Unused; reserved."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="tune_anomaly_threshold",
            read_only=False,
            catalog_operation="tune_anomaly_threshold",
        ),
        ToolSpec(
            name="anomaly_capability_matrix",
            description="Return honest anomaly backend/method capability matrix.",
            parameters={"type": "object", "properties": {}, "required": []},
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="anomaly_capability_matrix",
            read_only=True,
            catalog_operation="anomaly_capability_matrix",
        ),
        ToolSpec(
            name="save_anomaly_bundle",
            description=(
                "Persist the active AnomalyPlan as buildml.anomaly_bundle.v1. "
                "Distinct from Session checkpoints and Torch/RAG/unsupervised bundles."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_anomaly_bundle",
            read_only=False,
            catalog_operation="save_anomaly_bundle",
        ),
        ToolSpec(
            name="load_anomaly_bundle",
            description="Load a buildml.anomaly_bundle.v1 AnomalyPlan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_anomaly_bundle",
            read_only=False,
            catalog_operation="load_anomaly_bundle",
        ),
        ToolSpec(
            name="fit_semisupervised",
            description=(
                "Fit a semi-supervised classifier on scarce labeled + unlabeled "
                "train rows. Backends: sklearn (default), industry (XGB/LGBM "
                "pseudo-label), torch (FixMatch/MixMatch tabular), hf (text "
                "pseudo-label). Target NaNs mark unlabeled. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch", "hf"],
                        "description": "Semi-supervised backend (honest default when omitted).",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "label_propagation",
                            "label_spreading",
                            "self_training",
                            "pseudo_label_xgb",
                            "pseudo_label_lgbm",
                            "fixmatch_tabular",
                            "mixmatch_tabular",
                            "text_pseudo_label",
                        ],
                    },
                    "base_estimator": {
                        "type": "string",
                        "enum": ["logistic_regression", "hist_gradient_boosting"],
                        "description": "Sklearn self-training base classifier.",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Pseudo-label confidence threshold.",
                    },
                    "n_neighbors": {
                        "type": "integer",
                        "description": "Graph neighborhood size.",
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Torch consistency training epochs.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Text feature column for hf backend.",
                    },
                    "prefer_reduce_components": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_semisupervised",
            read_only=False,
            catalog_operation="fit_semisupervised",
        ),
        ToolSpec(
            name="evaluate_semisupervised",
            description=(
                "Evaluate semi-supervised predictions on labeled holdout rows only. "
                "Unlabeled holdout rows are disclosed, never treated as truth."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_semisupervised",
            read_only=True,
            catalog_operation="evaluate_semisupervised",
        ),
        ToolSpec(
            name="save_semisupervised_bundle",
            description=(
                "Persist the active SemiSupervisedPlan as "
                "buildml.semisupervised_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_semisupervised_bundle",
            read_only=False,
            catalog_operation="save_semisupervised_bundle",
        ),
        ToolSpec(
            name="load_semisupervised_bundle",
            description=(
                "Load a buildml.semisupervised_bundle.v1 SemiSupervisedPlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_semisupervised_bundle",
            read_only=False,
            catalog_operation="load_semisupervised_bundle",
        ),
        ToolSpec(
            name="fit_ssl_pretext",
            description=(
                "Fit a self-supervised masked tabular pretext on the train partition "
                "(labels ignored). Not BERT-from-scratch. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["masked_tabular"],
                    },
                    "latent_dim": {
                        "type": "integer",
                        "description": "Bottleneck / representation width.",
                    },
                    "mask_ratio": {
                        "type": "number",
                        "description": "Fraction of features masked per view.",
                    },
                    "prefer_reduce_components": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_ssl_pretext",
            read_only=False,
            catalog_operation="fit_ssl_pretext",
        ),
        ToolSpec(
            name="finetune_ssl_head",
            description=(
                "Fit a supervised head on frozen SSL embeddings using labeled train "
                "rows only (skips NaN targets)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "enum": ["logistic_regression", "hist_gradient_boosting"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="finetune_ssl_head",
            read_only=False,
            catalog_operation="finetune_ssl_head",
        ),
        ToolSpec(
            name="evaluate_ssl",
            description=(
                "Evaluate frozen SSL pretext + head on labeled partition rows. "
                "Reconstruction MAE is not predictive utility."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_ssl",
            read_only=True,
            catalog_operation="evaluate_ssl",
        ),
        ToolSpec(
            name="save_ssl_bundle",
            description=(
                "Persist the active SelfSupervisedPlan (+ optional head) as "
                "buildml.selfsupervised_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_ssl_bundle",
            read_only=False,
            catalog_operation="save_ssl_bundle",
        ),
        ToolSpec(
            name="load_ssl_bundle",
            description=(
                "Load a buildml.selfsupervised_bundle.v1 plan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_ssl_bundle",
            read_only=False,
            catalog_operation="load_ssl_bundle",
        ),
        ToolSpec(
            name="fit_active_learner",
            description=(
                "Fit / initialize an active learner on labeled train rows only. "
                "Unlabeled pool is train target NaNs. Labels come from the user "
                "(no oracle in core). Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                    },
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "least_confidence",
                            "margin",
                            "entropy",
                            "committee",
                            "expected_model_change_lite",
                            "core_set",
                            "qbc_kl",
                            "qbc_variation_ratios",
                            "bald",
                            "mc_dropout",
                        ],
                    },
                    "base_estimator": {
                        "type": "string",
                        "enum": ["logistic_regression", "hist_gradient_boosting"],
                    },
                    "batch_size": {"type": "integer"},
                    "label_budget": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_active_learner",
            read_only=False,
            catalog_operation="fit_active_learner",
        ),
        ToolSpec(
            name="suggest_query",
            description=(
                "Suggest unlabeled train-pool indices for human labeling. "
                "Never queries validation/test. Does not invent labels."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer"},
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "least_confidence",
                            "margin",
                            "entropy",
                            "committee",
                            "expected_model_change_lite",
                            "core_set",
                            "qbc_kl",
                            "qbc_variation_ratios",
                            "bald",
                            "mc_dropout",
                        ],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="suggest_query",
            read_only=True,
            catalog_operation="suggest_query",
        ),
        ToolSpec(
            name="evaluate_active_learning",
            description=(
                "Evaluate the active learner on labeled holdout rows only. "
                "Unlabeled holdout rows are disclosed, never treated as truth."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_active_learning",
            read_only=True,
            catalog_operation="evaluate_active_learning",
        ),
        ToolSpec(
            name="save_active_learning_bundle",
            description=(
                "Persist the active ActiveLearningPlan as "
                "buildml.activelearning_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_active_learning_bundle",
            read_only=False,
            catalog_operation="save_active_learning_bundle",
        ),
        ToolSpec(
            name="load_active_learning_bundle",
            description=(
                "Load a buildml.activelearning_bundle.v1 ActiveLearningPlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_active_learning_bundle",
            read_only=False,
            catalog_operation="load_active_learning_bundle",
        ),
        ToolSpec(
            name="fit_online",
            description=(
                "Warm-start an incremental partial_fit estimator on a train chunk. "
                "Validation/test are never used for updates. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                    },
                    "estimator": {"type": "string"},
                    "chunk_size": {"type": "integer"},
                    "n_init": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                    "allow_refit_fallback": {"type": "boolean"},
                    "drift_disclose": {"type": "boolean"},
                    "drift_detector": {
                        "type": "string",
                        "enum": ["mean_shift", "adwin", "page_hinkley", "none"],
                    },
                    "buffer_size": {"type": "integer"},
                    "epochs_per_update": {"type": "integer"},
                    "device": {"type": "string"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_online",
            read_only=False,
            catalog_operation="fit_online",
        ),
        ToolSpec(
            name="partial_fit_online",
            description=(
                "Incremental partial_fit update on the next train chunk. "
                "Refuses validation/test indices. Never silently full-refits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n_rows": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="partial_fit_online",
            read_only=False,
            catalog_operation="partial_fit_online",
        ),
        ToolSpec(
            name="evaluate_online",
            description=(
                "Evaluate the online learner on a holdout partition. "
                "Holdout rows are never used for partial_fit updates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_online",
            read_only=True,
            catalog_operation="evaluate_online",
        ),
        ToolSpec(
            name="save_online_bundle",
            description=(
                "Persist the active OnlinePlan as buildml.online_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_online_bundle",
            read_only=False,
            catalog_operation="save_online_bundle",
        ),
        ToolSpec(
            name="load_online_bundle",
            description=(
                "Load a buildml.online_bundle.v1 OnlinePlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_online_bundle",
            read_only=False,
            catalog_operation="load_online_bundle",
        ),
        ToolSpec(
            name="fit_multitask",
            description=(
                "Fit a multi-target estimator on train only (sklearn/industry/torch). "
                "Requires >=2 targets. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "multi_output",
                            "classifier_chain",
                            "regressor_chain",
                            "multi_output_xgb",
                            "multi_output_lgbm",
                            "multi_output_catboost",
                            "shared_trunk_multihead",
                        ],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto", "mixed"],
                    },
                    "base_estimator": {
                        "type": "string",
                        "enum": [
                            "logistic_regression",
                            "hist_gradient_boosting",
                            "ridge",
                            "hist_gradient_boosting_regressor",
                        ],
                    },
                    "prefer_reduce_components": {"type": "boolean"},
                    "epochs": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "device": {"type": "string"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_multitask",
            read_only=False,
            catalog_operation="fit_multitask",
        ),
        ToolSpec(
            name="evaluate_multitask",
            description=(
                "Evaluate multi-task predictions with per-task and aggregate "
                "metrics. Holdout rows are never used for fitting."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_multitask",
            read_only=True,
            catalog_operation="evaluate_multitask",
        ),
        ToolSpec(
            name="save_multitask_bundle",
            description=(
                "Persist the active MultiTaskPlan as buildml.multitask_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_multitask_bundle",
            read_only=False,
            catalog_operation="save_multitask_bundle",
        ),
        ToolSpec(
            name="load_multitask_bundle",
            description=(
                "Load a buildml.multitask_bundle.v1 MultiTaskPlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_multitask_bundle",
            read_only=False,
            catalog_operation="load_multitask_bundle",
        ),
        ToolSpec(
            name="fit_metalearning",
            description=(
                "Meta-train a tabular few-shot / episodic learner on train tasks "
                "only. Needs a task/group column and one target. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "torch", "industry"],
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "prototypical",
                            "warm_start",
                            "prototypical_torch",
                            "maml",
                            "reptile",
                        ],
                    },
                    "task_column": {"type": "string"},
                    "k_shot": {"type": "integer"},
                    "n_query": {"type": "integer"},
                    "n_episodes": {"type": "integer"},
                    "base_estimator": {
                        "type": "string",
                        "enum": ["logistic_regression", "sgd_classifier"],
                    },
                    "prefer_reduce_components": {"type": "boolean"},
                    "task_holdout_fraction": {"type": "number"},
                    "meta_epochs": {"type": "integer"},
                    "inner_lr": {"type": "number"},
                    "inner_steps": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_metalearning",
            read_only=False,
            catalog_operation="fit_metalearning",
        ),
        ToolSpec(
            name="adapt_to_task",
            description=(
                "Fast-adapt the meta-learner to one task's labeled support set. "
                "Write operation (stores adapt result)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "description": "Task id to adapt to (from partition).",
                    },
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                    "max_support_per_class": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="adapt_to_task",
            read_only=False,
            catalog_operation="adapt_to_task",
        ),
        ToolSpec(
            name="evaluate_metalearning",
            description=(
                "Evaluate episodic few-shot performance on a holdout partition. "
                "Prefer novel task ids. Holdout rows are never used for meta-train."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k_shot": {"type": "integer"},
                    "prefer_novel_tasks": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_metalearning",
            read_only=True,
            catalog_operation="evaluate_metalearning",
        ),
        ToolSpec(
            name="save_metalearning_bundle",
            description=(
                "Persist the active MetaLearningPlan as "
                "buildml.metalearning_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_metalearning_bundle",
            read_only=False,
            catalog_operation="save_metalearning_bundle",
        ),
        ToolSpec(
            name="load_metalearning_bundle",
            description=(
                "Load a buildml.metalearning_bundle.v1 MetaLearningPlan into "
                "the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_metalearning_bundle",
            read_only=False,
            catalog_operation="load_metalearning_bundle",
        ),
        ToolSpec(
            name="fit_federated",
            description=(
                "Simulate federated averaging on Session train clients "
                "(local FedAvg/FedProx). backend='native' or 'flower' when "
                "buildml[federated-industry] installed. Needs a client/group "
                "column and one target. Not a networked FL deployment. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "flower"],
                    },
                    "method": {
                        "type": "string",
                        "enum": ["fedavg", "fedprox"],
                    },
                    "estimator": {
                        "type": "string",
                        "enum": [
                            "sgd_classifier",
                            "sgd_regressor",
                            "logistic_regression",
                            "ridge",
                            "linear_regression",
                        ],
                    },
                    "client_column": {"type": "string"},
                    "n_rounds": {"type": "integer"},
                    "local_epochs": {"type": "integer"},
                    "client_fraction": {"type": "number"},
                    "mu": {"type": "number"},
                    "prefer_reduce_components": {"type": "boolean"},
                    "min_client_rows": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_federated",
            read_only=False,
            catalog_operation="fit_federated",
        ),
        ToolSpec(
            name="evaluate_federated",
            description=(
                "Evaluate the global federated model on a holdout partition. "
                "Holdout rows are never used for local client updates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "per_client": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_federated",
            read_only=True,
            catalog_operation="evaluate_federated",
        ),
        ToolSpec(
            name="predict_federated",
            description=(
                "Predict with the global federated estimator (no update)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_federated",
            read_only=True,
            catalog_operation="predict_federated",
        ),
        ToolSpec(
            name="save_federated_bundle",
            description=(
                "Persist the active FederatedPlan as "
                "buildml.federated_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_federated_bundle",
            read_only=False,
            catalog_operation="save_federated_bundle",
        ),
        ToolSpec(
            name="load_federated_bundle",
            description=(
                "Load a buildml.federated_bundle.v1 FederatedPlan into "
                "the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_federated_bundle",
            read_only=False,
            catalog_operation="load_federated_bundle",
        ),
        ToolSpec(
            name="fit_probabilistic",
            description=(
                "Fit a Bayesian / probabilistic estimator with uncertainty "
                "(native sklearn, optional MAPIE conformal, optional NGBoost). "
                "Not a PyMC/Stan platform. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "mapie", "ngboost"],
                    },
                    "estimator": {
                        "type": "string",
                        "enum": [
                            "bayesian_ridge",
                            "gaussian_process_regressor",
                            "gaussian_process_classifier",
                            "gaussian_nb",
                            "split",
                            "cv_plus",
                            "jackknife_plus",
                            "ngboost_regressor",
                            "ngboost_classifier",
                        ],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "alpha": {"type": "number"},
                    "conformal": {"type": "boolean"},
                    "conformal_calibration_fraction": {"type": "number"},
                    "interval_method": {
                        "type": "string",
                        "enum": [
                            "posterior_std",
                            "split_conformal",
                            "both",
                            "none",
                            "mapie",
                            "mapie_cv_plus",
                            "mapie_jackknife_plus",
                        ],
                    },
                    "prefer_reduce_components": {"type": "boolean"},
                    "n_restarts_optimizer": {"type": "integer"},
                    "n_estimators": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_probabilistic",
            read_only=False,
            catalog_operation="fit_probabilistic",
        ),
        ToolSpec(
            name="evaluate_probabilistic",
            description=(
                "Evaluate probabilistic predictions with NLL / coverage / "
                "Brier on a holdout partition. Holdout never used for fit or "
                "conformal calibration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "alpha": {"type": "number"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_probabilistic",
            read_only=True,
            catalog_operation="evaluate_probabilistic",
        ),
        ToolSpec(
            name="predict_probabilistic",
            description=(
                "Predict with the probabilistic estimator (optional std/proba)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "return_std": {"type": "boolean"},
                    "return_proba": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_probabilistic",
            read_only=True,
            catalog_operation="predict_probabilistic",
        ),
        ToolSpec(
            name="predict_interval",
            description=(
                "Predictive intervals (regression) or conformal prediction "
                "sets (classification) from the ProbabilisticPlan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "alpha": {"type": "number"},
                    "method": {
                        "type": "string",
                        "enum": ["posterior_std", "split_conformal", "both"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_interval",
            read_only=True,
            catalog_operation="predict_interval",
        ),
        ToolSpec(
            name="save_probabilistic_bundle",
            description=(
                "Persist the active ProbabilisticPlan as "
                "buildml.probabilistic_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_probabilistic_bundle",
            read_only=False,
            catalog_operation="save_probabilistic_bundle",
        ),
        ToolSpec(
            name="load_probabilistic_bundle",
            description=(
                "Load a buildml.probabilistic_bundle.v1 ProbabilisticPlan into "
                "the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_probabilistic_bundle",
            read_only=False,
            catalog_operation="load_probabilistic_bundle",
        ),
        ToolSpec(
            name="declare_causal_assumptions",
            description=(
                "Declare CausalAssumptions required before causal estimation "
                "(treatment, outcome, confounders, estimand, unconfoundedness "
                "+ positivity acknowledgements). EDA is not a substitute. "
                "Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "outcome": {"type": "string"},
                    "confounders": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "estimand": {"type": "string", "enum": ["ATE"]},
                    "identification": {"type": "string", "enum": ["backdoor"]},
                    "acknowledge_unconfoundedness": {"type": "boolean"},
                    "acknowledge_positivity": {"type": "boolean"},
                    "allow_empty_confounders": {"type": "boolean"},
                },
                "required": [
                    "treatment",
                    "outcome",
                    "confounders",
                    "acknowledge_unconfoundedness",
                    "acknowledge_positivity",
                ],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="declare_causal_assumptions",
            read_only=False,
            catalog_operation="declare_causal_assumptions",
        ),
        ToolSpec(
            name="fit_causal",
            description=(
                "Fit train-only causal models under declared CausalAssumptions "
                "and estimate backdoor ATE. Backends: native (T-learner/IPW/AIPW), "
                "dowhy, econml when buildml[causal-industry] installed. Refuses "
                "without assumptions. Not causal discovery. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "dowhy", "econml"],
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "t_learner",
                            "ipw",
                            "aipw",
                            "backdoor_linear",
                            "backdoor_propensity_score",
                            "backdoor_propensity_weighting",
                            "dml",
                            "causal_forest",
                            "policy_tree",
                        ],
                    },
                    "bootstrap_samples": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "outcome_model": {"type": "string"},
                    "propensity_model": {"type": "string"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_causal",
            read_only=False,
            catalog_operation="fit_causal",
        ),
        ToolSpec(
            name="estimate_causal",
            description=(
                "Estimate ATE on a partition using fitted train nuisances."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "bootstrap_samples": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="estimate_causal",
            read_only=True,
            catalog_operation="estimate_causal",
        ),
        ToolSpec(
            name="evaluate_causal",
            description=(
                "Holdout nuisance predictive checks + ATE. Not proof of "
                "identification; CausalAssumptions remain required."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "bootstrap_samples": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_causal",
            read_only=True,
            catalog_operation="evaluate_causal",
        ),
        ToolSpec(
            name="refute_causal",
            description=(
                "Refutation / sensitivity disclosure. Native: placebo_treatment, "
                "random_confounder. DoWhy backend adds random_common_cause, "
                "add_unobserved_common_cause, data_subset, placebo_outcome."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "placebo_treatment",
                            "random_confounder",
                            "random_common_cause",
                            "add_unobserved_common_cause",
                            "data_subset",
                            "placebo_outcome",
                        ],
                    },
                    "random_state": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="refute_causal",
            read_only=True,
            catalog_operation="refute_causal",
        ),
        ToolSpec(
            name="save_causal_bundle",
            description=(
                "Persist the active CausalPlan as buildml.causal_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_causal_bundle",
            read_only=False,
            catalog_operation="save_causal_bundle",
        ),
        ToolSpec(
            name="load_causal_bundle",
            description=(
                "Load a buildml.causal_bundle.v1 CausalPlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_causal_bundle",
            read_only=False,
            catalog_operation="load_causal_bundle",
        ),
        ToolSpec(
            name="set_graph",
            description=(
                "Attach an edge list for Graph ML. Session rows are nodes; "
                "node_id_col must uniquely match edge endpoints. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "description": "List of [source, target] pairs.",
                        "items": {
                            "type": "array",
                            "items": {},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "source_col": {"type": "string"},
                    "target_col": {"type": "string"},
                    "node_id_col": {"type": "string"},
                    "directed": {"type": "boolean"},
                },
                "required": ["edges"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="set_graph",
            read_only=False,
            catalog_operation="set_graph",
        ),
        ToolSpec(
            name="fit_graph",
            description=(
                "Fit graph node classification: classical NetworkX+sklearn "
                "(buildml[graph]), pure-Torch GCN (buildml[torch]), or PyG "
                "GCN/GraphSAGE/GAT (buildml[graph-pyg]). "
                "Default inductive train-induced subgraph. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["classical", "gcn", "pyg"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["inductive", "transductive"],
                    },
                    "classical_estimator": {
                        "type": "string",
                        "enum": ["logistic_regression", "random_forest"],
                    },
                    "pyg_model": {
                        "type": "string",
                        "enum": ["gcn", "graphsage", "gat"],
                    },
                    "epochs": {"type": "integer"},
                    "hidden_dim": {"type": "integer"},
                    "heads": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "include_graph_metrics": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_graph",
            read_only=False,
            catalog_operation="fit_graph",
        ),
        ToolSpec(
            name="predict_graph",
            description="Predict node labels with the fitted GraphPlan.",
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_graph",
            read_only=True,
            catalog_operation="predict_graph",
        ),
        ToolSpec(
            name="evaluate_graph",
            description=(
                "Holdout node-classification metrics under the plan's "
                "inductive/transductive edge filter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_graph",
            read_only=True,
            catalog_operation="evaluate_graph",
        ),
        ToolSpec(
            name="save_graph_bundle",
            description=(
                "Persist the active GraphPlan as buildml.graph_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_graph_bundle",
            read_only=False,
            catalog_operation="save_graph_bundle",
        ),
        ToolSpec(
            name="load_graph_bundle",
            description=(
                "Load a buildml.graph_bundle.v1 GraphPlan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_graph_bundle",
            read_only=False,
            catalog_operation="load_graph_bundle",
        ),
        ToolSpec(
            name="fit_symbolic",
            description=(
                "Compile or induce tabular if-then rules on Session train "
                "(sklearn tree/list or industry skope-rules/imodels when installed). "
                "Not Prolog/Z3/AGI. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry"],
                    },
                    "source": {
                        "type": "string",
                        "enum": ["declared", "decision_tree", "decision_list"],
                    },
                    "method": {
                        "type": "string",
                        "enum": ["skope_rules", "rulefit", "boosted_rules"],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "max_depth": {"type": "integer"},
                    "min_samples_leaf": {"type": "integer"},
                    "max_rules": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                    "verify_constraints": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_symbolic",
            read_only=False,
            catalog_operation="fit_symbolic",
        ),
        ToolSpec(
            name="evaluate_symbolic",
            description=(
                "Holdout metrics and rule coverage for the SymbolicPlan. "
                "Holdout never used to induce rules."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_symbolic",
            read_only=True,
            catalog_operation="evaluate_symbolic",
        ),
        ToolSpec(
            name="predict_symbolic",
            description=(
                "Predict with the symbolic rule base; optional rule-firing traces."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "return_traces": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_symbolic",
            read_only=True,
            catalog_operation="predict_symbolic",
        ),
        ToolSpec(
            name="fit_neuro_symbolic",
            description=(
                "Fit sklearn or lite torch + symbolic hybrid (constraint_overlay / "
                "rules_as_features / constraint_repair) on Session train. "
                "Not a deep neuro-symbolic research platform. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "torch"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": [
                            "constraint_overlay",
                            "rules_as_features",
                            "constraint_repair",
                        ],
                    },
                    "base_estimator": {
                        "type": "string",
                        "enum": [
                            "logistic_regression",
                            "ridge",
                            "random_forest",
                            "decision_tree",
                            "concept_bottleneck_lite",
                            "neural_additive_lite",
                        ],
                    },
                    "torch_method": {
                        "type": "string",
                        "enum": ["concept_bottleneck_lite", "neural_additive_lite"],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "rule_source": {
                        "type": "string",
                        "enum": ["declared", "decision_tree", "decision_list"],
                    },
                    "soft_strength": {"type": "number"},
                    "max_depth": {"type": "integer"},
                    "max_rules": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "torch_epochs": {"type": "integer"},
                    "device": {"type": "string"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_neuro_symbolic",
            read_only=False,
            catalog_operation="fit_neuro_symbolic",
        ),
        ToolSpec(
            name="evaluate_neuro_symbolic",
            description=(
                "Holdout metrics for the neuro-symbolic hybrid (coverage / repair_rate)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_neuro_symbolic",
            read_only=True,
            catalog_operation="evaluate_neuro_symbolic",
        ),
        ToolSpec(
            name="predict_neuro_symbolic",
            description=(
                "Hybrid predict with neural outputs and symbolic overlay/repair traces."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "return_traces": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_neuro_symbolic",
            read_only=True,
            catalog_operation="predict_neuro_symbolic",
        ),
        ToolSpec(
            name="save_symbolic_bundle",
            description=(
                "Persist SymbolicPlan / NeuroSymbolicPlan as "
                "buildml.symbolic_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_symbolic_bundle",
            read_only=False,
            catalog_operation="save_symbolic_bundle",
        ),
        ToolSpec(
            name="load_symbolic_bundle",
            description=(
                "Load a buildml.symbolic_bundle.v1 plan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_symbolic_bundle",
            read_only=False,
            catalog_operation="load_symbolic_bundle",
        ),
        ToolSpec(
            name="fit_cbr",
            description=(
                "Build a tabular case memory from Session train "
                "(sklearn/industry/embedding/torch backends). Case→solution CBR — "
                "not RAG. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "embedding", "torch"],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["euclidean", "manhattan", "cosine", "mixed"],
                    },
                    "reuse": {
                        "type": "string",
                        "enum": [
                            "majority",
                            "distance_weighted",
                            "local_mean",
                            "local_ridge",
                        ],
                    },
                    "adapt": {
                        "type": "string",
                        "enum": ["none", "offset"],
                    },
                    "k": {"type": "integer"},
                    "text_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "text_model_name": {"type": "string"},
                    "standardize": {"type": "boolean"},
                    "distance_eps": {"type": "number"},
                    "random_state": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_cbr",
            read_only=False,
            catalog_operation="fit_cbr",
        ),
        ToolSpec(
            name="retrieve_cases",
            description=(
                "Retrieve k nearest cases from the train-built case memory "
                "(no reuse / no memory update)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "embedding", "torch"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="retrieve_cases",
            read_only=True,
            catalog_operation="retrieve_cases",
        ),
        ToolSpec(
            name="predict_cbr",
            description=(
                "Predict via retrieve + reuse; optional case-influence traces."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                    "return_traces": {"type": "boolean"},
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "embedding", "torch"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_cbr",
            read_only=True,
            catalog_operation="predict_cbr",
        ),
        ToolSpec(
            name="evaluate_cbr",
            description=(
                "Holdout metrics for CBR (case memory not updated from holdout)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_cbr",
            read_only=True,
            catalog_operation="evaluate_cbr",
        ),
        ToolSpec(
            name="retain_cbr",
            description=(
                "Retain newly labeled cases into case memory. Requires "
                "source_disclosure. Refuses Session validation/test indices. "
                "Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_disclosure": {
                        "type": "string",
                        "description": "Required provenance for new labels.",
                    },
                    "solution_column": {"type": "string"},
                    "allow_overlap_with_train": {"type": "boolean"},
                },
                "required": ["source_disclosure"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="retain_cbr",
            read_only=False,
            catalog_operation="retain_cbr",
        ),
        ToolSpec(
            name="save_cbr_bundle",
            description=(
                "Persist CbrPlan as buildml.cbr_bundle.v1."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_cbr_bundle",
            read_only=False,
            catalog_operation="save_cbr_bundle",
        ),
        ToolSpec(
            name="load_cbr_bundle",
            description=(
                "Load a buildml.cbr_bundle.v1 plan into the Session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_cbr_bundle",
            read_only=False,
            catalog_operation="load_cbr_bundle",
        ),
        ToolSpec(
            name="fit_imitation",
            description=(
                "Fit behavioral cloning from Session train demonstrations "
                "(state→action). Not inverse RL / robotics. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry"],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "estimator": {
                        "type": "string",
                        "enum": [
                            "logistic_regression",
                            "hist_gradient_boosting",
                            "ridge",
                            "hist_gradient_boosting_regressor",
                        ],
                    },
                    "method": {
                        "type": "string",
                        "enum": ["bc_mlp", "gail_lite"],
                    },
                    "action_column": {"type": "string"},
                    "env_id": {"type": "string"},
                    "n_epochs": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_imitation",
            read_only=False,
            catalog_operation="fit_imitation",
        ),
        ToolSpec(
            name="predict_imitation_action",
            description="Predict actions under the fitted behavioral cloning policy.",
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_imitation_action",
            read_only=True,
            catalog_operation="predict_imitation_action",
        ),
        ToolSpec(
            name="evaluate_imitation",
            description=(
                "Holdout imitation metrics vs demonstration actions "
                "(policy not updated from holdout)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_imitation",
            read_only=True,
            catalog_operation="evaluate_imitation",
        ),
        ToolSpec(
            name="save_imitation_bundle",
            description="Persist ImitationPlan as buildml.imitation_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_imitation_bundle",
            read_only=False,
            catalog_operation="save_imitation_bundle",
        ),
        ToolSpec(
            name="load_imitation_bundle",
            description="Load a buildml.imitation_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_imitation_bundle",
            read_only=False,
            catalog_operation="load_imitation_bundle",
        ),
        ToolSpec(
            name="fit_rl",
            description=(
                "Fit contextual bandit (core), Gymnasium REINFORCE-lite (buildml[rl]), "
                "or SB3 PPO/DQN/A2C (buildml[rl-industry]). Offline bandit metrics "
                "disclosed. Not MuJoCo/robotics. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "native", "industry"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["contextual_bandit", "gym_reinforce", "gym_sb3"],
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": [
                            "linucb",
                            "epsilon_greedy",
                            "softmax",
                            "ppo",
                            "dqn",
                            "a2c",
                        ],
                    },
                    "action_column": {"type": "string"},
                    "reward_column": {"type": "string"},
                    "alpha": {"type": "number"},
                    "epsilon": {"type": "number"},
                    "temperature": {"type": "number"},
                    "random_state": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                    "env_id": {"type": "string"},
                    "n_episodes": {"type": "integer"},
                    "max_steps": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "gamma": {"type": "number"},
                    "total_timesteps": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_rl",
            read_only=False,
            catalog_operation="fit_rl",
        ),
        ToolSpec(
            name="act_rl",
            description=(
                "Choose actions under the fitted RL policy "
                "(bandit partition or gym observations)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "deterministic": {"type": "boolean"},
                    "random_state": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="act_rl",
            read_only=True,
            catalog_operation="act_rl",
        ),
        ToolSpec(
            name="evaluate_rl",
            description=(
                "Evaluate RL: offline DM/IPS for bandits, or Gymnasium episode "
                "returns (buildml[rl])."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "n_episodes": {"type": "integer"},
                    "max_steps": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "deterministic": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_rl",
            read_only=True,
            catalog_operation="evaluate_rl",
        ),
        ToolSpec(
            name="save_rl_bundle",
            description="Persist RlPlan as buildml.rl_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_rl_bundle",
            read_only=False,
            catalog_operation="save_rl_bundle",
        ),
        ToolSpec(
            name="load_rl_bundle",
            description="Load a buildml.rl_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_rl_bundle",
            read_only=False,
            catalog_operation="load_rl_bundle",
        ),
        ToolSpec(
            name="fit_tda",
            description=(
                "Fit topological features on train (native ripser or giotto-tda "
                "backend) with optional sklearn head. Requires buildml[tda] or "
                "buildml[tda-industry]. Not a Mapper research suite. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "giotto"],
                    },
                    "vectorization": {
                        "type": "string",
                        "enum": [
                            "persistence_image",
                            "landscape",
                            "silhouette",
                            "betti_curve",
                            "persistence_landscape",
                        ],
                    },
                    "knn": {"type": "integer"},
                    "n_bins": {"type": "integer"},
                    "n_layers": {"type": "integer"},
                    "standardize": {"type": "boolean"},
                    "head": {
                        "type": "string",
                        "enum": [
                            "logistic_regression",
                            "random_forest",
                            "ridge",
                            "hist_gradient_boosting",
                            "none",
                        ],
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression"],
                    },
                    "random_state": {"type": "integer"},
                    "prefer_reduce_components": {"type": "boolean"},
                    "max_points_guard": {"type": "integer"},
                    "subsample_strategy": {
                        "type": "string",
                        "enum": ["error", "random", "stratified"],
                    },
                    "mapper": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_tda",
            read_only=False,
            catalog_operation="fit_tda",
        ),
        ToolSpec(
            name="transform_tda",
            description=(
                "Transform a partition with the frozen train-fitted TDA pipeline "
                "(buildml[tda])."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="transform_tda",
            read_only=True,
            catalog_operation="transform_tda",
        ),
        ToolSpec(
            name="predict_tda",
            description="Predict with the optional TDA supervised head.",
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_tda",
            read_only=True,
            catalog_operation="predict_tda",
        ),
        ToolSpec(
            name="evaluate_tda",
            description=(
                "Score the TDA head on a holdout partition (frozen train pipeline); "
                "optional persim Wasserstein/bottleneck diagram distances."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["native", "giotto"],
                    },
                    "compare_diagram_distances": {"type": "boolean"},
                    "diagram_distance_metric": {
                        "type": "string",
                        "enum": ["wasserstein", "bottleneck"],
                    },
                    "diagram_distance_dim": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_tda",
            read_only=True,
            catalog_operation="evaluate_tda",
        ),
        ToolSpec(
            name="save_tda_bundle",
            description="Persist TdaPlan as buildml.tda_bundle.v2.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_tda_bundle",
            read_only=False,
            catalog_operation="save_tda_bundle",
        ),
        ToolSpec(
            name="load_tda_bundle",
            description="Load a buildml.tda_bundle.v2 plan into the Session (v1 compatible).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_tda_bundle",
            read_only=False,
            catalog_operation="load_tda_bundle",
        ),
        ToolSpec(
            name="fit_recommender",
            description=(
                "Fit a recommender on train user–item interactions (item/user "
                "kNN, SVD/NMF, content, or industry ALS/BPR/LightFM when "
                "recommenders-industry extra installed). Requires user_column "
                "and item_column. Not RAG and not EDA Recommendation Findings. "
                "Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": [
                            "item_knn",
                            "user_knn",
                            "svd",
                            "nmf",
                            "content",
                            "als",
                            "bpr",
                            "lightfm",
                        ],
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "implicit", "lightfm"],
                    },
                    "user_column": {"type": "string"},
                    "item_column": {"type": "string"},
                    "rating_column": {"type": "string"},
                    "feedback": {
                        "type": "string",
                        "enum": ["explicit", "implicit"],
                    },
                    "n_neighbors": {"type": "integer"},
                    "n_factors": {"type": "integer"},
                    "min_rating": {"type": "number"},
                    "item_feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "user_feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "n_iterations": {"type": "integer"},
                    "lightfm_epochs": {"type": "integer"},
                    "cold_start": {
                        "type": "string",
                        "enum": ["popularity", "skip"],
                    },
                    "random_state": {"type": "integer"},
                },
                "required": ["user_column", "item_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_recommender",
            read_only=False,
            catalog_operation="fit_recommender",
        ),
        ToolSpec(
            name="recommend",
            description=(
                "Top-K item recommendations from the frozen train recommender "
                "(known-item catalog)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                    "exclude_train_items": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="recommend",
            read_only=True,
            catalog_operation="recommend",
        ),
        ToolSpec(
            name="evaluate_recommender",
            description=(
                "Holdout ranking metrics (Precision@K, Recall@K, nDCG@K, MAP@K) "
                "for the frozen recommender."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_recommender",
            read_only=True,
            catalog_operation="evaluate_recommender",
        ),
        ToolSpec(
            name="save_recommender_bundle",
            description="Persist RecommenderPlan as buildml.recommender_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_recommender_bundle",
            read_only=False,
            catalog_operation="save_recommender_bundle",
        ),
        ToolSpec(
            name="load_recommender_bundle",
            description="Load a buildml.recommender_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_recommender_bundle",
            read_only=False,
            catalog_operation="load_recommender_bundle",
        ),
        ToolSpec(
            name="ranking_capability_matrix",
            description=(
                "Return honest tabular LTR backend/method capability matrix "
                "(sklearn fallback, industry GBDT rankers, torch listwise-lite). "
                "Distinguishes LTR from RAG and recommenders."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="ranking_capability_matrix",
            read_only=True,
            catalog_operation="ranking_capability_matrix",
        ),
        ToolSpec(
            name="fit_ranker",
            description=(
                "Fit tabular learning-to-rank on train query–item feature rows. "
                "Backends: sklearn pointwise/pairwise fallback; industry "
                "LightGBM LambdaRank / XGBoost rank:ndcg / CatBoost YetiRank "
                "(buildml[ranking-industry]); torch listwise-lite (buildml[torch]). "
                "Requires query_column and item_column. Prefer group_split on query id. "
                "Not RAG and not recommender CF. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                        "description": "LTR backend (see ranking_capability_matrix).",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "pointwise",
                            "pairwise",
                            "lambdarank_lgbm",
                            "rank_ndcg_xgb",
                            "yetirank_catboost",
                            "listwise_lite",
                        ],
                    },
                    "query_column": {"type": "string"},
                    "item_column": {"type": "string"},
                    "relevance_column": {"type": "string"},
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "pointwise_estimator": {
                        "type": "string",
                        "enum": ["ridge", "hgb"],
                    },
                    "pairwise_estimator": {
                        "type": "string",
                        "enum": ["ranksvm"],
                    },
                    "max_pairs_per_query": {"type": "integer"},
                    "relevance_threshold": {"type": "number"},
                    "alpha": {"type": "number"},
                    "C": {"type": "number"},
                    "n_estimators": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "hidden_dim": {"type": "integer"},
                    "epochs": {"type": "integer"},
                    "device": {"type": "string"},
                    "random_state": {"type": "integer"},
                },
                "required": ["query_column", "item_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_ranker",
            read_only=False,
            catalog_operation="fit_ranker",
        ),
        ToolSpec(
            name="rank",
            description=(
                "Order items for queries with the frozen tabular ranker "
                "(top-K item ids + scores per query)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="rank",
            read_only=True,
            catalog_operation="rank",
        ),
        ToolSpec(
            name="evaluate_ranker",
            description=(
                "Holdout per-query ranking metrics (nDCG@K, MAP@K, MRR@K) "
                "for the frozen tabular ranker. Distinct from RAG and recommender eval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                    "backend": {
                        "type": "string",
                        "enum": ["sklearn", "industry", "torch"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_ranker",
            read_only=True,
            catalog_operation="evaluate_ranker",
        ),
        ToolSpec(
            name="save_ranker_bundle",
            description="Persist RankerPlan as buildml.ranker_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_ranker_bundle",
            read_only=False,
            catalog_operation="save_ranker_bundle",
        ),
        ToolSpec(
            name="load_ranker_bundle",
            description="Load a buildml.ranker_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_ranker_bundle",
            read_only=False,
            catalog_operation="load_ranker_bundle",
        ),
        ToolSpec(
            name="fit_kg",
            description=(
                "Fit knowledge-graph embeddings on train (head, relation, tail) "
                "triples. Backends: native (numpy TransE/DistMult) or pykeen "
                "(RotatE/ComplEx when buildml[kg-industry] installed). Requires "
                "head_column, relation_column, tail_column. Not Graph ML "
                "node-classify, not Neo4j, not RAG. Write operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "pykeen"],
                    },
                    "method": {
                        "type": "string",
                        "enum": ["transe", "distmult", "rotate", "complex"],
                    },
                    "head_column": {"type": "string"},
                    "relation_column": {"type": "string"},
                    "tail_column": {"type": "string"},
                    "embedding_dim": {"type": "integer"},
                    "epochs": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                    "learning_rate": {"type": "number"},
                    "margin": {"type": "number"},
                    "neg_ratio": {"type": "integer"},
                    "norm": {"type": "string", "enum": ["l1", "l2"]},
                    "random_state": {"type": "integer"},
                },
                "required": ["head_column", "relation_column", "tail_column"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_kg",
            read_only=False,
            catalog_operation="fit_kg",
        ),
        ToolSpec(
            name="score_triples",
            description=(
                "Score complete (head, relation, tail) triples with the frozen "
                "KG embeddings (higher is better)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="score_triples",
            read_only=True,
            catalog_operation="score_triples",
        ),
        ToolSpec(
            name="predict_links",
            description=(
                "Predict missing KG link components (tail, head, or relation) "
                "with filtered top-K ranking over the train catalog."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["tail", "head", "relation"],
                    },
                    "heads": {"type": "array", "items": {}},
                    "relations": {"type": "array", "items": {}},
                    "tails": {"type": "array", "items": {}},
                    "k": {"type": "integer"},
                    "filtered": {"type": "boolean"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="predict_links",
            read_only=True,
            catalog_operation="predict_links",
        ),
        ToolSpec(
            name="query_kg",
            description=(
                "Symbolic neighbors / path / typed query over the train KG "
                "adjacency (not LLM, not Neo4j, not RAG)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["neighbors", "path", "typed"],
                    },
                    "entity": {},
                    "source": {},
                    "target": {},
                    "relation": {},
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                    },
                    "max_hops": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="query_kg",
            read_only=True,
            catalog_operation="query_kg",
        ),
        ToolSpec(
            name="evaluate_kg",
            description=(
                "Holdout filtered link-prediction metrics (MRR, Hits@1/3/K) "
                "for the frozen KG. Distinct from Graph ML and RAG eval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                    "k": {"type": "integer"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_kg",
            read_only=True,
            catalog_operation="evaluate_kg",
        ),
        ToolSpec(
            name="save_kg_bundle",
            description="Persist KgPlan as buildml.kg_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_kg_bundle",
            read_only=False,
            catalog_operation="save_kg_bundle",
        ),
        ToolSpec(
            name="load_kg_bundle",
            description="Load a buildml.kg_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_kg_bundle",
            read_only=False,
            catalog_operation="load_kg_bundle",
        ),
        ToolSpec(
            name="decision_capability_matrix",
            description=(
                "Honest capability matrix for decision-policy backends "
                "(native scipy/numpy, PuLP/OR-Tools MIP, CVXPY LP, XGB/calibrated "
                "thresholds). Read-only catalog."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="decision_capability_matrix",
            read_only=True,
            catalog_operation="decision_capability_matrix",
        ),
        ToolSpec(
            name="fit_decision_policy",
            description=(
                "Fit a decision policy on train/validation: threshold "
                "(wraps tune_threshold engine or XGB/calibrated when installed), "
                "cost_matrix, topk, knapsack (PuLP/OR-Tools MIP when installed), "
                "or lp_allocate (CVXPY when installed). Test tuning requires "
                "allow_test_tuning=True. Not a general OR platform."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": [
                            "threshold",
                            "cost_matrix",
                            "topk",
                            "knapsack",
                            "lp_allocate",
                        ],
                    },
                    "backend": {
                        "type": "string",
                        "enum": [
                            "native",
                            "pulp",
                            "ortools",
                            "cvxpy",
                            "calibrated",
                            "xgb",
                        ],
                        "description": (
                            "Solver/scorer backend (see decision_capability_matrix)."
                        ),
                    },
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                    "allow_test_tuning": {"type": "boolean"},
                    "fp_cost": {"type": "number"},
                    "fn_cost": {"type": "number"},
                    "tp_benefit": {"type": "number"},
                    "tn_benefit": {"type": "number"},
                    "cost_matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "class_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "capacity": {"type": "integer"},
                    "budget": {"type": "number"},
                    "score_source": {
                        "type": "string",
                        "enum": [
                            "model_proba",
                            "model_decision_function",
                            "column",
                        ],
                    },
                    "score_column": {"type": "string"},
                    "cost_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "id_column": {"type": "string"},
                    "knapsack_solver": {
                        "type": "string",
                        "enum": ["dp", "greedy"],
                    },
                    "objective": {
                        "type": "string",
                        "enum": [
                            "maximize_score",
                            "maximize_value",
                            "minimize_cost",
                        ],
                    },
                    "min_score": {"type": "number"},
                    "lp_max_fraction": {"type": "number"},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_decision_policy",
            read_only=False,
            catalog_operation="fit_decision_policy",
        ),
        ToolSpec(
            name="apply_decisions",
            description="Apply a frozen DecisionPlan to a partition.",
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test", "all"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="apply_decisions",
            read_only=True,
            catalog_operation="apply_decisions",
        ),
        ToolSpec(
            name="evaluate_decisions",
            description="Evaluate a frozen DecisionPlan on a holdout partition.",
            parameters={
                "type": "object",
                "properties": {
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_decisions",
            read_only=True,
            catalog_operation="evaluate_decisions",
        ),
        ToolSpec(
            name="save_decision_bundle",
            description="Persist DecisionPlan as buildml.decision_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_decision_bundle",
            read_only=False,
            catalog_operation="save_decision_bundle",
        ),
        ToolSpec(
            name="load_decision_bundle",
            description="Load a buildml.decision_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_decision_bundle",
            read_only=False,
            catalog_operation="load_decision_bundle",
        ),
        ToolSpec(
            name="synthetic_capability_matrix",
            description=(
                "Honest capability matrix for synthetic backends (native vs SDV) "
                "and evaluation paths (builtin fidelity/TSTR vs SDMetrics)."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="synthetic_capability_matrix",
            read_only=True,
            catalog_operation="synthetic_capability_matrix",
        ),
        ToolSpec(
            name="fit_synthesizer",
            description=(
                "Fit a train-only tabular synthesizer. Native: bootstrap, "
                "gaussian_copula, smote (buildml[imbalanced]). SDV industry: "
                "ctgan, tvae, copulagan (buildml[synthetic-industry]). "
                "Distinct from resample. Not differential privacy."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["native", "sdv"],
                        "description": "Backend (see synthetic_capability_matrix).",
                    },
                    "method": {
                        "type": "string",
                        "enum": [
                            "bootstrap",
                            "gaussian_copula",
                            "smote",
                            "ctgan",
                            "tvae",
                            "copulagan",
                        ],
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "random_state": {"type": "integer"},
                    "smooth_sigma": {"type": "number"},
                    "correlation_ridge": {"type": "number"},
                    "target_column": {"type": "string"},
                    "k_neighbors": {"type": "integer"},
                    "sampling_strategy": {},
                    "epochs": {"type": "integer", "description": "SDV training epochs."},
                    "batch_size": {"type": "integer", "description": "SDV batch size."},
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="fit_synthesizer",
            read_only=False,
            catalog_operation="fit_synthesizer",
        ),
        ToolSpec(
            name="sample_synthetic",
            description=(
                "Sample from a frozen synthesizer. Default returns a Frame; "
                "merge_mode=extend_train appends to train with provenance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "merge_mode": {
                        "type": "string",
                        "enum": ["none", "extend_train"],
                    },
                    "provenance_column": {"type": "string"},
                    "validate": {
                        "type": "boolean",
                        "description": "Run built-in validate_synthetic on sample.",
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="sample_synthetic",
            read_only=False,
            catalog_operation="sample_synthetic",
        ),
        ToolSpec(
            name="evaluate_synthetic",
            description=(
                "Evaluate a frozen synthesizer: fidelity (KS/TV/corr) or "
                "tstr (train-on-synthetic test-on-real). Not a privacy audit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fidelity", "tstr"],
                    },
                    "eval_backend": {
                        "type": "string",
                        "enum": ["auto", "builtin", "sdmetrics"],
                        "description": "Fidelity eval backend (SDMetrics when installed).",
                    },
                    "partition": {
                        "type": "string",
                        "enum": ["train", "validation", "test"],
                    },
                    "n_synthetic": {"type": "integer"},
                    "random_state": {"type": "integer"},
                    "estimator": {
                        "type": "string",
                        "enum": ["auto", "logistic", "ridge"],
                    },
                },
                "required": [],
            },
            confirm_policy=ConfirmPolicy.AUTO,
            session_method="evaluate_synthetic",
            read_only=True,
            catalog_operation="evaluate_synthetic",
        ),
        ToolSpec(
            name="save_synthetic_bundle",
            description="Persist SynthesizerPlan as buildml.synthetic_bundle.v1.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="save_synthetic_bundle",
            read_only=False,
            catalog_operation="save_synthetic_bundle",
        ),
        ToolSpec(
            name="load_synthetic_bundle",
            description="Load a buildml.synthetic_bundle.v1 plan into the Session.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Bundle directory."},
                },
                "required": ["path"],
            },
            confirm_policy=ConfirmPolicy.CONFIRM,
            session_method="load_synthetic_bundle",
            read_only=False,
            catalog_operation="load_synthetic_bundle",
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
