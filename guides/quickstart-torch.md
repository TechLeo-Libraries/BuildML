# Torch quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout), then
> the Torch extra. See [installation](../docs/installation.rst).

Optional Torch path on the same `Session` as classical ML: tabular loaders,
built-in MLP, text/sequence loaders, fold-local CV, history, and explain.
Install the Torch extra; core `import buildml` never requires it.

**Go deeper:** [Torch deep](torch-deep.md) ·
[Speech ASR + classify](speech-asr-finetune.md) ·
[Pretrained backbones](pretrained-backbones.md) ·
[Serve & deploy](serve-deploy.md) ·
[Artifacts](artifacts-checkpoints-bundles.md).

```bash
# After a GitHub / editable 2.x install:
pip install "buildml[torch]"
# alias: pip install "buildml[dl]"
# or: pip install "buildml[torch] @ git+https://github.com/TechLeo-Libraries/BuildML.git"
```

Classical `Session.fit` stays the default sklearn path. Torch methods use the
`*_torch` prefix and store results in `session.dl_train_result`.

```python
import pandas as pd
import torch
from torch import nn

from buildml import Session

frame = pd.DataFrame(
    {
        "a": [0.1, 0.4, 0.2, 0.8, 0.3, 0.7, 0.5, 0.9, 0.15, 0.65],
        "b": [1.0, 0.2, 0.9, 0.1, 0.8, 0.3, 0.6, 0.4, 0.75, 0.25],
        "label": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    }
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


session = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(
        test_size=0.25,
        validation_size=0.25,
        stratify=True,
        random_state=42,
    )
)

# Optional classical prep first (not auto-applied before loaders).
# session.impute(strategy="median").scale(method="standard")

session.make_torch_loaders(batch_size=4, normalize=True, seed=42)
session.fit_torch(
    TinyMLP(),
    epochs=6,
    learning_rate=5e-3,
    device="cpu",
    early_stopping_patience=3,
    scheduler="none",
)

# Prefer validation while iterating; reserve test for a fixed recipe.
validation = session.evaluate_torch(partition="validation")
test = session.evaluate_torch(partition="test")
print(test.metrics)

curve = session.torch_training_curve()
print(curve.disclosures)

bundle = session.save_torch_bundle("artifacts/torch_bundle")
```

Reload and optionally resume:

```python
restored = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(
        test_size=0.25,
        validation_size=0.25,
        stratify=True,
        random_state=42,
    )
)
restored.load_torch_bundle(bundle, TinyMLP(), map_location="cpu")
restored.make_torch_loaders(batch_size=4, normalize=True, seed=42)
restored.evaluate_torch(partition="test")
# Additional epochs; optimizer/scheduler state restored when compatible.
restored.fit_torch(TinyMLP(), epochs=2, resume=True, device="cpu")
```

Explain catalog coverage:

```python
before = session.explain("fit_torch", moment="before")
print(before.operation, before.prerequisites)
```

## Artifacts

| Artifact | Schema | Contains | Does not contain |
| --- | --- | --- | --- |
| Session checkpoint | existing checkpoint formats | data, roles, splits, history | Torch weights / optimizer |
| Torch trainer bundle | `buildml.torch_bundle.v1` | weights, optimizer (+ scheduler), config, history, feature contract, optional multimodal_preprocess (image/audio stats, rates, layout) | dataset rows, split indices; load does not rebuild DataLoaders |

Layout: `<path>/meta.json` + `<path>/trainer.pt`.

## Built-in models and text path

```python
# Tabular happy path — omit module to use the built-in MLP
session.make_torch_loaders()
session.fit_torch(epochs=5, device="auto")  # builds TabularMLP from the contract

# Fold-local CV (not nested hyperparameter search)
cv = session.cross_validate_torch(n_folds=3, epochs=2)

# Text / sequence modality
text_session.make_text_torch_loaders(text_column="text")
text_session.fit_torch(epochs=3)  # builds embedding text classifier
```

## Nested search and multimodal (Pass G)

```python
# Nested Torch HPO (outer estimate after inner search; fold-local normalize)
nested = session.nested_cv_torch(
    param_grid={"learning_rate": [1e-3, 1e-2], "hidden": [(32,), (64, 32)]},
    outer_cv=3,
    inner_cv=2,
    epochs=2,
)
print(nested.mean_metrics, nested.consensus_params)

# Tabular + text fusion
mm = (
    Session.ingest(df)
    .set_roles({"x1": "feature", "text": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)
mm.make_multimodal_torch_loaders(text_column="text")
mm.fit_torch(epochs=5, mixed_precision=False)  # AMP is CUDA-only
mm.export_torch("model.ts.pt", format="torchscript")

# Image multimodal (path or array column ⊕ tabular and/or text)
img = (
    Session.ingest(df_with_images)
    .set_roles({"x1": "feature", "image": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)
img.make_image_multimodal_torch_loaders(
    image_column="image", image_size=(32, 32), normalize_images=True
)
img.fit_torch(epochs=5, device="cpu")

# Audio multimodal (path or waveform column ⊕ tabular and/or text and/or image)
aud = (
    Session.ingest(df_with_audio)
    .set_roles({"x1": "feature", "audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)
aud.make_audio_multimodal_torch_loaders(
    audio_column="audio",
    audio_sample_rate=16000,
    audio_max_samples=16000,
    normalize_audio=True,
)
aud.fit_torch(epochs=5, device="cpu")

# Speech ASR (stub is CI-safe) + classify finetune-lite / domain adapt
# pip install "buildml[speech]" for transformers Whisper-class backend
speech = (
    Session.ingest(df_with_audio)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
)
asr = speech.transcribe_speech(audio_column="audio", backend="stub")
speech.domain_adapt_speech_torch(epochs=5, device="cpu", audio_column="audio")
# speech.refuse_speech_foundation_pretrain()  # honest refuse for FM-from-scratch asks

# Pretrained backbone hooks (mock weights = CI-safe; not a full zoo product)
# pip install "buildml[pretrained]"  # vision+speech extras
# backbone = session.load_pretrained_backbone("vision", "resnet18", weights="mock")

# Multi-node DDP (launch under torchrun; each process runs this)
# torchrun --nnodes=2 --nproc_per_node=2 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py
# session.fit_torch_ddp(module_factory, multi_node=True, epochs=5)
# session.emit_k8s_ddp_job("job.yaml", nnodes=2)  # template only — not live multi-cluster

# Managed local serving (pip install "buildml[serve]")
# classical.save_pipeline("bundle/")
# classical.serve_bundle("bundle/", kind="pipeline", api_keys=["dev-key"])
# or: buildml-serve --bundle bundle/ --kind pipeline --api-key dev-key
# Pack helpers (operator runs TorchServe / trtexec):
# session.export_torch("model.ts.pt"); session.pack_torchserve("torchserve_dir/")
# session.export_torch("model.onnx", format="onnx"); session.prepare_tensorrt_export("trt_plan/")
```

## Known limits (honest)

- **CPU-first merge gate.** CI runs Torch on CPU (Python 3.11–3.12). CUDA/MPS
  are supported when available with explicit fallback warnings; GPU CI is not a
  PR blocker.
- **Tabular + text + image + audio multimodal in scope.** Built-in MLP, text
  classifier, and fusion (small CNN image branch + small 1D-CNN audio branch)
  cover the happy path; custom `nn.Module` still works. Audio multimodal fusion
  is honest alpha — not a speech FM. For ASR / speech classify finetune-lite see
  `transcribe_speech` / `fit_speech_torch` (`buildml[speech]`). Short clips are
  repeat-padded to `audio_max_samples` so global pooling stays informative
  without a lengths tensor in forward/export. Trainer bundles may store frozen
  multimodal preprocess meta; `load_torch_bundle` restores it for inspection but
  does not rebuild loaders.
- **Materialization.** Partition rows become tensors via the current Session
  frame (Pandas/NumPy bridge). No Polars/DuckDB zero-copy into DataLoaders.
- **Classical plans.** Session impute/encode/scale mutate the frame and are
  disclosed on loaders; `apply_plans=True` re-applies fitted plans (no refit).
  Fold-local classical plan refit is not automatic inside `cross_validate_torch`.
- **CV + nested search.** Use `cross_validate_torch` for fold-local estimates and
  `nested_cv_torch` / `search_torch` for hyperparameter selection. Do not tune
  early stop on test.
- **DDP / export / serve / packs.** `fit_torch_ddp` supports single-node spawn and
  torchrun multi-node (`multi_node=True`). `emit_k8s_ddp_job` writes Job YAML
  templates only (not live multi-cluster orchestration). `export_torch` is an
  alpha TorchScript/ONNX escape hatch; `pack_torchserve` /
  `prepare_tensorrt_export` write operator-owned recipes (not a cloud). Managed
  local serving is `buildml[serve]` / `Session.serve_bundle` / `buildml-serve`
  (localhost; optional API-key middleware — still not managed IAM).
- **Pretrained hooks vs FM pretrain.** `load_pretrained_backbone` loads curated
  ResNet/ViT/Wav2Vec/Whisper-encoder hooks (`weights=mock` in CI). BuildML does
  **not** train Whisper-scale foundation models from scratch.
- **RAG / AI** are separate domains (`rag_*`, `ai_*`) that can call Torch tools.

See [glossary](glossary.md).
