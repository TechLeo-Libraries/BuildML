# Torch quickstart

Optional Torch path on the same `Session` as classical ML: tabular loaders,
built-in MLP, text/sequence loaders, fold-local CV, history, and explain.
Install the Torch extra; core `import buildml` never requires it.

```bash
pip install "buildml[torch]"
# alias: pip install "buildml[dl]"
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
| Torch trainer bundle | `buildml.torch_bundle.v1` | weights, optimizer (+ scheduler), config, history, feature contract | dataset rows, split indices |

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
```

## Known limits (honest)

- **CPU-first merge gate.** CI runs Torch on CPU (Python 3.11–3.12). CUDA/MPS
  are supported when available with explicit fallback warnings; GPU CI is not a
  PR blocker.
- **Tabular + text + image + audio multimodal in scope.** Built-in MLP, text
  classifier, and fusion (small CNN image branch + small 1D-CNN audio branch)
  cover the happy path; custom `nn.Module` still works. Audio is honest alpha
  fusion — not a speech foundation-model product.
- **Materialization.** Partition rows become tensors via the current Session
  frame (Pandas/NumPy bridge). No Polars/DuckDB zero-copy into DataLoaders.
- **Classical plans.** Session impute/encode/scale mutate the frame and are
  disclosed on loaders; `apply_plans=True` re-applies fitted plans (no refit).
  Fold-local classical plan refit is not automatic inside `cross_validate_torch`.
- **CV + nested search.** Use `cross_validate_torch` for fold-local estimates and
  `nested_cv_torch` / `search_torch` for hyperparameter selection. Do not tune
  early stop on test.
- **DDP / export.** `fit_torch_ddp` is single-node only; `export_torch` is an
  alpha TorchScript/ONNX escape hatch, not a managed serving product.
- **RAG / AI** are separate domains (`rag_*`, `ai_*`) that can call Torch tools.

See [glossary](glossary.md).
