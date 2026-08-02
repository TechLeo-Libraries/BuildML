# Torch deep guide

> **Install (GitHub 2.x + Torch):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[torch]"
> # aliases: buildml[dl], buildml[audio]
> # ONNX checker: buildml[onnx]
> ```
> PyPI `buildml` is legacy 1.x. See [installation](../docs/installation.rst).

Optional deep learning on the **same Session** as classical ML. Classical
`Session.fit` stays the sklearn path; Torch uses `*_torch` methods and stores
results in `dl_*_result` properties.

Short on-ramp: [quickstart-torch](quickstart-torch.md). Speech-specific path:
[speech-asr-finetune](speech-asr-finetune.md). Pretrained hooks:
[pretrained-backbones](pretrained-backbones.md). Serve/export packs:
[serve-deploy](serve-deploy.md).

---

## Why a Session-native Torch path

1. **Shared roles and splits** — train/val/test membership stays authoritative.
2. **Train-only normalize / vocab** — loaders fit statistics on train partitions.
3. **History + explain** — Torch ops appear in the teaching catalog.
4. **Honest limits** — multimodal fusion and speech finetune-lite are alpha
   helpers, not foundation-model training products.

---

## Use case A — Tabular MLP (custom or built-in)

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=42)
)

# Optional classical prep first (mutates frame; disclosed on loaders).
# session.impute(strategy="median").scale(method="standard")

session.make_torch_loaders(batch_size=4, normalize=True, seed=42)
session.fit_torch(
    TinyMLP(),
    epochs=8,
    learning_rate=5e-3,
    device="cpu",
    early_stopping_patience=3,
    mixed_precision=False,  # AMP is CUDA-only
)

print(session.evaluate_torch(partition="validation").metrics)
print(session.evaluate_torch(partition="test").metrics)
print(session.torch_training_curve().disclosures)

bundle = session.save_torch_bundle("artifacts/torch_bundle")
```

Built-in MLP (omit module):

```python
session.make_torch_loaders()
session.fit_torch(epochs=5, device="auto", hidden=(64, 32), dropout=0.1)
```

---

## Use case B — Text / sequence loaders

```python
text_df = pd.DataFrame(
    {
        "text": [
            "approved quickly",
            "denied for risk",
            "manual review",
            "approved payroll",
            "denied fraud",
            "approved loyal",
            "denied late",
            "approved deposit",
        ],
        "y": [1, 0, 0, 1, 0, 1, 0, 1],
    }
)

text_session = (
    Session.ingest(text_df)
    .set_roles({"text": "feature", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
)
text_session.make_text_torch_loaders(text_column="text", max_len=32, max_vocab=500)
text_session.fit_torch(epochs=4, device="cpu")  # built-in embedding classifier
print(text_session.evaluate_torch(partition="test").metrics)
```

Vocab and length rules come from **train**. After a text fit, rebuilding tabular
loaders and calling `evaluate_torch` is refused — keep loader kind consistent.

---

## Use case C — Multimodal fusion (tabular + text)

Default built-in fusion (when `fit_torch` omits a module) uses **concat** late
fusion. Pass V also ships **gated** late fusion via
`build_multimodal_fusion(..., fusion="gated")` (aliases: `fusion_type`,
`fusion_mode`).

```python
from buildml.dl.multimodal import build_multimodal_fusion

mm_df = pd.DataFrame(
    {
        "x1": [0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8],
        "text": [
            "low risk",
            "high risk",
            "low risk",
            "high risk",
            "medium",
            "high risk",
            "low risk",
            "medium",
        ],
        "y": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

mm = (
    Session.ingest(mm_df)
    .set_roles({"x1": "feature", "text": "feature", "y": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)
bundle = mm.make_multimodal_torch_loaders(text_column="text")
# Built-in concat fusion:
# mm.fit_torch(epochs=5, device="cpu", mixed_precision=False)

# Explicit gated fusion (Pass V):
contract = bundle.multimodal_contract
gated = build_multimodal_fusion(contract, fusion="gated")
mm.fit_torch(gated, epochs=5, device="cpu", mixed_precision=False)
mm.export_torch("artifacts/mm.ts.pt", format="torchscript")
```

### Frozen `multimodal_preprocess` restore

Trainer bundles may persist train-fit multimodal stats (normalize mean/std,
vocab, image/audio rates/layout) as `multimodal_preprocess`.
`load_torch_bundle` restores that meta for inspection but does **not** rebuild
DataLoaders. To rebuild loaders with the frozen stats:

```python
# After fit_torch / load_torch_bundle with multimodal_preprocess present:
mm.make_multimodal_torch_loaders(
    text_column="text",
    use_saved_preprocess=True,  # reuses dl_train_result.multimodal_preprocess
)
# Or pass an explicit contract/dict:
# mm.make_multimodal_torch_loaders(
#     text_column="text",
#     preprocess=mm.dl_train_result.multimodal_preprocess,
# )
```

Do not pass both `preprocess=` and `use_saved_preprocess=True`.

---

## Use case D — Image multimodal

```python
# image_column: filesystem path or array cell; train-only normalize stats
# img.make_image_multimodal_torch_loaders(
#     image_column="image", image_size=(32, 32), normalize_images=True
# )
# img.fit_torch(epochs=5, device="cpu")
```

Small CNN branch for fusion — not a full vision FM product. Paths need readable
files in your environment; array cells work for CI-style tests.

---

## Use case E — Audio multimodal

```python
# pip install "buildml[torch]"  # includes soundfile
# aud.make_audio_multimodal_torch_loaders(
#     audio_column="audio",
#     audio_sample_rate=16000,
#     audio_max_samples=16000,
#     normalize_audio=True,
# )
# aud.fit_torch(epochs=5, device="cpu")
```

Short clips are repeat-padded to `audio_max_samples` so global pooling stays
informative without a lengths tensor in forward/export. This is **not** ASR;
see [speech](speech-asr-finetune.md).

---

## Use case F — Fold-local CV, search, nested

```python
cv = session.cross_validate_torch(n_folds=3, epochs=2)
print(cv)

search = session.search_torch(
    param_grid={"learning_rate": [1e-3, 1e-2], "hidden": [(32,), (64, 32)]},
    n_folds=2,
    epochs=2,
)
print(search)

nested = session.nested_cv_torch(
    param_grid={"learning_rate": [1e-3, 1e-2], "hidden": [(32,), (64,)]},
    outer_cv=3,
    inner_cv=2,
    epochs=2,
)
print(nested.mean_metrics, getattr(nested, "consensus_params", None))
```

Normalize stats are fold-local inside these APIs. Do **not** tune early stopping
or architecture on Session test. Classical Session-global plans are **not**
automatically refit inside `cross_validate_torch`.

---

## Use case G — AMP, DDP, export, reload

```python
# AMP (CUDA only; ignored/safe on CPU when mixed_precision=False)
# session.fit_torch(TinyMLP(), epochs=5, device="cuda", mixed_precision=True)

# Single-node DDP refuses 1-GPU unless allow_cpu_ddp=True for experiments
# session.fit_torch_ddp(lambda: TinyMLP(), epochs=5, world_size=2, allow_cpu_ddp=True)

# Multi-node under torchrun:
# torchrun --nnodes=2 --nproc_per_node=2 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py
# session.fit_torch_ddp(module_factory, multi_node=True, epochs=5)

session.export_torch("artifacts/model.ts.pt", format="torchscript")
# session.export_torch("artifacts/model.onnx", format="onnx")  # optional buildml[onnx]

restored = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=42)
)
restored.load_torch_bundle(bundle, TinyMLP(), map_location="cpu")
restored.make_torch_loaders(batch_size=4, normalize=True, seed=42)
restored.evaluate_torch(partition="test")
restored.fit_torch(TinyMLP(), epochs=2, resume=True, device="cpu")
```

`emit_k8s_ddp_job`, `pack_torchserve`, and `prepare_tensorrt_export` are
**recipe emitters** — see [serve-deploy](serve-deploy.md).

---

## Classical plans vs Torch loaders

| Pattern | Meaning |
| --- | --- |
| Prep then `make_torch_loaders` | Loaders see mutated frame; disclosed |
| `apply_plans=True` on loaders | Re-apply fitted classical plans (no refit) |
| Fold-local classical refit in Torch CV | **Not automatic** |

Prefer keeping classical CV ([leakage guide](leakage-cv-recipes.md)) and Torch
CV as separate honesty protocols unless you know the interaction.

---

## Artifacts

| Artifact | Notes |
| --- | --- |
| Checkpoint | No Torch weights |
| `buildml.torch_bundle.v1` | `meta.json` + `trainer.pt`; load ≠ rebuild loaders |
| TorchScript / ONNX | Escape hatch via `export_torch` |

---

## Failure modes / limits

- **CPU-first CI** — GPU not a PR merge gate.
- **No Polars zero-copy** into DataLoaders.
- **Wrong loader kind after text/multimodal/speech fit** → `ValidationError`.
- **DDP with 1 GPU** refused unless `allow_cpu_ddp=True`.
- **Not** Whisper-scale FM pretrain — see speech refuse API.
- Multimodal image/audio are honest alpha fusion helpers (`concat` / `gated`).
- `use_saved_preprocess=True` without prior `multimodal_preprocess` meta →
  `ValidationError`.

---

## Related

- [Speech](speech-asr-finetune.md)
- [Pretrained](pretrained-backbones.md)
- [Serve & deploy](serve-deploy.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
- [AI tools](ai-tools-operator-patterns.md) (Torch tools on the allowlist)
