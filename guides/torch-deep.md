# Torch on the same Session

```bash
pip install "buildml[torch]"
# aliases: buildml[dl], buildml[audio]
# ASR (not stub): pip install "buildml[speech]"
# ONNX checker: pip install "buildml[onnx]"
```

Same rows, roles, and split as classical ML. `session.dl.fit` trains
a Torch module on loaders you built from that split. It does **not**
replace `session.fit`. Classical sklearn stays on `session.fit`.
Torch results live on `session.dl.train_result` and friends.

Loaders need a split. Train-only normalize and vocab come from the
train partition. Omit `module` and you get a built-in MLP
(`hidden=(64, 32)`, `dropout=0.1`). Device default is `"auto"`.
Epochs default to 5. Zoo backbones default to `weights="mock"`
(random init for plumbing). Speech ASR without `buildml[speech]` is
a disclosed stub: fingerprints, not transcripts. Foundation-model
pretrain is refused (`session.dl.refuse_speech_pretrain`).

You choose the module, device, and whether mock weights are enough.
The API refuses loaders without a split, mixing loader kinds after a
text/multimodal/speech fit, DDP on one GPU unless
`allow_cpu_ddp=True`, and FM-from-scratch speech pretrain.

Short on-ramp: [Torch quickstart](quickstart-torch.md). Speech:
[speech-asr-finetune](speech-asr-finetune.md). Backbones:
[pretrained-backbones](pretrained-backbones.md). Serve:
[serve-deploy](serve-deploy.md).

## Tabular MLP

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

# Optional classical prep first (mutates the frame; disclosed on loaders).
# session.impute(strategy="median").scale(method="standard")

session.dl.make_loaders(batch_size=4, normalize=True, seed=42)
session.dl.fit(
    TinyMLP(),
    epochs=8,
    learning_rate=5e-3,
    device="cpu",
    early_stopping_patience=3,
    mixed_precision=False,  # AMP is CUDA-only
)

print(session.dl.evaluate(partition="validation").metrics)
print(session.dl.evaluate(partition="test").metrics)
print(session.dl.training_curve().disclosures)

bundle = session.dl.save_bundle("artifacts/torch_bundle")
```

Built-in MLP (omit the module):

```python
session.dl.make_loaders()
session.dl.fit(epochs=5, device="auto", hidden=(64, 32), dropout=0.1)
```

`make_loaders` defaults: `batch_size=32`, `normalize=True`,
`shuffle_train=True`. `apply_plans=True` re-applies already fitted
classical plans without refitting them. Prefer validation while you
iterate; every extra look at test spends a little of its independence.

## Text loaders

Vocab and length rules come from train. After a text fit, rebuilding
tabular loaders and calling `session.dl.evaluate` is refused: keep
loader kind consistent.

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
text_session.dl.make_text_loaders(text_column="text", max_len=32, max_vocab=500)
text_session.dl.fit(epochs=4, device="cpu")
print(text_session.dl.evaluate(partition="test").metrics)
```

This is a small embedding classifier on token ids, not Hugging Face
fine-tuning. Frozen encoder document vectors for sklearn heads live
on `session.nlp.fit_classifier(backend="embedding")`.

## Multimodal fusion

When `session.dl.fit` omits a module after multimodal loaders, the
built-in fusion is **concat**. Gated late fusion is available via
`build_multimodal_fusion(..., fusion="gated")`.

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
bundle = mm.dl.make_multimodal_loaders(text_column="text")
gated = build_multimodal_fusion(bundle.multimodal_contract, fusion="gated")
mm.dl.fit(gated, epochs=5, device="cpu", mixed_precision=False)
mm.dl.export("artifacts/mm.ts.pt", format="torchscript")
```

Trainer bundles may persist train-fit multimodal stats as
`multimodal_preprocess`. `session.dl.load_bundle` restores that meta
for inspection but does not rebuild DataLoaders. Rebuild with frozen
stats:

```python
mm.dl.make_multimodal_loaders(
    text_column="text",
    use_saved_preprocess=True,
)
```

Do not pass both `preprocess=` and `use_saved_preprocess=True`.
Missing saved preprocess with that flag raises.

Image and audio columns are small fusion branches, not a vision or
ASR product. `make_image_loaders` needs `image_column` (path or
array). `make_audio_loaders` needs `audio_column`. Short clips
repeat-pad to `audio_max_samples`. That is not transcription; see
the speech path below.

## Fold-local CV, search, nested

Normalize stats are fold-local inside these APIs. Do not tune early
stopping or architecture on Session test. Classical Session-global
plans are **not** automatically refit inside
`session.dl.cross_validate`.

```python
cv = session.dl.cross_validate(n_folds=3, epochs=2)
print(cv)

search = session.dl.search(
    param_grid={"learning_rate": [1e-3, 1e-2], "hidden": [(32,), (64, 32)]},
    n_folds=2,
    epochs=2,
)
print(search)

nested = session.dl.nested_cv(
    param_grid={"learning_rate": [1e-3, 1e-2], "hidden": [(32,), (64,)]},
    outer_cv=3,
    inner_cv=2,
    epochs=2,
)
print(nested.mean_metrics)
```

Keep classical CV ([leakage guide](leakage-cv-recipes.md)) and Torch
CV as separate protocols unless you know the interaction.

## AMP, DDP, export, reload

AMP is CUDA-only. DDP with one GPU is refused unless
`allow_cpu_ddp=True`. `load_bundle` needs the same module class.
Load does not rebuild loaders.

```python
# session.dl.fit(TinyMLP(), epochs=5, device="cuda", mixed_precision=True)
# session.dl.fit_ddp(lambda: TinyMLP(), epochs=5, world_size=2, allow_cpu_ddp=True)

session.dl.export("artifacts/model.ts.pt", format="torchscript")
# session.dl.export("artifacts/model.onnx", format="onnx")  # buildml[onnx]

restored = (
    Session.ingest(frame)
    .set_roles({"a": "feature", "b": "feature", "label": "target"})
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=42)
)
restored.dl.load_bundle(bundle, TinyMLP(), map_location="cpu", trusted=True)
restored.dl.make_loaders(batch_size=4, normalize=True, seed=42)
restored.dl.evaluate(partition="test")
restored.dl.fit(TinyMLP(), epochs=2, resume=True, device="cpu")
```

`session.dl.emit_k8s_ddp`, `session.dl.pack_torchserve`, and
`session.dl.prepare_tensorrt` emit recipes. They do not run a
cluster for you. See [serve-deploy](serve-deploy.md).

## Backbones and speech

`session.dl.load_backbone` defaults to `weights="mock"`. That is
random init for CI. Pass `weights="pretrained"` when you want real
transfer weights (downloads). Then `session.dl.attach_head(n_classes)`.

```python
# session.dl.load_backbone("vision", "resnet18", weights="mock", freeze=True)
# session.dl.attach_head(n_classes=2)
```

Speech classify: `session.dl.make_speech_loaders` /
`session.dl.fit_speech` (tiny encoder + head, finetune-lite).
`domain_adapt_speech` freezes the encoder by default. That is not
foundation-model continued pretrain.

ASR: `session.dl.transcribe(audio_column=...)`. Backend `auto`
prefers transformers when `buildml[speech]` is installed, otherwise
stub. Stub texts are waveform fingerprints. They are disclosed on
the result. Do not quote them as speech. `session.dl.evaluate_asr`
scores WER/CER against references (reuses last transcripts if you
omit hypotheses).

```python
# session.dl.transcribe(audio_column="wav", backend="stub")
# session.dl.evaluate_asr(references=["hello world", "yes"])
try:
    session.dl.refuse_speech_pretrain()
except Exception as exc:
    print(type(exc).__name__, exc)
```

## Classical plans vs Torch loaders

| Pattern | Meaning |
| --- | --- |
| Prep, then `session.dl.make_loaders` | Loaders see the mutated frame; disclosed |
| `apply_plans=True` | Re-apply fitted classical plans, no refit |
| Fold-local classical refit in Torch CV | not automatic |

## Artifacts

| Artifact | Notes |
| --- | --- |
| Session checkpoint | no Torch weights |
| `buildml.torch_bundle.v1` | `meta.json` + `trainer.pt`; load ≠ rebuild loaders |
| TorchScript / ONNX | `session.dl.export` |

Wrong loader kind after a text, multimodal, or speech fit raises
`ValidationError`. There is no Polars zero-copy into DataLoaders.
GPU is not a PR merge gate; CPU-first is the supported CI path.
