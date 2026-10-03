# Pretrained backbones

```bash
pip install "buildml[pretrained]"
# or individually: buildml[vision] / buildml[speech]
```

`session.dl.load_backbone` exposes curated vision / audio / speech
encoder hooks with `weights=none|mock|pretrained`, plus
`session.dl.attach_head` for a linear classify/probe head. Discover the
shipped list with `list_pretrained_backbones()`. This is not a full
Hugging Face / TorchVision model catalog. Default `weights="mock"` supports deterministic examples without downloads. `pretrained` downloads when you opt in. Multimodal fusion
and speech finetune-lite are separate paths.

Related: [torch-deep](torch-deep.md), [speech](speech-asr-finetune.md),
[features](../docs/features.rst).

---

## What `list_pretrained_backbones` returns

```python
from buildml.dl.zoo import list_pretrained_backbones

for row in list_pretrained_backbones():
    print(row["modality"], row["architecture"], row["provider"])
```

Curated architectures:

| Modality | Architectures | Provider |
| --- | --- | --- |
| vision | `resnet18`, `resnet34`, `resnet50`, `vit_b_16`, `vit_b_32` | torchvision |
| audio | `wav2vec2_base`, `hubert_base` | transformers |
| speech | `whisper_tiny_encoder`, `whisper_base_encoder` | transformers |

Prefer `list_pretrained_backbones()` / `session.explain("load_pretrained_backbone")`
to inspect the architectures supported by the installed version.

---

## Use case: Vision backbone + attach head (mock)

```python
# Requires: pip install "buildml[vision]". Mock weights do not download a model.
from buildml import Session

session = Session()
backbone = session.dl.load_backbone(
    "vision",
    "resnet34",  # or resnet18 / resnet50 / vit_b_16 / vit_b_32
    weights="mock",
    freeze=True,
    seed=0,
)
print(backbone.feature_dim, backbone.architecture)

head = session.dl.attach_head(n_classes=2, freeze_backbone=True)
# head.module is an nn.Module (backbone + linear head); also on session.dl.backbone_head
print(session.dl.backbone_head.n_classes)
```

`session.dl.attach_head` uses the last `session.dl.load_backbone` result on the
Session. `freeze_backbone=True` freezes encoder params and trains the linear
head (linear-probe style).

---

## Use case: Audio / speech encoders

For audio, `session.dl.load_backbone` accepts modality `"audio"` with
`"hubert_base"` or `"wav2vec2_base"`. For speech, use `"speech"` with
`"whisper_base_encoder"` or `"whisper_tiny_encoder"`. The loading and
head-attachment sequence is the same as the complete vision example above.

Use `weights="mock"` for deterministic test weights. Set
`weights="pretrained"` and a compatible `model_id`, such as
`"openai/whisper-tiny"`, to download actual pretrained weights. Install the
speech extra first and check the model's license and storage requirements.

---

## Weights modes

| Mode | Behavior |
| --- | --- |
| `none` | Architecture shell without meaningful weights |
| `mock` | Deterministic/CI-safe tensors: default for tests |
| `pretrained` | Load upstream weights when extras + network allow |

`freeze=True` is typical when attaching a small task head.

---

## AI tool exposure

The AI operator allowlist can call `session.dl.load_backbone` and
`session.dl.attach_head` as typed tools
([ai-tools-operator-patterns](ai-tools-operator-patterns.md)). Still verify
architecture names and weight modes before confirming execution.

---

## Failure modes / limits

- Missing `vision` / `speech` / `pretrained` extra → `MissingExtraError`.
- Unknown architecture → validation error (no fallback to another architecture).
- `pretrained` without network/cache → upstream download errors.
- `session.dl.attach_head` without a prior `session.dl.load_backbone` → validation error.
- Not a substitute for `session.dl.make_image_loaders` contracts.
- Not Whisper-scale training: see `session.dl.refuse_speech_pretrain`.

---

## Related

- [Torch deep](torch-deep.md)
- [Speech](speech-asr-finetune.md)
- [AI tools](ai-tools-operator-patterns.md)
