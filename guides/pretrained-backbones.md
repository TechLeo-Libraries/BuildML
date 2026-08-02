# Pretrained backbones

> **Install (GitHub 2.x):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[pretrained]"   # vision + speech extras
> # or individually: buildml[vision] / buildml[speech]
> ```
> See [installation](../docs/installation.rst).

`Session.load_pretrained_backbone` exposes **curated** vision / audio / speech
encoder hooks with `weights=none|mock|pretrained`, plus
`Session.attach_backbone_head` for a linear classify/probe head. Discover the
shipped catalog with `list_pretrained_backbones()`. This is **not** a full
Hugging Face / TorchVision zoo product.

Related: [torch-deep](torch-deep.md), [speech](speech-asr-finetune.md),
[features](../docs/features.rst).

---

## Why curated hooks (not a zoo)

A zoo product implies continuous coverage of architectures, weight variants,
preprocessing contracts, and breaking upstream changes. BuildML instead:

1. Offers a small, tested surface for Session/AI-tool integration.
2. Defaults to **`weights="mock"`** for CI-safe graphs.
3. Allows `pretrained` downloads when operators opt in.
4. Stays explicit that multimodal fusion and speech finetune-lite are separate
   paths.

---

## Catalog — `list_pretrained_backbones`

```python
from buildml.dl.zoo import list_pretrained_backbones

for row in list_pretrained_backbones():
    print(row["modality"], row["architecture"], row["provider"])
```

Curated architectures (Pass V expansion):

| Modality | Architectures | Provider |
| --- | --- | --- |
| vision | `resnet18`, `resnet34`, `resnet50`, `vit_b_16`, `vit_b_32` | torchvision |
| audio | `wav2vec2_base`, `hubert_base` | transformers |
| speech | `whisper_tiny_encoder`, `whisper_base_encoder` | transformers |

Prefer `list_pretrained_backbones()` / `session.explain("load_pretrained_backbone")`
over memorizing a stale table when the installed version may differ.

---

## Use case — Vision backbone + attach head (mock)

```python
from buildml import Session

session = Session()
backbone = session.load_pretrained_backbone(
    "vision",
    "resnet34",  # or resnet18 / resnet50 / vit_b_16 / vit_b_32
    weights="mock",
    freeze=True,
    seed=0,
)
print(backbone.feature_dim, backbone.architecture)

head = session.attach_backbone_head(n_classes=2, freeze_backbone=True)
# head.module is an nn.Module (backbone + linear head); also on session.dl_backbone_head
print(session.dl_backbone_head.n_classes)
```

`attach_backbone_head` uses the last `load_pretrained_backbone` result on the
Session. `freeze_backbone=True` freezes encoder params and trains the linear
head (linear-probe style).

---

## Use case — Audio / speech encoders

```python
# audio modality
# audio_bb = session.load_pretrained_backbone(
#     "audio", "hubert_base", weights="mock", freeze=True
# )
# # also: "wav2vec2_base"

# speech encoder hook (not FM pretrain)
# speech_bb = session.load_pretrained_backbone(
#     "speech", "whisper_base_encoder", weights="mock", freeze=True
# )
# # also: "whisper_tiny_encoder"

# Real weights (may download; operator-owned cache/license):
# speech_bb = session.load_pretrained_backbone(
#     "speech",
#     "whisper_tiny_encoder",
#     weights="pretrained",
#     model_id="openai/whisper-tiny",
# )
```

---

## Weights modes

| Mode | Behavior |
| --- | --- |
| `none` | Architecture shell without meaningful weights |
| `mock` | Deterministic/CI-safe tensors — default for tests |
| `pretrained` | Load upstream weights when extras + network allow |

`freeze=True` is typical when attaching a small task head.

---

## AI tool exposure

The AI operator allowlist can call `load_pretrained_backbone` and
`attach_backbone_head` as typed tools
([ai-tools-operator-patterns](ai-tools-operator-patterns.md)). Still verify
architecture names and weight modes before confirming execution.

---

## Failure modes / limits

- Missing `vision` / `speech` / `pretrained` extra → `MissingExtraError`.
- Unknown architecture → validation error (not silent fallback to a random zoo model).
- `pretrained` without network/cache → upstream download errors.
- `attach_backbone_head` without a prior `load_pretrained_backbone` → validation error.
- Not a substitute for `make_image_multimodal_torch_loaders` contracts.
- Not Whisper-scale training — see `refuse_speech_foundation_pretrain`.

---

## Related

- [Torch deep](torch-deep.md)
- [Speech](speech-asr-finetune.md)
- [AI tools](ai-tools-operator-patterns.md)
