# Pretrained backbones

> **Install (GitHub 2.x):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[pretrained]"   # vision + speech extras
> # or individually: buildml[vision] / buildml[speech]
> ```
> See [installation](../docs/installation.rst).

`Session.load_pretrained_backbone` exposes **curated** ResNet/ViT, Wav2Vec2, and
Whisper-encoder hooks with `weights=none|mock|pretrained`. This is **not** a
full Hugging Face / TorchVision zoo product.

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

## Use case — Vision backbone (mock)

```python
from buildml import Session

session = Session()
backbone = session.load_pretrained_backbone(
    "vision",
    "resnet18",
    weights="mock",
    freeze=True,
    seed=0,
)
print(backbone)
```

Other curated vision names (when available in your installed version) follow the
same pattern — prefer `session.explain("load_pretrained_backbone")` for the
current list rather than memorizing a stale zoo table.

---

## Use case — Audio / speech encoders

```python
# audio modality (e.g. wav2vec-class hooks)
# audio_bb = session.load_pretrained_backbone("audio", "wav2vec2", weights="mock")

# speech encoder hook (not FM pretrain)
# speech_bb = session.load_pretrained_backbone(
#     "speech", "whisper_encoder", weights="mock", freeze=True
# )

# Real weights (may download; operator-owned cache/license):
# speech_bb = session.load_pretrained_backbone(
#     "speech",
#     "whisper_encoder",
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

The AI operator allowlist can call pretrained loading as a typed tool
([ai-tools-operator-patterns](ai-tools-operator-patterns.md)). Still verify
architecture names and weight modes before confirming execution.

---

## Failure modes / limits

- Missing `vision` / `speech` / `pretrained` extra → `MissingExtraError`.
- Unknown architecture → validation error (not silent fallback to a random zoo model).
- `pretrained` without network/cache → upstream download errors.
- Not a substitute for `make_image_multimodal_torch_loaders` contracts.
- Not Whisper-scale training — see `refuse_speech_foundation_pretrain`.

---

## Related

- [Torch deep](torch-deep.md)
- [Speech](speech-asr-finetune.md)
- [AI tools](ai-tools-operator-patterns.md)
