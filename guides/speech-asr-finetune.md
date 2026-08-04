# Speech: ASR transcription and classify finetune-lite

> **Install (GitHub 2.x + speech):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[speech]"   # torch + transformers
> ```
> PyPI `buildml` is legacy 1.x. See [installation](../docs/installation.rst).

BuildML ships an **integration path** for speech: environment-aware ASR
(default prefers **transformers** when `buildml[speech]` is installed; falls
back to a deterministic **stub** for CI / absent extras), plus classify
**finetune-lite** / domain adapt on Session partitions. It does **not** train
Whisper-scale foundation models from scratch. Stub use is always disclosed.

Related: [torch-deep](torch-deep.md), [pretrained-backbones](pretrained-backbones.md),
[features](../docs/features.rst).

---

## Why this boundary exists

Foundation-model pretraining needs massive corpora, specialized distributed
stacks, and months of compute. BuildML’s job here is:

1. Attach audio columns to the same roles/splits as tabular workflows.
2. Transcribe with transformers by default when installed; pass
   `backend="stub"` for CI / offline fingerprints (always disclosed).
3. Finetune a small classifier head / domain-adapt with frozen encoders.
4. **Hard-refuse** “train Whisper from scratch” product expectations via
   `session.dl.refuse_speech_pretrain()`.

Catalog: `dl_capability_matrix()["modalities"]["speech"]["default_asr_backend"]`
is `"transformers"` when the speech stack is present, else `"stub"`.

---

## Use case A: Stub ASR (CI-safe) + WER/CER

```python
import pandas as pd

from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# In real projects, audio cells are paths or arrays; stub backend tolerates demos.
df = pd.DataFrame(
    {
        "audio": ["clip_a.wav", "clip_b.wav", "clip_c.wav", "clip_d.wav"],
        "y": [0, 1, 0, 1],
    }
)

speech = (
    Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
)

print("default ASR backend:", resolve_default_asr_backend())
# Explicit stub for CI / offline (default would prefer transformers when installed):
asr = speech.dl.transcribe(audio_column="audio", backend="stub")
print(asr.backend, asr.disclosures[:2])
assert asr.meta.get("stub") is True

# Score hypotheses vs gold references (string edit distance: not a MOS product).
# Session path reuses last session.dl.transcribe texts when hypotheses= is omitted:
scored = speech.dl.evaluate_asr(
    references=["hello world", "good morning", "approved", "denied"],
)
print(scored.wer, scored.cer)
assert speech.dl.asr_eval is scored

# Standalone helper (same metrics, no Session required):
standalone = evaluate_asr(
    hypotheses=["hello world", "good night"],
    references=["hello world", "good morning"],
    lowercase=True,
)
print(standalone.wer, standalone.cer)
```

`session.dl.evaluate_asr` returns `AsrEvalResult` with corpus WER/CER plus optional
per-utterance rows. It does not download ASR models.

---

## Use case B: Transformers Whisper-class transcription

When `transformers` is installed, **omitting `backend=`** (or passing
`backend="auto"`) resolves to transformers. Name a real model for production
quality; the library default model id is a tiny testing checkpoint.

```python
# Requires buildml[speech]; may download weights the first time.
# asr = speech.dl.transcribe(
#     audio_column="audio",
#     # backend defaults to transformers when available
#     model_id="openai/whisper-tiny",
#     partition="test",
# )
```

Treat downloaded weights as an operator concern (license, cache, GPU).
Keep CI on `backend="stub"`.

---

## Use case C: Speech classify finetune-lite + `SpeechContract`

```python
from buildml.dl.speech import SpeechContract

speech.dl.make_speech_loaders(
    audio_column="audio",
    sample_rate=16000,
    max_samples=16000,
    encoder_dim=64,
)
speech.dl.fit_speech(epochs=5, freeze_encoder=True, device="cpu")
print(speech.dl.speech_result)

# Contract round-trip for bundle / meta persistence:
contract = SpeechContract(
    audio_column="audio",
    target_column="y",
    class_labels=(0, 1),
    sample_rate=16_000,
    max_samples=8_000,
    encoder_dim=32,
)
restored = SpeechContract.from_dict(contract.to_dict())
assert restored.audio_column == "audio"
```

`freeze_encoder=True` is the common domain-adapt pattern: train a light head
without claiming full ASR FM training. `SpeechContract.to_dict` /
`from_dict` keep sample rate, max samples, amp stats, and class labels aligned
across save/load paths.

---

## Use case D: Domain adapt helper

```python
speech.dl.domain_adapt_speech(
    epochs=5,
    freeze_encoder=True,
    device="cpu",
    audio_column="audio",
)
```

This is explicit **domain adapt**, not continued foundation pretrain.

---

## Use case E: Honest refuse for FM-from-scratch asks

```python
try:
    speech.dl.refuse_speech_pretrain()
except Exception as exc:
    print(type(exc).__name__, exc)
```

Call this (or cite it) when operators ask BuildML to “pretrain Whisper on our
data.” The answer is a refuse, not a half-implemented trainer.

---

## Pretrained speech encoders

```python
# pip install "buildml[pretrained]" or buildml[speech]
# backbone = speech.dl.load_backbone(
#     "speech", "whisper_encoder", weights="mock", freeze=True
# )
```

See [pretrained-backbones](pretrained-backbones.md). `weights="mock"` is CI-safe;
`pretrained` may download.

---

## Failure modes / limits

| Limit | Honesty |
| --- | --- |
| FM pretrain from scratch | Refused |
| Audio multimodal fusion vs speech path | Different APIs: fusion is not ASR |
| Missing files in path cells | Loader/transcribe errors: validate paths |
| Transformers backend | Needs `buildml[speech]` + download |
| Bundle load | Torch speech loaders are not rebuilt by `session.dl.load_bundle` |
| `session.dl.evaluate_asr` | String WER/CER only: not speech quality / MOS |

---

## Related

- [Torch deep](torch-deep.md)
- [Serve & deploy](serve-deploy.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
