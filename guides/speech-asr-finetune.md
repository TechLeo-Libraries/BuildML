# Speech: ASR transcription and classify finetune-lite

> **Install (GitHub 2.x + speech):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[speech]"   # torch + transformers
> ```
> PyPI `buildml` is legacy 1.x. See [installation](../docs/installation.rst).

BuildML ships an **integration path** for speech: stub-safe ASR for CI,
optional transformers Whisper-class transcription, and classify
**finetune-lite** / domain adapt on Session partitions. It does **not** train
Whisper-scale foundation models from scratch.

Related: [torch-deep](torch-deep.md), [pretrained-backbones](pretrained-backbones.md),
[features](../docs/features.rst).

---

## Why this boundary exists

Foundation-model pretraining needs massive corpora, specialized distributed
stacks, and months of compute. BuildML’s job here is:

1. Attach audio columns to the same roles/splits as tabular workflows.
2. Transcribe with a stub (CI) or an optional HF backend.
3. Finetune a small classifier head / domain-adapt with frozen encoders.
4. **Hard-refuse** “train Whisper from scratch” product expectations via
   `refuse_speech_foundation_pretrain()`.

---

## Use case A — Stub ASR (CI-safe) + WER/CER

```python
import pandas as pd

from buildml import Session
from buildml.dl.speech import evaluate_asr

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

asr = speech.transcribe_speech(audio_column="audio", backend="stub")
print(asr)

# Score hypotheses vs gold references (string edit distance — not a MOS product).
# Session path reuses last transcribe_speech texts when hypotheses= is omitted:
scored = speech.evaluate_asr(
    references=["hello world", "good morning", "approved", "denied"],
)
print(scored.wer, scored.cer)
assert speech.dl_asr_eval is scored

# Standalone helper (same metrics, no Session required):
standalone = evaluate_asr(
    hypotheses=["hello world", "good night"],
    references=["hello world", "good morning"],
    lowercase=True,
)
print(standalone.wer, standalone.cer)
```

`evaluate_asr` returns `AsrEvalResult` with corpus WER/CER plus optional
per-utterance rows. It does not download ASR models.

---

## Use case B — Transformers Whisper-class transcription

```python
# Requires network + model download the first time; not used in default CI.
# asr = speech.transcribe_speech(
#     audio_column="audio",
#     backend="transformers",
#     model_id="openai/whisper-tiny",
#     partition="test",
# )
```

Treat downloaded weights as an operator concern (license, cache, GPU).

---

## Use case C — Speech classify finetune-lite + `SpeechContract`

```python
from buildml.dl.speech import SpeechContract

speech.make_speech_torch_loaders(
    audio_column="audio",
    sample_rate=16000,
    max_samples=16000,
    encoder_dim=64,
)
speech.fit_speech_torch(epochs=5, freeze_encoder=True, device="cpu")
print(speech.dl_speech_result)

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

## Use case D — Domain adapt helper

```python
speech.domain_adapt_speech_torch(
    epochs=5,
    freeze_encoder=True,
    device="cpu",
    audio_column="audio",
)
```

This is explicit **domain adapt**, not continued foundation pretrain.

---

## Use case E — Honest refuse for FM-from-scratch asks

```python
try:
    speech.refuse_speech_foundation_pretrain()
except Exception as exc:
    print(type(exc).__name__, exc)
```

Call this (or cite it) when operators ask BuildML to “pretrain Whisper on our
data.” The answer is a refuse, not a half-implemented trainer.

---

## Pretrained speech encoders

```python
# pip install "buildml[pretrained]" or buildml[speech]
# backbone = speech.load_pretrained_backbone(
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
| Audio multimodal fusion vs speech path | Different APIs — fusion is not ASR |
| Missing files in path cells | Loader/transcribe errors — validate paths |
| Transformers backend | Needs `buildml[speech]` + download |
| Bundle load | Torch speech loaders are not rebuilt by `load_torch_bundle` |
| `evaluate_asr` | String WER/CER only — not speech quality / MOS |

---

## Related

- [Torch deep](torch-deep.md)
- [Serve & deploy](serve-deploy.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
