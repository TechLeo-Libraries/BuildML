# Speech: ASR transcription and classify finetune-lite

```bash
pip install "buildml[speech]"
```

You have audio on a Session split. Transcribe it, or train a small
classify head on the same partitions. When `buildml[speech]` is
installed, default ASR prefers transformers. Otherwise you get a
deterministic stub. Stub use is always disclosed. This does not train
Whisper-scale foundation models from scratch.
`session.dl.refuse_speech_pretrain()` raises an error for unsupported
foundation-model pretraining requests.

Related: [torch-deep](torch-deep.md), [pretrained-backbones](pretrained-backbones.md),
[features](../docs/features.rst).

---

## Use case A: Stub ASR (CI-safe) + WER/CER

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

print("default ASR backend:", resolve_default_asr_backend())
# Explicit stub for CI / offline (default would prefer transformers when installed):
asr = speech.dl.transcribe(audio_column="audio", backend="stub")
print(asr.backend, asr.disclosures[:2])
assert asr.meta.get("stub") is True

# Score hypotheses vs gold references (word and character edit distances).
# Session path reuses last session.dl.transcribe texts when hypotheses= is omitted:
scored = speech.dl.evaluate_asr(
    references=["synthetic tone"] * len(asr.texts),
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
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

# Requires: pip install "buildml[speech]" and a network connection for model weights.
# Replace these synthetic tones with speech recordings for meaningful transcripts.
asr = speech.dl.transcribe(audio_column="audio", backend="transformers",
                           model_id="openai/whisper-tiny", partition="test")
print(asr.texts)
```

Treat downloaded weights as an operator concern (license, cache, GPU).
Keep CI on `backend="stub"`.

---

## Use case C: Speech classify finetune-lite + `SpeechContract`

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

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
while keeping the pretrained encoder fixed. `SpeechContract.to_dict` /
`from_dict` keep sample rate, max samples, amp stats, and class labels aligned
across save/load paths.

---

## Use case D: Domain adapt helper

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

speech.dl.domain_adapt_speech(
    epochs=5,
    freeze_encoder=True,
    device="cpu",
    audio_column="audio",
)
```

This is explicit **domain adapt**, not continued foundation pretrain.

---

## Unsupported foundation-model pretraining

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

try:
    speech.dl.refuse_speech_pretrain()
except Exception as exc:
    print(type(exc).__name__, exc)
```

BuildML does not provide foundation-model pretraining. The method below
returns the explicit unsupported-operation error.

---

## Pretrained speech encoders

```python
import numpy as np
import pandas as pd
from buildml import Session
from buildml.dl.speech import evaluate_asr, resolve_default_asr_backend

# Synthetic tones demonstrate the audio API; they contain no spoken words.
time = np.arange(16000, dtype=np.float32) / 16000
df = pd.DataFrame({
    "audio": [np.sin(2 * np.pi * (220 if i % 2 else 440) * time).astype("float32") for i in range(20)],
    "y": [i % 2 for i in range(20)],
})
speech = (Session.ingest(df)
    .set_roles({"audio": "feature", "y": "target"})
    .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0))

# Requires: pip install "buildml[pretrained]"
backbone = speech.dl.load_backbone("speech", "whisper_tiny_encoder", weights="mock", freeze=True)
print(backbone)
```

See [pretrained-backbones](pretrained-backbones.md). `weights="mock"` is CI-safe;
`pretrained` may download.

---

## Failure modes / limits

| Limit | Behavior |
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
