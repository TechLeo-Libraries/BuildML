"""Deep-learning capability matrix: honest defaults and modality boundaries."""

from __future__ import annotations

from typing import Any

from buildml.dl.extras import torch_available, torch_spec_available
from buildml.dl.speech import speech_stack_available

TABULAR_MODALITIES = ("tabular",)
VISION_MODALITIES = ("image",)
AUDIO_MODALITIES = ("audio",)
SPEECH_MODALITIES = ("speech",)
TEXT_MODALITIES = ("text",)
MULTIMODAL_MODALITIES = ("multimodal",)

WEIGHT_MODES = ("none", "mock", "pretrained")
SPEECH_BACKENDS = ("stub", "transformers")


def dl_capability_matrix() -> dict[str, Any]:
    """Report what this installation can do for Torch supervised and speech paths.

    Probes optional dependencies each call so UI, tests, and explain surfaces
    reflect the environment now rather than at import time.

    Returns
    -------
    dict[str, Any]
        Modalities, weight-mode defaults, speech backend availability, export
        paths, install hints, and explicit non-goals.
    """
    torch_ok = torch_available()
    speech_ok = speech_stack_available()
    return {
        "modalities": {
            "tabular": {
                "available": torch_ok,
                "extra": "torch",
                "entrypoints": ["fit_torch", "cross_validate_torch", "search_torch"],
                "notes": "Tabular MLP supervised training via Session.fit_torch.",
            },
            "image": {
                "available": torch_ok,
                "extra": "torch",
                "entrypoints": ["fit_torch (image columns)", "load_vision_backbone"],
                "notes": "Vision backbones via torchvision; requires Pillow.",
            },
            "audio": {
                "available": torch_ok,
                "extra": "torch",
                "entrypoints": ["fit_torch (audio columns)", "load_audio_backbone"],
                "notes": "Waveform classifiers and Wav2Vec2/HuBERT feature heads.",
            },
            "speech": {
                "available": torch_ok,
                "extra": "torch",
                "asr_backends": list(SPEECH_BACKENDS),
                "default_asr_backend": "transformers" if speech_ok else "stub",
                "default_asr_backend_policy": (
                    "Prefer transformers when the speech stack is installed; "
                    "fall back to deterministic stub for CI / absent extras. "
                    "Stub use is disclosed on SpeechTranscribeResult."
                ),
                "transformers_asr_available": speech_ok,
                "entrypoints": ["transcribe_from_dataset", "evaluate_asr", "fit_torch"],
                "notes": (
                    "Default ASR backend prefers transformers when "
                    "buildml[speech] / transformers is available; otherwise stub. "
                    "Pass backend='stub' explicitly for offline CI. "
                    "Stub texts are fingerprints, not speech."
                ),
            },
            "text": {
                "available": torch_ok,
                "extra": "torch",
                "entrypoints": ["fit_torch (text column)", "make_text_loaders"],
                "notes": "Char/word vocab classifiers: not HF transformer fine-tuning.",
            },
            "multimodal": {
                "available": torch_ok,
                "extra": "torch",
                "entrypoints": ["make_multimodal_loaders", "build_multimodal_fusion"],
                "notes": "Early-fusion tabular+image/audio loaders when columns exist.",
            },
        },
        "weight_modes": {
            "supported": list(WEIGHT_MODES),
            "default_zoo_mode": "mock",
            "disclosure": (
                "mock/none use random or uninitialized weights: for plumbing tests only. "
                "Use weight_mode='pretrained' for real transfer learning."
            ),
        },
        "training": {
            "single_device": torch_ok,
            "ddp": torch_ok,
            "export": ["torchscript", "onnx"] if torch_ok else [],
            "bundle_format": "buildml.torch_bundle.v1",
        },
        "default_backend_when_installed": "torch" if torch_ok else None,
        "install_hints": {
            "torch": "pip install 'buildml[torch]'  # supervised Torch training",
            "dl": "pip install 'buildml[dl]'  # alias extra for torch stack",
            "speech": "pip install 'buildml[speech]'  # Whisper-class ASR backend",
        },
        "serving": {
            "entrypoint": "buildml-serve / session.dl.serve / ServeConfig",
            "extra": "serve",
            "auth": ["api_key_bearer", "http_basic"],
            "docs_default_when_auth": "closed",
            "dockerfile": "deploy/serve/Dockerfile",
            "k8s_template": "deploy/k8s/serve-deployment.example.yaml",
            "notes": (
                "Local managed FastAPI serve with trust gates, optional API-key/"
                "Basic auth, ServeConfig YAML/env/CLI, and operator-owned Docker/K8s "
                "recipes. Not a managed cloud IAM / multi-cluster product."
            ),
        },
        "non_goals": [
            "Foundation-model pretraining from scratch (refuse_foundation_model_pretrain)",
            "Full Hugging Face hub mirror / arbitrary checkpoint zoo",
            "Managed cloud IAM / multi-cluster MLOps control plane "
            "(first-party serve Dockerfile + K8s templates are operator-owned recipes)",
            "Automatic mixed-precision product defaults",
        ],
        "torch_present": torch_ok,
        "torch_spec_present": torch_spec_available(),
        "speech_stack_present": speech_ok,
    }
