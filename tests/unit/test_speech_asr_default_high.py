"""HIGH-depth speech ASR default: prefer transformers when available; stub disclosed."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml.dl.catalog import dl_capability_matrix
from buildml.dl.speech import (
    resolve_asr_backend,
    resolve_default_asr_backend,
    speech_stack_available,
    transcribe_audio_values,
)

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _wave(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 4 * t) + 0.02 * rng.normal(size=n)).astype(
        np.float32
    )


def test_resolve_default_asr_backend_matches_stack() -> None:
    expected = "transformers" if speech_stack_available() else "stub"
    assert resolve_default_asr_backend() == expected
    assert resolve_asr_backend(None) == expected
    assert resolve_asr_backend("auto") == expected
    assert resolve_asr_backend("stub") == "stub"


def test_catalog_default_asr_backend_honest() -> None:
    matrix = dl_capability_matrix()
    speech = matrix["modalities"]["speech"]
    expected = "transformers" if speech["transformers_asr_available"] else "stub"
    assert speech["default_asr_backend"] == expected
    assert "prefer transformers" in speech["default_asr_backend_policy"].lower()


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_stub_backend_disclosed_clearly() -> None:
    result = transcribe_audio_values(
        [_wave(0), _wave(1)],
        backend="stub",
        sample_rate=8_000,
        max_samples=256,
    )
    assert result.backend == "stub"
    assert result.meta.get("stub") is True
    joined = " ".join(result.disclosures + tuple(result.warnings)).lower()
    assert "stub asr in use" in joined
    assert all(t.startswith("[stub-asr]") for t in result.texts)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_session_default_backend_follows_policy() -> None:
    from buildml import Session

    frame = pd.DataFrame(
        {"audio": [_wave(i) for i in range(6)], "y": [0, 1, 0, 1, 0, 1]}
    )
    session = Session.ingest(frame).set_roles({"audio": "feature", "y": "target"})
    if speech_stack_available():
        # Avoid downloading weights in unit tests: explicit stub still works;
        # default resolution is covered by resolve_* / catalog tests above.
        result = session.dl.transcribe(
            audio_column="audio",
            backend="stub",
            sample_rate=8_000,
            max_samples=256,
        )
        assert result.backend == "stub"
        assert any("stub asr in use" in w.lower() for w in result.warnings)
    else:
        result = session.dl.transcribe(
            audio_column="audio",
            sample_rate=8_000,
            max_samples=256,
        )
        assert result.backend == "stub"
        assert result.meta.get("stub") is True
