"""Pass Q: contiguous Torch class-id encoding for non-contiguous labels."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.labels import encode_class_targets, fit_class_labels, n_classes_from_labels

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_AUDIO_LEN = 256
_AUDIO_SR = 8_000


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch import failed: {exc}")


def test_fit_and_encode_non_contiguous_labels() -> None:
    labels = fit_class_labels(pd.Series([10, 30, 10, 20]))
    assert labels == (10, 20, 30)
    assert n_classes_from_labels(labels) == 3
    encoded = encode_class_targets(pd.Series([30, 10, 20, 10]), labels)
    assert encoded.tolist() == [2, 0, 1, 0]


def test_encode_rejects_unseen_label() -> None:
    labels = fit_class_labels([0, 2])
    with pytest.raises(ValidationError, match="not present in the train"):
        encode_class_targets([0, 5], labels)


def _synthetic_wave(seed: int, n: int = _AUDIO_LEN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    freq = 3.0 + (seed % 5)
    return (0.4 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=n)).astype(
        np.float32
    )


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_tabular_non_contiguous_labels_train_and_eval() -> None:
    _require_torch_or_skip()
    # Sparse ids {10, 20} — old footgun used n_classes=2 with raw targets 10/20.
    y = np.asarray([10, 20] * 24, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "x1": np.linspace(-1, 1, len(y)),
            "x2": np.linspace(0, 1, len(y)),
            "y": y,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_torch_loaders(batch_size=8, seed=0)
    assert bundle.contract.class_labels == (10, 20)
    xs, ys = next(iter(bundle.loaders["train"]))
    assert int(ys.min()) >= 0
    assert int(ys.max()) <= 1
    session.fit_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    assert getattr(session.dl_train_result.module, "n_classes", None) == 2
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0
    assert "accuracy" in ev.metrics
    assert ev.class_labels == (10, 20)
    assert ev.confusion_matrix is not None


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_speech_non_contiguous_labels_train_and_eval() -> None:
    _require_torch_or_skip()
    n = 48
    rng = np.random.default_rng(1)
    # Three sparse class ids — n_classes must be 3 with targets in {0,1,2}.
    raw = rng.choice([10, 20, 30], size=n)
    frame = pd.DataFrame(
        {
            "audio": [_synthetic_wave(i) for i in range(n)],
            "y": raw,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=1)
    )
    bundle = session.make_speech_torch_loaders(
        audio_column="audio",
        sample_rate=_AUDIO_SR,
        max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=1,
    )
    assert bundle.contract.class_labels == (10, 20, 30)
    _, ys = next(iter(bundle.loaders["train"]))
    assert int(ys.min()) >= 0
    assert int(ys.max()) <= 2
    session.fit_speech_torch(epochs=2, device="cpu")
    assert getattr(session.dl_train_result.module, "n_classes", None) == 3
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0
    assert ev.class_labels == (10, 20, 30)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_text_non_contiguous_labels_train() -> None:
    _require_torch_or_skip()
    texts = [
        "alpha beta gamma",
        "beta gamma delta",
        "gamma delta epsilon",
        "delta epsilon zeta",
    ] * 12
    y = np.asarray([100, 200] * 24, dtype=np.int64)
    frame = pd.DataFrame({"text": texts, "y": y})
    session = (
        Session.ingest(frame)
        .set_roles({"text": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=2)
    )
    bundle = session.make_text_torch_loaders(text_column="text", batch_size=8, seed=2)
    assert bundle.contract.class_labels == (100, 200)
    _, ys = next(iter(bundle.loaders["train"]))
    assert set(int(v) for v in ys.unique().tolist()) <= {0, 1}
    session.fit_torch(epochs=1, device="cpu")
    assert getattr(session.dl_train_result.module, "n_classes", None) == 2
