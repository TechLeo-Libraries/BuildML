"""Integration smoke for the tabular Torch Session path."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _torch_usable() -> bool:
    if not _TORCH_SPEC:
        return False
    try:
        from buildml.dl.extras import torch_available

        return torch_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")


def test_session_torch_vertical_slice(tmp_path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    import torch
    from torch import nn

    rng = np.random.default_rng(7)
    n = 120
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "label": (rng.normal(size=n) > 0).astype(int),
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
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=7)
    )
    session.make_torch_loaders(batch_size=32, normalize=True, seed=7)
    session.fit_torch(TinyMLP(), epochs=4, learning_rate=5e-3, device="cpu")
    result = session.evaluate_torch(partition="test")
    assert result.partition == "test"
    assert result.n_rows > 0
    assert "accuracy" in result.metrics

    bundle_path = session.save_torch_bundle(tmp_path / "bundle")
    restored = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=7)
    )
    restored.load_torch_bundle(bundle_path, TinyMLP(, trusted=True), map_location="cpu")
    restored.make_torch_loaders(batch_size=32, normalize=True, seed=7)
    again = restored.evaluate_torch(partition="test")
    assert again.metrics["accuracy"] == pytest.approx(result.metrics["accuracy"], abs=1e-5)

    before = session.explain("fit_torch", moment="before")
    assert before.operation == "fit_torch"
