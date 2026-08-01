"""DL alpha-gate smoke: ingest through bundle resume + training curve (torch extra)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def _frame(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    labels = np.asarray([0, 1] * (n // 2), dtype=np.int64)
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "label": labels,
        }
    )


def test_dl_alpha_gate_smoke(tmp_path: Path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    import torch
    from torch import nn

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    frame = _frame()
    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=11)
    )
    loaders = session.make_torch_loaders(batch_size=16, normalize=True, seed=11)
    assert loaders.report.normalize is True
    assert "train" in loaders.loaders

    session.fit_torch(
        TinyMLP(),
        epochs=4,
        learning_rate=5e-3,
        device="cpu",
        early_stopping_patience=2,
        scheduler="none",
    )
    assert session.dl_train_result is not None
    assert session.dl_train_result.n_epochs_ran >= 1

    evaluation = session.evaluate_torch(partition="test")
    assert evaluation.partition == "test"
    assert evaluation.n_rows > 0
    assert "accuracy" in evaluation.metrics
    before_acc = evaluation.metrics["accuracy"]

    curve = session.torch_training_curve()
    assert curve.epochs
    assert curve.disclosures
    assert any("Device resolved=" in item for item in curve.disclosures)

    bundle = session.save_torch_bundle(tmp_path / "torch_bundle")
    assert (bundle / "meta.json").is_file()
    assert (bundle / "trainer.pt").is_file()

    restored = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=11)
        .load_torch_bundle(bundle, TinyMLP(), map_location="cpu")
    )
    restored.make_torch_loaders(batch_size=16, normalize=True, seed=11)
    again = restored.evaluate_torch(partition="test")
    assert again.metrics["accuracy"] == pytest.approx(before_acc, abs=1e-5)

    epochs_before = restored.dl_train_result.n_epochs_ran
    restored.fit_torch(TinyMLP(), epochs=1, resume=True, device="cpu", learning_rate=5e-3)
    assert restored.dl_train_result.n_epochs_ran >= epochs_before

    before = session.explain("fit_torch", moment="before")
    assert before.operation == "fit_torch"
