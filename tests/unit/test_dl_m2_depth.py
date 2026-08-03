"""M2 Torch depth: early stop, schedulers, resume, group/time, curves."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.session.walkthrough import torch_training_status_for_walkthrough

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _torch_usable() -> bool:
    if not _TORCH_SPEC:
        return False
    try:
        from buildml.dl.extras import torch_available

        return torch_available()
    except Exception:
        return False


def _cls_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": np.asarray([0, 1] * (n // 2), dtype=np.int64),
        }
    )


def _tiny_module():
    import torch
    from torch import nn

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return TinyMLP()


def test_catalog_covers_m2_ops_and_concepts() -> None:
    assert "torch_training_curve" in OPERATION_CATALOG
    assert "training-curves" in OPERATION_CATALOG["fit_torch"].concept_links
    assert "early_stopping_patience" in {
        p.name for p in OPERATION_CATALOG["fit_torch"].parameters
    }
    assert "resume" in {p.name for p in OPERATION_CATALOG["fit_torch"].parameters}


def test_torch_training_status_empty_without_trainer() -> None:
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    status = torch_training_status_for_walkthrough(session)
    assert status["enabled"] is False


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_early_stopping_triggers_and_records_reason() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    from buildml.dl.types import TrainConfig

    session = (
        Session.ingest(_cls_frame(120))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_torch_loaders(batch_size=16, normalize=True, seed=0)
    # High LR + tiny patience encourages a stop before the full budget on noisy val.
    session.fit_torch(
        _tiny_module(),
        config=TrainConfig(
            epochs=30,
            learning_rate=0.5,
            device="cpu",
            early_stopping_patience=2,
            early_stopping_monitor="val_loss",
            restore_best_weights=True,
            log_every=1,
        ),
    )
    result = session.dl_train_result
    assert result is not None
    assert result.early_stop is not None
    assert result.early_stop.enabled is True
    assert result.early_stop.partition == "validation"
    assert result.early_stop.reason
    assert result.n_epochs_ran <= 30
    assert result.training_curve is not None
    assert result.training_curve.disclosures


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_early_stopping_requires_validation() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(60))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.3, random_state=0)
    )
    session.make_torch_loaders(batch_size=16, normalize=False, seed=0)
    with pytest.raises(ValidationError, match="validation"):
        session.fit_torch(
            _tiny_module(),
            epochs=3,
            device="cpu",
            early_stopping_patience=1,
        )


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_scheduler_and_grad_clip_defaults() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    from buildml.dl.types import (
        DEFAULT_GRAD_CLIP_NORM,
        DEFAULT_SCHEDULER,
        TrainConfig,
    )

    assert DEFAULT_SCHEDULER == "none"
    assert DEFAULT_GRAD_CLIP_NORM is None

    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=1)
    )
    session.make_torch_loaders(batch_size=16, seed=1)
    session.fit_torch(
        _tiny_module(),
        config=TrainConfig(
            epochs=3,
            learning_rate=1e-2,
            device="cpu",
            scheduler="step",
            scheduler_step_size=1,
            scheduler_gamma=0.5,
            grad_clip_norm=1.0,
        ),
    )
    result = session.dl_train_result
    assert result is not None
    assert result.scheduler_name == "step"
    assert result.scheduler_state is not None
    assert any("lr" in row for row in result.history)
    lrs = [row["lr"] for row in result.history if "lr" in row]
    assert lrs[-1] <= lrs[0]


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_group_and_time_loaders_honor_membership() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    rng = np.random.default_rng(2)
    n = 90
    groups = np.repeat(np.arange(15), 6)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": groups,
            "y": (groups % 2).astype(int),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "g": "group", "y": "target"})
        .group_split(test_size=0.3, validation_size=0.3, random_state=0)
    )
    bundle = session.make_torch_loaders(batch_size=8, normalize=True, seed=0)
    assert bundle.report.split_kind == "group"
    assert bundle.report.groups_disjoint is True
    assert bundle.report.n_train == len(session.split_plan.train_indices)
    assert bundle.report.n_test == len(session.split_plan.test_indices)

    # Reconstruct group sets from partition frames and ensure disjointness.
    train_g = set(session.partition("train")["g"])
    valid_g = set(session.partition("validation")["g"])
    test_g = set(session.partition("test")["g"])
    assert not (train_g & valid_g)
    assert not (train_g & test_g)
    assert not (valid_g & test_g)

    stamps = pd.date_range("2024-01-01", periods=n, freq="h")
    tframe = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "t": stamps,
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )
    tsession = (
        Session.ingest(tframe)
        .set_roles({"x1": "feature", "x2": "feature", "t": "time", "y": "target"})
        .time_split(test_size=0.25, validation_size=0.25)
    )
    tbundle = tsession.make_torch_loaders(batch_size=8, shuffle_train=True, seed=0)
    assert tbundle.report.split_kind == "time"
    assert tbundle.report.time_order_ok is True
    assert any("Time split" in w for w in tbundle.report.warnings)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_resume_train_from_bundle(tmp_path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    session.make_torch_loaders(batch_size=16, seed=0)
    session.fit_torch(_tiny_module(), epochs=2, learning_rate=1e-2, device="cpu")
    first = session.dl_train_result
    assert first is not None
    assert first.n_epochs_ran == 2
    path = session.save_torch_bundle(tmp_path / "bundle")

    other = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    other.make_torch_loaders(batch_size=16, seed=0)
    other.load_torch_bundle(path, _tiny_module(), map_location="cpu", trusted=True)
    other.fit_torch(_tiny_module(), epochs=2, learning_rate=1e-2, device="cpu", resume=True)
    resumed = other.dl_train_result
    assert resumed is not None
    assert resumed.resumed_from_epochs == 2
    assert resumed.n_epochs_ran == 4
    assert len(resumed.history) >= 4
    assert resumed.optimizer_state is not None


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_training_curve_walkthrough_and_evaluate_diagnostics() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(80))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=3)
    )
    session.make_torch_loaders(batch_size=16, seed=3)
    session.fit_torch(
        _tiny_module(),
        epochs=3,
        learning_rate=1e-2,
        device="cpu",
        early_stopping_patience=5,
    )
    curve = session.torch_training_curve()
    assert curve.epochs
    assert curve.disclosures
    assert curve.limitations

    walk = session.walkthrough()
    assert walk.torch_training_status["enabled"] is True
    assert walk.torch_training_status["early_stop"]["partition"] == "validation"

    metrics = session.evaluate_torch(partition="test")
    assert metrics.confusion_matrix is not None
    assert metrics.n_rows > 0

    explanation = session.explain("torch_training_curve")
    assert explanation.operation == "torch_training_curve"
