"""Unit coverage for the tabular Torch thin slice (skip-friendly)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _torch_usable() -> bool:
    """True only when torch imports without error (may be expensive)."""
    if not _TORCH_SPEC:
        return False
    try:
        from buildml.dl.extras import torch_available

        return torch_available()
    except Exception:
        return False


def _cls_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": np.asarray([0, 1] * (n // 2), dtype=np.int64),
        }
    )


def _session() -> Session:
    return (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )


def test_core_import_does_not_require_torch() -> None:
    import buildml
    import buildml.dl as dl

    assert hasattr(buildml, "Session")
    assert hasattr(dl, "torch_available")


def test_missing_torch_extra_message() -> None:
    if _torch_usable():
        pytest.skip("torch installed and importable in this environment")
    if _TORCH_SPEC and not _torch_usable():
        pytest.skip("torch package present but not importable (broken local wheel)")
    session = _session()
    with pytest.raises(MissingExtraError, match="buildml\\[torch\\]"):
        session.make_torch_loaders()


def test_catalog_covers_torch_operations() -> None:
    for name in (
        "make_torch_loaders",
        "fit_torch",
        "evaluate_torch",
        "save_torch_bundle",
        "load_torch_bundle",
        "torch_training_curve",
    ):
        assert name in OPERATION_CATALOG
    assert "batch-leakage" in OPERATION_CATALOG["make_torch_loaders"].concept_links
    assert "early-stopping-partition" in OPERATION_CATALOG["fit_torch"].concept_links
    assert "training-curves" in OPERATION_CATALOG["torch_training_curve"].concept_links


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_train_shuffle_only_and_normalize_train_fit() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    import torch

    from buildml.dl.dataset import build_feature_contract
    from buildml.dl.transforms import apply_standardize, fit_standardize

    session = _session()
    bundle = session.make_torch_loaders(batch_size=16, normalize=True, shuffle_train=True, seed=1)
    assert isinstance(bundle.loaders["train"].sampler, torch.utils.data.RandomSampler)
    assert isinstance(bundle.loaders["validation"].sampler, torch.utils.data.SequentialSampler)
    assert isinstance(bundle.loaders["test"].sampler, torch.utils.data.SequentialSampler)

    assert session.split_plan is not None
    contract, arrays = build_feature_contract(
        session.dataset,
        session.split_plan,
        normalize=True,
    )
    assert contract.normalize_mean is not None
    mean = np.asarray(contract.normalize_mean)
    std = np.asarray(contract.normalize_std)
    mean2, std2 = fit_standardize(
        session.partition("train")[list(contract.feature_columns)].to_numpy(dtype=float)
    )
    assert np.allclose(mean, mean2)
    assert np.allclose(std, std2)
    x_test_raw = session.partition("test")[list(contract.feature_columns)].to_numpy(dtype=float)
    expected = apply_standardize(x_test_raw, mean, std)
    assert np.allclose(arrays["test"][0], expected)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_fit_evaluate_and_bundle_roundtrip(tmp_path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    import torch
    from torch import nn

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    session = _session()
    session.make_torch_loaders(batch_size=16, normalize=True, seed=0)
    session.fit_torch(TinyMLP(), epochs=3, learning_rate=1e-2, device="cpu")
    assert session.dl_train_result is not None
    assert session.fit_result is None
    assert session.dl_train_result.n_epochs_ran == 3
    assert session.dl_train_result.history

    metrics = session.evaluate_torch(partition="test")
    assert metrics.n_rows > 0
    assert "accuracy" in metrics.metrics

    path = session.save_torch_bundle(tmp_path / "torch_bundle")
    assert (path / "meta.json").is_file()
    assert (path / "trainer.pt").is_file()

    other = _session()
    other.load_torch_bundle(path, TinyMLP(), map_location="cpu")
    assert other.dl_train_result is not None
    again = other.evaluate_torch(partition="test")
    assert again.metrics["accuracy"] == pytest.approx(metrics.metrics["accuracy"], abs=1e-5)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_bundle_rejects_wrong_format(tmp_path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    from torch import nn

    from buildml.dl.checkpoint import load_torch_bundle

    root = tmp_path / "not_torch"
    root.mkdir()
    (root / "meta.json").write_text('{"format": "buildml.pipeline_bundle.v2"}', encoding="utf-8")
    (root / "trainer.pt").write_bytes(b"nope")
    with pytest.raises(ValidationError, match="Unsupported trainer bundle format"):
        load_torch_bundle(root, nn.Linear(2, 2))


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_explain_fit_torch_catalog_hit() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = _session()
    explanation = session.explain("fit_torch")
    assert explanation.operation == "fit_torch"
    assert explanation.concept_notes
