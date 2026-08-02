"""Pass G DL depth: nested search, multimodal, AMP, export, DDP skip path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.explain.catalog import OPERATION_CATALOG

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _torch_usable() -> bool:
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


def _mm_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    texts = ["good product alpha", "bad item beta", "good widget", "bad gadget"] * (n // 4 + 1)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "text": texts[:n],
            "y": np.asarray([0, 1] * (n // 2), dtype=np.int64)[:n],
        }
    )


def test_catalog_covers_pass_g_ops() -> None:
    for name in (
        "nested_cv_torch",
        "search_torch",
        "make_multimodal_torch_loaders",
        "export_torch",
        "fit_torch_ddp",
        "ai_run_autonomous",
    ):
        assert name in OPERATION_CATALOG


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_search_torch_and_nested_cv_tiny() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(72))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    search = session.search_torch(
        param_grid={"learning_rate": [1e-2, 1e-3], "hidden": [(8,), (16, 8)]},
        n_folds=2,
        epochs=1,
        batch_size=16,
        device="cpu",
        seed=0,
    )
    assert search.best_params
    assert "accuracy" in search.best_metrics or "loss" in search.best_metrics
    assert "test" in search.held_out_partitions
    assert any(
        "not a nested" in lim.lower() or "inner-style" in lim.lower()
        for lim in search.limitations
    )

    nested = session.nested_cv_torch(
        param_grid={"learning_rate": [1e-2, 1e-3]},
        outer_cv=2,
        inner_cv=2,
        epochs=1,
        batch_size=16,
        device="cpu",
        seed=0,
    )
    assert nested.n_outer_folds == 2
    assert nested.mean_metrics
    assert session.dl_nested_cv_result is nested
    assert "test" in nested.held_out_partitions
    assert any("outer" in lim.lower() for lim in nested.limitations)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_search_torch_requires_space() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(40))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, random_state=0)
    )
    with pytest.raises(ValidationError, match="param_grid|param_distributions"):
        session.search_torch(epochs=1, n_folds=2, device="cpu")


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_multimodal_fusion_fit_evaluate() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_mm_frame(48))
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "text": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_multimodal_torch_loaders(batch_size=8, max_len=16, seed=0)
    assert getattr(bundle, "modality", None) == "tabular_text_fusion"
    session.fit_torch(epochs=2, device="cpu", mixed_precision=True)
    assert session.dl_train_result is not None
    assert any("AMP" in w or "mixed_precision" in w for w in session.dl_train_result.warnings)
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0
    assert "accuracy" in ev.metrics or "loss" in ev.metrics


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_export_torchscript_roundtrip(tmp_path: Path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    import torch

    from buildml.dl.export import load_torchscript

    session = (
        Session.ingest(_cls_frame(40))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    session.make_torch_loaders(batch_size=8, seed=0)
    session.fit_torch(epochs=1, device="cpu")
    out = tmp_path / "model.ts.pt"
    result = session.export_torch(out, format="torchscript")
    assert result.path.exists()
    loaded = load_torchscript(result.path)
    xb, _ = next(iter(session._torch_loaders.loaders["train"]))
    with torch.no_grad():
        y1 = session.dl_train_result.module.cpu()(xb.cpu())
        y2 = loaded(xb.cpu())
    assert y1.shape == y2.shape


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_export_onnx_smoke(tmp_path: Path) -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame(40))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    session.make_torch_loaders(batch_size=8, seed=0)
    session.fit_torch(epochs=1, device="cpu")
    out = tmp_path / "model.onnx"
    result = session.export_torch(out, format="onnx", opset=17)
    assert result.path.exists()
    assert result.opset == 17
    assert result.format == "onnx"


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_ddp_refuses_without_multi_gpu_unless_opt_in() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    from buildml.dl.ddp import ddp_cuda_device_count
    from buildml.dl.models import build_tabular_mlp

    session = (
        Session.ingest(_cls_frame(40))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )
    session.make_torch_loaders(batch_size=8, seed=0)

    def factory():
        return build_tabular_mlp(2, task="classification", n_classes=2, hidden=(8,))

    if ddp_cuda_device_count() < 2:
        with pytest.raises(ValidationError, match="device_count|allow_cpu_ddp"):
            session.fit_torch_ddp(factory, epochs=1, world_size=2)
    else:
        result = session.fit_torch_ddp(factory, epochs=1)
        assert result.train_result is not None
        assert result.world_size >= 2
