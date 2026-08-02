"""Phase C DL depth: MLP zoo, plans bridge, CV, text modality."""

from __future__ import annotations

import importlib.util

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


def _cls_frame(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": np.asarray([0, 1] * (n // 2), dtype=np.int64),
        }
    )


def _text_frame(n: int = 60) -> pd.DataFrame:
    pos = ["great product loved it excellent quality"] * (n // 2)
    neg = ["bad item terrible waste poor quality"] * (n // 2)
    texts = pos + neg
    labels = [1] * (n // 2) + [0] * (n // 2)
    return pd.DataFrame({"text": texts, "y": labels})


def test_catalog_covers_phase_c_dl_ops() -> None:
    for name in (
        "make_text_torch_loaders",
        "cross_validate_torch",
        "make_torch_loaders",
        "fit_torch",
    ):
        assert name in OPERATION_CATALOG


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_builtin_mlp_happy_path() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_torch_loaders(batch_size=16, normalize=True)
    session.fit_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    assert session.dl_train_result.n_epochs_ran >= 1
    metrics = session.evaluate_torch(partition="validation")
    assert "accuracy" in metrics.metrics or "loss" in metrics.metrics


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_classical_plans_disclosed_on_loaders() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_cls_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale()
    )
    bundle = session.make_torch_loaders(batch_size=16, apply_plans=True)
    joined = " ".join(bundle.report.warnings)
    assert "Classical preprocess plans" in joined or "scale" in joined.lower()


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_cross_validate_torch_fold_local() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = Session.ingest(_cls_frame(n=60)).set_roles(
        {"x1": "feature", "x2": "feature", "y": "target"}
    )
    result = session.cross_validate_torch(n_folds=3, epochs=1, batch_size=16, device="cpu")
    assert result.n_folds == 3
    assert "accuracy" in result.mean_metrics or "loss" in result.mean_metrics
    assert any("nested" in lim.lower() for lim in result.limitations)
    assert session.dl_cv_result is result


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_text_torch_path() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = (
        Session.ingest(_text_frame())
        .set_roles({"text": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_text_torch_loaders(text_column="text", batch_size=8, max_len=16)
    assert getattr(bundle, "text_vocab", None) is not None
    session.fit_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    eval_result = session.evaluate_torch(partition="test")
    assert eval_result.n_rows > 0


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_text_vocab_refuses_without_split() -> None:
    if not _torch_usable():
        pytest.skip("torch not importable")
    session = Session.ingest(_text_frame()).set_roles({"text": "feature", "y": "target"})
    with pytest.raises(ValidationError):
        session.make_text_torch_loaders(text_column="text")
