"""Torch SSL unit tests (skip gracefully without torch)."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.selfsupervised.torch.catalog import list_ssl_methods

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(-1.0, 0.8, size=(n // 2, 3))
    x1 = rng.normal(1.5, 0.8, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b", "c"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


@pytest.mark.parametrize(
    "method",
    ["simclr_tabular", "byol_tabular", "vicreg_tabular", "mae_tabular", "vae_tabular"],
)
def test_torch_tabular_methods(method: str) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_ssl_pretext(
        method=method,
        latent_dim=6,
        epochs=8,
        batch_size=16,
        prefer_reduce_components=False,
        random_state=0,
    )
    assert fit.method == method
    assert fit.modality == "tabular"
    head = session.finetune_ssl_head()
    assert head.n_labeled_train > 0
    ev = session.evaluate_ssl(partition="test")
    assert "accuracy" in ev.metrics


def test_legacy_masked_tabular_deprecation() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        session.fit_ssl_pretext(method="masked_tabular", latent_dim=4, max_iter=40)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_ssl_bundle_v2_roundtrip(tmp_path: Path) -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_ssl_pretext(method="simclr_tabular", latent_dim=5, epochs=6, batch_size=16)
    session.finetune_ssl_head()
    ev = session.evaluate_ssl(partition="test")
    out = session.save_ssl_bundle(tmp_path / "ssl_v2")
    meta = (out / "meta.json").read_text(encoding="utf-8")
    assert "buildml.ssl_bundle.v2" in meta
    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    restored.load_ssl_bundle(out)
    again = restored.evaluate_ssl(partition="test")
    assert again.metrics["accuracy"] == pytest.approx(ev.metrics["accuracy"])


def test_ssl_method_catalog() -> None:
    rows = list_ssl_methods()
    names = {r["method"] for r in rows}
    assert "simclr_tabular" in names
    assert "masked_tabular" in names
