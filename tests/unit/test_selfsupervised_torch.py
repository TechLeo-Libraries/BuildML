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
    restored.load_ssl_bundle(out, trusted=True)
    again = restored.evaluate_ssl(partition="test")
    assert again.metrics["accuracy"] == pytest.approx(ev.metrics["accuracy"])


def test_ssl_method_catalog() -> None:
    rows = list_ssl_methods()
    names = {r["method"] for r in rows}
    assert "simclr_tabular" in names
    assert "masked_tabular" in names


def test_vision_ssl_state_dict_roundtrip(tmp_path: Path) -> None:
    """Vision SSL encoder must restore Torch weights from encoder_torch.pt."""
    from buildml.core.errors import MissingExtraError
    from buildml.selfsupervised.checkpoint import load_ssl_bundle, save_ssl_bundle
    from buildml.selfsupervised.results import SelfSupervisedPlan
    from buildml.selfsupervised.torch.vision import VisionSSLEncoder

    rng = np.random.default_rng(0)
    images = [rng.random((3, 16, 16)).astype(np.float32) for _ in range(8)]
    encoder = VisionSSLEncoder(
        architecture="resnet18",
        weight_mode="mock",
        projector_dim=16,
        epochs=1,
        batch_size=4,
        image_size=(16, 16),
        random_state=0,
    )
    try:
        encoder.fit(images)
    except (MissingExtraError, OSError) as exc:
        pytest.skip(f"torch not usable for vision SSL on this host: {exc}")
    before = encoder.transform(images)
    latent = int(encoder.latent_dim)
    rep_cols = tuple(f"ssl_{i}" for i in range(latent))

    plan = SelfSupervisedPlan(
        method="vision_ssl",
        columns=("image",),
        n_train_rows=len(images),
        latent_dim=latent,
        representation_prefix="ssl_",
        representation_columns=rep_cols,
        encoder_=encoder,
        modality="vision",
        disclosures=("vision ssl unit roundtrip",),
    )

    out = tmp_path / "vision_ssl_bundle"
    save_ssl_bundle(out, plan)
    assert (out / "encoder_torch.pt").is_file()

    loaded_plan, _ = load_ssl_bundle(out, trusted=True)
    restored = loaded_plan.encoder_
    assert isinstance(restored, VisionSSLEncoder)
    assert restored._backbone is not None
    assert restored._projector is not None
    after = restored.transform(images)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
