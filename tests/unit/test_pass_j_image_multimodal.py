"""Pass J: image multimodal fusion (path/array ⊕ tabular and/or text)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - broken wheel environments
        pytest.skip(f"torch import failed: {exc}")


def _synthetic_rgb(seed: int, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # HWC uint8-like floats in 0..1
    return rng.random((size[0], size[1], 3), dtype=np.float32)


def _image_tabular_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "image": [_synthetic_rgb(seed + i) for i in range(n)],
        "y": (rng.random(n) > 0.5).astype(int),
    }
    return pd.DataFrame(rows)


def _image_text_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    words_a = ["alpha cat dog", "beta bird fish", "gamma rock tree"]
    words_b = ["delta moon star", "epsilon river lake", "zeta cloud rain"]
    texts = [words_a[i % 3] if i % 2 == 0 else words_b[i % 3] for i in range(n)]
    return pd.DataFrame(
        {
            "text": texts,
            "image": [_synthetic_rgb(seed + i) for i in range(n)],
            "y": (rng.random(n) > 0.45).astype(int),
        }
    )


def _image_all_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    base = _image_tabular_frame(n, seed=seed)
    base["text"] = [f"sample row {i} class token" for i in range(n)]
    return base


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_tabular_image_fusion_fit_evaluate() -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_image_tabular_frame(48))
        .set_roles(
            {"x1": "feature", "x2": "feature", "image": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_image_multimodal_torch_loaders(
        image_column="image",
        image_size=(8, 8),
        batch_size=8,
        seed=0,
    )
    assert getattr(bundle, "modality", None) == "tabular_image_fusion"
    assert list(getattr(bundle, "input_layout", ())) == ["numeric", "image"]
    session.fit_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    mod = session.dl_train_result.module
    assert getattr(mod, "image_channels", 0) == 3
    assert getattr(mod, "n_numeric", 0) == 2
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_text_image_and_triple_fusion() -> None:
    _require_torch_or_skip()
    text_img = (
        Session.ingest(_image_text_frame(48))
        .set_roles({"text": "feature", "image": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=1)
    )
    b1 = text_img.make_image_multimodal_torch_loaders(
        image_column="image", text_column="text", image_size=(8, 8), batch_size=8
    )
    assert getattr(b1, "modality", None) == "text_image_fusion"
    text_img.fit_torch(epochs=1, device="cpu")

    triple = (
        Session.ingest(_image_all_frame(48))
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "text": "feature",
                "image": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=2)
    )
    b2 = triple.make_multimodal_torch_loaders(
        image_column="image",
        text_column="text",
        image_size=(8, 8),
        batch_size=8,
        max_len=16,
    )
    assert getattr(b2, "modality", None) == "tabular_text_image_fusion"
    assert list(getattr(b2, "input_layout", ())) == ["numeric", "tokens", "image"]
    triple.fit_torch(epochs=1, device="cpu")
    assert triple.dl_train_result is not None


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_image_channel_stats_are_train_only() -> None:
    _require_torch_or_skip()
    from buildml.dl.image import fit_image_channel_stats, stack_image_column

    session = (
        Session.ingest(_image_tabular_frame(60))
        .set_roles(
            {"x1": "feature", "x2": "feature", "image": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    frame = session.dataset._ensure_pandas()
    test_idx = list(session._split_plan.indices_for("test"))
    # Poison holdout images with a distinct constant so full-frame stats would shift.
    for i in test_idx:
        frame.at[i, "image"] = np.ones((8, 8, 3), dtype=np.float32)

    bundle = session.make_image_multimodal_torch_loaders(
        image_column="image", image_size=(8, 8), batch_size=8, seed=0
    )
    contract = bundle.multimodal_contract
    assert contract.image_mean is not None
    train_idx = list(session._split_plan.indices_for("train"))
    train_images = stack_image_column(
        frame.iloc[train_idx]["image"].tolist(), size=(8, 8), channels=3
    )
    mean, std = fit_image_channel_stats(train_images)
    assert np.allclose(contract.image_mean, mean)
    assert np.allclose(contract.image_std, std)
    # Full-frame mean must differ because holdout is all-ones.
    full_images = stack_image_column(frame["image"].tolist(), size=(8, 8), channels=3)
    full_mean, _ = fit_image_channel_stats(full_images)
    assert not np.allclose(contract.image_mean, full_mean)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_path_image_cells_and_export(tmp_path: Path) -> None:
    _require_torch_or_skip()
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    import torch

    from buildml.dl.export import load_torchscript

    n = 40
    paths: list[str] = []
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = (rng.random((12, 12, 3)) * 255).astype(np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "image": paths,
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "image": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_image_multimodal_torch_loaders(
        image_column="image", image_size=(8, 8), batch_size=8, seed=0
    )
    session.fit_torch(epochs=1, device="cpu")
    mod = session.dl_train_result.module.cpu().eval()
    batch = next(iter(session._torch_loaders.loaders["train"]))
    x_tab, images, _y = batch
    with torch.no_grad():
        y_tuple = mod((x_tab, images))
        y_args = mod(x_tab, images)
    assert y_tuple.shape == y_args.shape
    out = tmp_path / "img_mm.ts.pt"
    result = session.export_torch(out, format="torchscript")
    assert result.path.exists()
    loaded = load_torchscript(result.path)
    with torch.no_grad():
        y_loaded = loaded(x_tab.cpu(), images.cpu())
    assert y_loaded.shape == y_args.shape


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_export_refuses_silent_tabular_rebuild_after_image_fit(tmp_path: Path) -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_image_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "image": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_image_multimodal_torch_loaders(
        image_column="image", image_size=(8, 8), batch_size=8, seed=0
    )
    session.fit_torch(epochs=1, device="cpu")
    session._torch_loaders = None
    with pytest.raises(ValidationError, match="Refusing silent tabular loader rebuild"):
        session.export_torch(tmp_path / "bad.ts.pt", format="torchscript")


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_image_alone_refused() -> None:
    _require_torch_or_skip()
    frame = pd.DataFrame(
        {
            "image": [_synthetic_rgb(i) for i in range(24)],
            "y": [0, 1] * 12,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"image": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="tabular numeric|text column"):
        session.make_image_multimodal_torch_loaders(image_column="image", image_size=(8, 8))
