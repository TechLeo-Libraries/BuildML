"""Pass L: audio multimodal fusion (path/waveform ⊕ tabular and/or text and/or image)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_AUDIO_LEN = 256
_AUDIO_SR = 8_000


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - broken wheel environments
        pytest.skip(f"torch import failed: {exc}")


def _synthetic_wave(seed: int, n: int = _AUDIO_LEN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    freq = 3.0 + (seed % 5)
    return (0.4 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=n)).astype(
        np.float32
    )


def _synthetic_rgb(seed: int, size: tuple[int, int] = (8, 8)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size[0], size[1], 3), dtype=np.float32)


def _audio_tabular_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "audio": [_synthetic_wave(seed + i) for i in range(n)],
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )


def _audio_text_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    words_a = ["alpha cat dog", "beta bird fish", "gamma rock tree"]
    words_b = ["delta moon star", "epsilon river lake", "zeta cloud rain"]
    texts = [words_a[i % 3] if i % 2 == 0 else words_b[i % 3] for i in range(n)]
    return pd.DataFrame(
        {
            "text": texts,
            "audio": [_synthetic_wave(seed + i) for i in range(n)],
            "y": (rng.random(n) > 0.45).astype(int),
        }
    )


def _audio_all_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    base = _audio_tabular_frame(n, seed=seed)
    base["text"] = [f"sample row {i} class token" for i in range(n)]
    base["image"] = [_synthetic_rgb(seed + i) for i in range(n)]
    return base


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_tabular_audio_fusion_fit_evaluate() -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_audio_tabular_frame(48))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    assert getattr(bundle, "modality", None) == "tabular_audio_fusion"
    assert list(getattr(bundle, "input_layout", ())) == ["numeric", "audio"]
    session.fit_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    mod = session.dl_train_result.module
    assert getattr(mod, "audio_channels", 0) == 1
    assert getattr(mod, "n_numeric", 0) == 2
    assert hasattr(mod, "audio_net")
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_text_audio_and_richer_fusion() -> None:
    _require_torch_or_skip()
    text_aud = (
        Session.ingest(_audio_text_frame(48))
        .set_roles({"text": "feature", "audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=1)
    )
    b1 = text_aud.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        text_column="text",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
    )
    assert getattr(b1, "modality", None) == "text_audio_fusion"
    text_aud.fit_torch(epochs=1, device="cpu")

    rich = (
        Session.ingest(_audio_all_frame(48))
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "text": "feature",
                "image": "feature",
                "audio": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=2)
    )
    b2 = rich.make_multimodal_torch_loaders(
        audio_column="audio",
        image_column="image",
        text_column="text",
        image_size=(8, 8),
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        max_len=16,
    )
    assert getattr(b2, "modality", None) == "tabular_text_image_audio_fusion"
    assert list(getattr(b2, "input_layout", ())) == [
        "numeric",
        "tokens",
        "image",
        "audio",
    ]
    rich.fit_torch(epochs=1, device="cpu")
    assert rich.dl_train_result is not None


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_audio_waveform_stats_are_train_only() -> None:
    _require_torch_or_skip()
    from buildml.dl.audio import fit_audio_waveform_stats, stack_audio_column

    session = (
        Session.ingest(_audio_tabular_frame(60))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    frame = session.dataset._ensure_pandas()
    test_idx = list(session._split_plan.indices_for("test"))
    for i in test_idx:
        frame.at[i, "audio"] = np.ones(_AUDIO_LEN, dtype=np.float32)

    bundle = session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    contract = bundle.multimodal_contract
    assert contract.audio_mean is not None
    train_idx = list(session._split_plan.indices_for("train"))
    train_audio = stack_audio_column(
        frame.iloc[train_idx]["audio"].tolist(),
        sample_rate=_AUDIO_SR,
        max_samples=_AUDIO_LEN,
    )
    mean, std = fit_audio_waveform_stats(train_audio)
    assert np.allclose(contract.audio_mean, mean)
    assert np.allclose(contract.audio_std, std)
    full_audio = stack_audio_column(
        frame["audio"].tolist(), sample_rate=_AUDIO_SR, max_samples=_AUDIO_LEN
    )
    full_mean, _ = fit_audio_waveform_stats(full_audio)
    assert not np.allclose(contract.audio_mean, full_mean)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_path_audio_cells_and_export(tmp_path: Path) -> None:
    _require_torch_or_skip()
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    import torch

    from buildml.dl.export import load_torchscript

    n = 40
    paths: list[str] = []
    rng = np.random.default_rng(0)
    for i in range(n):
        wave = _synthetic_wave(i, n=_AUDIO_LEN)
        p = tmp_path / f"aud_{i}.wav"
        sf.write(str(p), wave, _AUDIO_SR)
        paths.append(str(p))
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "audio": paths,
            "y": (rng.random(n) > 0.5).astype(int),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    session.fit_torch(epochs=1, device="cpu")
    mod = session.dl_train_result.module.cpu().eval()
    batch = next(iter(session._torch_loaders.loaders["train"]))
    x_tab, audio, _y = batch
    with torch.no_grad():
        y_tuple = mod((x_tab, audio))
        y_args = mod(x_tab, audio)
    assert y_tuple.shape == y_args.shape
    out = tmp_path / "aud_mm.ts.pt"
    result = session.export_torch(out, format="torchscript")
    assert result.path.exists()
    loaded = load_torchscript(result.path)
    with torch.no_grad():
        y_loaded = loaded(x_tab.cpu(), audio.cpu())
    assert y_loaded.shape == y_args.shape


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_export_refuses_silent_tabular_rebuild_after_audio_fit(tmp_path: Path) -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    session.fit_torch(epochs=1, device="cpu")
    session._torch_loaders = None
    with pytest.raises(ValidationError, match="Refusing silent tabular loader rebuild"):
        session.export_torch(tmp_path / "bad.ts.pt", format="torchscript")


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_evaluate_and_fit_refuse_silent_tabular_rebuild_after_audio_fit() -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    session.fit_torch(epochs=1, device="cpu")
    session._torch_loaders = None
    with pytest.raises(ValidationError, match="Refusing silent tabular loader rebuild"):
        session.evaluate_torch(partition="validation")
    with pytest.raises(ValidationError, match="Refusing silent tabular loader rebuild"):
        session.fit_torch(epochs=1, device="cpu")


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_media_path_column_not_inferred_as_text() -> None:
    _require_torch_or_skip()
    n = 24
    frame = pd.DataFrame(
        {
            "x1": np.random.default_rng(0).normal(size=n),
            "audio": [f"clip_{i}.wav" for i in range(n)],
            "y": [0, 1] * 12,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="audio_column=|image_column="):
        session.make_multimodal_torch_loaders(batch_size=4)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_short_clip_repeat_pad_preserves_pool_signal() -> None:
    """Default-length windows must not wipe short waveforms via zero-pad+GAP."""
    _require_torch_or_skip()
    import torch
    import torch.nn as nn

    from buildml.dl.audio import (
        apply_audio_waveform_stats,
        decode_audio_cell,
        fit_audio_waveform_stats,
        stack_audio_column,
    )

    short = np.ones(256, dtype=np.float32) * 0.5
    cell = decode_audio_cell(short, sample_rate=16_000, max_samples=16_000)
    assert cell.shape == (1, 16_000)
    # Repeat-pad: amplitude preserved across the window (not diluted by zeros).
    assert float(np.abs(cell.mean() - 0.5)) < 1e-5
    batch, lengths = stack_audio_column(
        [short, short * 0.8],
        sample_rate=16_000,
        max_samples=16_000,
        return_lengths=True,
    )
    assert lengths.tolist() == [256, 256]
    mean, std = fit_audio_waveform_stats(batch, lengths=lengths)
    # Length-aware stats match the short-clip region, not a zero-dominated window.
    short_batch = np.stack([short.reshape(1, -1), (short * 0.8).reshape(1, -1)])
    mean_s, std_s = fit_audio_waveform_stats(short_batch.reshape(2, 1, 256))
    assert np.allclose(mean, mean_s)
    assert np.allclose(std, std_s)
    norm = apply_audio_waveform_stats(batch, mean, std)
    pooled = nn.AdaptiveAvgPool1d(1)(torch.as_tensor(norm)).reshape(-1)
    assert float(pooled.abs().min()) > 0.5


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_audio_alone_refused() -> None:
    _require_torch_or_skip()
    frame = pd.DataFrame(
        {
            "audio": [_synthetic_wave(i) for i in range(24)],
            "y": [0, 1] * 12,
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    with pytest.raises(ValidationError, match="tabular numeric|text column|media"):
        session.make_audio_multimodal_torch_loaders(
            audio_column="audio",
            audio_sample_rate=_AUDIO_SR,
            audio_max_samples=_AUDIO_LEN,
        )


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_audio_multimodal_onnx_export(tmp_path: Path) -> None:
    _require_torch_or_skip()
    pytest.importorskip("onnx")
    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    session.fit_torch(epochs=1, device="cpu")
    out = tmp_path / "aud_mm.onnx"
    result = session.export_torch(out, format="onnx", opset=17)
    assert result.path.exists()
    assert result.format == "onnx"
    assert result.path.stat().st_size > 0
    assert result.meta.get("input_names") == ["numeric", "audio"]
    assert result.meta.get("input_layout") == ["numeric", "audio"]


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_audio_loader_bundle_slots_hold_modality_metadata() -> None:
    _require_torch_or_skip()
    from buildml.dl.results import TorchLoaderBundle

    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    assert isinstance(bundle, TorchLoaderBundle)
    assert bundle.modality == "tabular_audio_fusion"
    assert bundle.multimodal_contract is not None
    assert bundle.multimodal_contract.audio_column == "audio"
    assert bundle.input_layout == ("numeric", "audio")
    bundle.modality = "tabular_audio_fusion"
    assert session._torch_loaders.modality == "tabular_audio_fusion"


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_ambiguous_2d_audio_layout_refused() -> None:
    _require_torch_or_skip()
    from buildml.dl.audio import decode_audio_cell

    with pytest.raises(ValidationError, match="Ambiguous 2D audio array"):
        decode_audio_cell(np.ones((16, 16), dtype=np.float32), max_samples=64)
    with pytest.raises(ValidationError, match="Ambiguous 2D audio array"):
        decode_audio_cell(np.ones((4, 4), dtype=np.float32), max_samples=64)
    # Clear channel layouts still accepted.
    ch_first = decode_audio_cell(np.ones((2, 32), dtype=np.float32), max_samples=32)
    assert ch_first.shape == (1, 32)
    ch_last = decode_audio_cell(np.ones((32, 2), dtype=np.float32), max_samples=32)
    assert ch_last.shape == (1, 32)


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_torch_bundle_persists_audio_preprocess_meta(tmp_path: Path) -> None:
    _require_torch_or_skip()
    import json

    from buildml.dl.multimodal import build_multimodal_fusion

    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_audio_multimodal_torch_loaders(
        audio_column="audio",
        audio_sample_rate=_AUDIO_SR,
        audio_max_samples=_AUDIO_LEN,
        audio_source_sample_rate=_AUDIO_SR,
        batch_size=8,
        seed=0,
    )
    session.fit_torch(epochs=1, device="cpu")
    assert session.dl_train_result is not None
    meta_fit = session.dl_train_result.multimodal_preprocess
    assert meta_fit is not None
    assert meta_fit["audio_column"] == "audio"
    assert meta_fit["audio_sample_rate"] == _AUDIO_SR
    assert meta_fit["audio_max_samples"] == _AUDIO_LEN
    assert meta_fit["audio_source_sample_rate"] == _AUDIO_SR
    assert meta_fit["audio_mean"] is not None
    assert meta_fit["input_layout"] == ["numeric", "audio"]

    path = session.save_torch_bundle(tmp_path / "aud_bundle")
    disk_meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert disk_meta["multimodal_preprocess"]["audio_mean"] == meta_fit["audio_mean"]

    shell = build_multimodal_fusion(
        n_numeric=2,
        vocab_size=0,
        audio_channels=1,
        audio_samples=_AUDIO_LEN,
        task="classification",
        n_classes=2,
    )
    other = (
        Session.ingest(_audio_tabular_frame(40, seed=1))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    other.load_torch_bundle(path, shell, map_location="cpu")
    assert other.dl_train_result is not None
    loaded = other.dl_train_result.multimodal_preprocess
    assert loaded is not None
    assert loaded["audio_mean"] == meta_fit["audio_mean"]
    assert loaded["audio_source_sample_rate"] == _AUDIO_SR
    assert any("DataLoaders were not rebuilt" in w for w in other.dl_train_result.warnings)
    # Load-path honesty: evaluate refuses silent tabular rebuild.
    other._torch_loaders = None
    with pytest.raises(ValidationError, match="Refusing silent tabular loader rebuild"):
        other.evaluate_torch(partition="validation")


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_ai_executor_dispatches_make_audio_multimodal_torch_loaders() -> None:
    """Registry tool must reach Session, not die as missing dispatch."""
    _require_torch_or_skip()
    from buildml.ai.executor import execute_tool, propose_tool_execution
    from buildml.ai.tools import build_default_registry

    session = (
        Session.ingest(_audio_tabular_frame(40))
        .set_roles(
            {"x1": "feature", "x2": "feature", "audio": "feature", "y": "target"}
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
        .ai_configure(provider="mock")
    )
    registry = build_default_registry()
    assert registry.get("make_audio_multimodal_torch_loaders") is not None
    proposal = propose_tool_execution(
        "make_audio_multimodal_torch_loaders",
        {
            "audio_column": "audio",
            "batch_size": 8,
            "normalize_audio": True,
            "audio_sample_rate": _AUDIO_SR,
            "audio_max_samples": _AUDIO_LEN,
        },
        registry,
    )
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.error is None, result.error
    assert session._torch_loaders is not None
    assert session._torch_loaders.modality == "tabular_audio_fusion"
    assert session._torch_loaders.multimodal_contract.audio_max_samples == _AUDIO_LEN
    assert session._torch_loaders.multimodal_contract.audio_sample_rate == _AUDIO_SR
