"""Pass O: speech FM path, multi-node DDP guards, managed serving."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.ddp import parse_torchrun_env

_TORCH_SPEC = importlib.util.find_spec("torch") is not None
_AUDIO_LEN = 256
_AUDIO_SR = 8_000


def _require_torch_or_skip() -> None:
    if not _TORCH_SPEC:
        pytest.skip("torch not installed")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch import failed: {exc}")


def _synthetic_wave(seed: int, n: int = _AUDIO_LEN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    freq = 3.0 + (seed % 5)
    return (0.4 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=n)).astype(
        np.float32
    )


def _speech_frame(n: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "audio": [_synthetic_wave(seed + i) for i in range(n)],
            "y": (rng.random(n) > 0.45).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# O2 speech
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_speech_classify_finetune_and_evaluate() -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(_speech_frame(48))
        .set_roles({"audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_speech_torch_loaders(
        audio_column="audio",
        sample_rate=_AUDIO_SR,
        max_samples=_AUDIO_LEN,
        batch_size=8,
        seed=0,
    )
    assert getattr(bundle, "modality", None) == "speech_classify"
    assert getattr(bundle, "speech_contract", None) is not None
    session.fit_speech_torch(epochs=2, device="cpu")
    assert session.dl_train_result is not None
    assert getattr(session.dl_train_result.module, "modality", None) == "speech_classify"
    ev = session.evaluate_torch(partition="validation")
    assert ev.n_rows > 0


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_transcribe_speech_stub_backend() -> None:
    _require_torch_or_skip()
    session = Session.ingest(_speech_frame(12)).set_roles(
        {"audio": "feature", "y": "target"}
    )
    result = session.transcribe_speech(
        audio_column="audio",
        backend="stub",
        sample_rate=_AUDIO_SR,
        max_samples=_AUDIO_LEN,
    )
    assert result.n_rows == 12
    assert result.backend == "stub"
    assert all(isinstance(t, str) and t.startswith("[stub-asr]") for t in result.texts)
    assert session.dl_speech_result is result


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_speech_train_only_amp_stats() -> None:
    _require_torch_or_skip()
    from buildml.dl.speech import make_speech_loaders
    from buildml.dl.types import FeatureContract  # noqa: F401

    session = (
        Session.ingest(_speech_frame(40, seed=3))
        .set_roles({"audio": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=3)
    )
    bundle = make_speech_loaders(
        session.dataset,
        session.split_plan,
        audio_column="audio",
        config=__import__("buildml.dl.speech", fromlist=["SpeechLoaderConfig"]).SpeechLoaderConfig(
            sample_rate=_AUDIO_SR,
            max_samples=_AUDIO_LEN,
            normalize_audio=True,
            batch_size=8,
            seed=3,
        ),
    )
    contract = bundle.speech_contract
    assert contract.audio_mean is not None
    assert contract.audio_std is not None
    assert contract.audio_std > 0


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_speech_ai_tools_registered() -> None:
    _require_torch_or_skip()
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    names = {t.name for t in registry.tools}
    assert "make_speech_torch_loaders" in names
    assert "fit_speech_torch" in names
    assert "transcribe_speech" in names


def test_speech_stack_missing_extra_message() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed")
    from buildml.dl.speech import require_speech_stack

    with pytest.raises(MissingExtraError) as exc:
        require_speech_stack()
    assert "speech" in str(exc.value).lower() or "buildml[speech]" in str(exc.value)


# ---------------------------------------------------------------------------
# O3 multi-node DDP
# ---------------------------------------------------------------------------


def test_parse_torchrun_env_ok() -> None:
    env = parse_torchrun_env(
        {
            "WORLD_SIZE": "4",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "10.0.0.2",
            "MASTER_PORT": "29501",
        }
    )
    assert env.world_size == 4
    assert env.rank == 1
    assert env.local_rank == 1
    assert env.master_addr == "10.0.0.2"
    assert env.master_port == "29501"


def test_parse_torchrun_env_missing_and_bad_rank() -> None:
    with pytest.raises(ValidationError, match="WORLD_SIZE"):
        parse_torchrun_env({"RANK": "0"})
    with pytest.raises(ValidationError, match="out of range"):
        parse_torchrun_env(
            {
                "WORLD_SIZE": "2",
                "RANK": "5",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
        )


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_multi_node_ddp_refuses_incomplete_env() -> None:
    _require_torch_or_skip()
    from buildml.dl.ddp import DDPConfig, train_supervised_module_ddp
    from buildml.dl.models import build_tabular_mlp

    session = (
        Session.ingest(
            pd.DataFrame(
                {
                    "x1": np.random.default_rng(0).normal(size=40),
                    "x2": np.random.default_rng(1).normal(size=40),
                    "y": (np.random.default_rng(2).random(40) > 0.5).astype(int),
                }
            )
        )
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    bundle = session.make_torch_loaders(batch_size=8, seed=0)
    with pytest.raises(ValidationError, match="torchrun|WORLD_SIZE|MASTER_ADDR"):
        train_supervised_module_ddp(
            lambda: build_tabular_mlp(2, task="classification", n_classes=2),
            bundle,
            ddp_config=DDPConfig(multi_node=True, allow_cpu_ddp=True),
            environ={"WORLD_SIZE": "2", "RANK": "0"},  # incomplete
        )


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_fit_torch_ddp_multi_node_flag_wires() -> None:
    _require_torch_or_skip()
    session = (
        Session.ingest(
            pd.DataFrame(
                {
                    "x1": np.linspace(-1, 1, 32),
                    "x2": np.linspace(0, 1, 32),
                    "y": [i % 2 for i in range(32)],
                }
            )
        )
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
    )
    session.make_torch_loaders(batch_size=8)
    from buildml.dl.models import build_tabular_mlp

    with pytest.raises(ValidationError, match="torchrun|WORLD_SIZE|MASTER"):
        session.fit_torch_ddp(
            lambda: build_tabular_mlp(2, task="classification", n_classes=2),
            epochs=1,
            multi_node=True,
            allow_cpu_ddp=True,
        )


# ---------------------------------------------------------------------------
# O4 managed serving
# ---------------------------------------------------------------------------


def test_serve_health_and_predict_pipeline(tmp_path: Path) -> None:
    fastapi_spec = importlib.util.find_spec("fastapi")
    if fastapi_spec is None:
        pytest.skip("fastapi not installed (buildml[serve])")

    from fastapi.testclient import TestClient
    from sklearn.linear_model import LogisticRegression

    from buildml.serving.app import clear_serving_state, create_serving_app

    frame = pd.DataFrame(
        {
            "x1": np.linspace(-1, 1, 40),
            "x2": np.linspace(0, 1, 40),
            "y": [i % 2 for i in range(40)],
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    bundle_dir = tmp_path / "pipe"
    session.save_pipeline(bundle_dir, evaluate_partition=None)

    try:
        app = create_serving_app(bundle_dir, kind="pipeline", title="PassO Serve")
        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is True
        assert body["product"] == "buildml-serve"
        assert body["auth"] is False
        assert body["kind"] == "pipeline"

        pred = client.post(
            "/predict",
            json={"rows": [{"x1": 0.1, "x2": 0.2}, {"x1": -0.3, "x2": 0.8}]},
        )
        assert pred.status_code == 200
        pdata = pred.json()
        assert pdata["ok"] is True
        assert pdata["n_rows"] == 2
        assert len(pdata["predictions"]) == 2
    finally:
        clear_serving_state()


def test_serve_missing_extra_message() -> None:
    # Only assert the error type when fastapi truly missing — otherwise smoke OK.
    if importlib.util.find_spec("fastapi") is not None:
        pytest.skip("fastapi installed")
    from buildml.serving.app import create_serving_app

    with pytest.raises(MissingExtraError) as exc:
        create_serving_app("unused", kind="pipeline")
    assert "serve" in str(exc.value).lower() or "buildml[serve]" in str(exc.value)
