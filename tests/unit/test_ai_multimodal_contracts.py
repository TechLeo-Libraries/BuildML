"""Agent-created media loaders must learn statistics from training rows only."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, **arguments):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, arguments, registry)
    result = execute_tool(session, proposal, confirmed=True, registry=registry)
    assert result.executed, result.error
    return result.result


@pytest.mark.parametrize("tool", ["make_multimodal_torch_loaders", "make_image_multimodal_torch_loaders",
                                  "make_audio_multimodal_torch_loaders"])
def test_media_statistics_are_train_only_and_all_modalities_train_on_cpu(tool):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    torch.manual_seed(53)
    count = 32
    frame = pd.DataFrame({
        "x": np.arange(count, dtype=float), "y": [0, 1] * (count // 2),
        "text": [f"sample token{i} positive" if i % 2 else f"sample token{i} negative" for i in range(count)],
        "image": [np.full((8, 8, 3), (i + 1) / 40, dtype=np.float32) for i in range(count)],
        "audio": [np.sin(np.arange(64) * np.pi / 8).astype(np.float32) * (i + 1) / 40
                  for i in range(count)],
    })
    session = (Session.ingest(frame)
               .set_roles({"x": "feature", "y": "target", "text": "feature",
                           "image": "feature", "audio": "feature"})
               .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=7))
    actual_frame = session.dataset._ensure_pandas()
    train = list(session.split_plan.train_indices)
    held_out = list(session.split_plan.test_indices) + list(session.split_plan.validation_indices)
    for index in held_out:
        actual_frame.at[index, "image"] = np.ones((8, 8, 3), dtype=np.float32)
        actual_frame.at[index, "audio"] = np.ones(64, dtype=np.float32) * 20
    image_before = np.stack(actual_frame.image).copy()
    audio_before = np.stack(actual_frame.audio).copy()
    _run(session, tool, batch_size=8, image_column="image", audio_column="audio", text_column="text",
         audio_sample_rate=8000, audio_max_samples=64, normalize_audio=True)
    bundle = session._torch_loaders
    contract = bundle.multimodal_contract
    expected_image_mean = image_before[train].mean(axis=(0, 1, 2), dtype=np.float64)
    np.testing.assert_allclose(contract.image_mean, expected_image_mean, rtol=1e-4)
    assert not np.allclose(contract.image_mean, image_before.mean(axis=(0, 1, 2), dtype=np.float64))
    assert list(bundle.input_layout) == ["numeric", "tokens", "image", "audio"]
    assert contract.audio_max_samples == 64
    _run(session, "fit_torch", epochs=1, learning_rate=0.001, device="cpu")
    scores = _run(session, "evaluate_torch", partition="test")
    assert scores.n_rows == len(session.split_plan.test_indices)
    assert np.isfinite(list(scores.metrics.values())).all()
    np.testing.assert_array_equal(np.stack(actual_frame.image), image_before)
    np.testing.assert_array_equal(np.stack(actual_frame.audio), audio_before)
