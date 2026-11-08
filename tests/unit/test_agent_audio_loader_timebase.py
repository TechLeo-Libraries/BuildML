"""Agent-created multimodal loaders preserve the recording timebase."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry

TOOLS = (
    "make_multimodal_torch_loaders",
    "make_image_multimodal_torch_loaders",
    "make_audio_multimodal_torch_loaders",
)


@pytest.fixture
def session():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    frame = pd.DataFrame(
        {
            "image": [np.full((6, 6, 3), 0.25, dtype=np.float32) for _ in range(8)],
            "audio": [np.array([0.0, 1.0, 0.0], dtype=np.float32) for _ in range(8)],
            "y": [0, 1] * 4,
        }
    )
    return (
        Session.ingest(frame)
        .set_roles({"image": "feature", "audio": "feature", "y": "target"})
        .inject_split(train_indices=[0, 1, 2, 3], validation_indices=[4, 5], test_indices=[6, 7])
    )


def _execute(session, tool, **overrides):
    registry = build_default_registry()
    arguments = {
        "image_column": "image",
        "audio_column": "audio",
        "audio_sample_rate": 16000,
        "audio_source_sample_rate": 8000,
        "audio_max_samples": 8,
        "normalize_audio": False,
        **overrides,
    }
    proposal = propose_tool_execution(tool, arguments, registry)
    return execute_tool(session, proposal, confirmed=True, registry=registry)


def test_provider_schema_exposes_array_source_rate_for_every_media_loader():
    schema = {
        tool["function"]["name"]: tool["function"]
        for tool in build_default_registry().to_openai_tools()
    }
    for name in TOOLS:
        parameters = schema[name]["parameters"]
        assert parameters["properties"]["audio_source_sample_rate"]["type"] == "integer"
        assert "audio_source_sample_rate" not in parameters["required"]


@pytest.mark.parametrize("tool", TOOLS)
def test_agent_loader_resamples_array_on_recorded_timebase(session, tool):
    result = _execute(session, tool)
    assert result.executed, result.error
    bundle = session._torch_loaders
    audio_index = bundle.input_layout.index("audio")
    batch = next(iter(bundle.loaders["test"]))
    # Half-rate timestamps add midpoints, then the six-sample clip repeats.
    expected = np.array([0, 0.5, 1, 0.5, 0, 0, 0, 0.5], dtype=np.float32)
    np.testing.assert_array_equal(batch[audio_index].numpy()[:, 0], [expected, expected])
    np.testing.assert_array_equal(batch[-1].numpy(), [0, 1])
    assert bundle.multimodal_contract.audio_source_sample_rate == 8000
    assert bundle.multimodal_contract.audio_sample_rate == 16000


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("audio_sample_rate", True),
        ("audio_source_sample_rate", 8000.5),
        ("audio_source_sample_rate", 0),
        ("audio_max_samples", 5.75),
    ],
)
def test_invalid_audio_parameters_cannot_replace_an_existing_loader(
    session, tool, parameter, value
):
    initial = _execute(session, tool)
    assert initial.executed, initial.error
    original = session._torch_loaders
    result = _execute(session, tool, **{parameter: value})
    assert not result.executed
    assert parameter.removeprefix("audio_") in result.error
    assert session._torch_loaders is original
