"""Soft/hard materialization gate behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import check_materialization, estimate_dataframe_bytes


def test_soft_gate_warns_and_returns_telemetry() -> None:
    frame = pd.DataFrame({"a": range(1000), "b": range(1000)})
    nbytes = estimate_dataframe_bytes(frame)
    with pytest.warns(UserWarning, match="soft limit"):
        telemetry = check_materialization(
            frame,
            context="unit soft",
            soft_limit_bytes=max(1, nbytes - 1),
            hard_limit_bytes=None,
        )
    assert telemetry.soft_exceeded
    assert not telemetry.hard_exceeded
    assert telemetry.nbytes == nbytes
    assert telemetry.guidance
    assert telemetry.to_dict()["soft_exceeded"] is True


def test_hard_gate_refuses() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    nbytes = estimate_dataframe_bytes(frame)
    with pytest.raises(ValidationError, match="hard limit"):
        check_materialization(
            frame,
            context="unit hard",
            soft_limit_bytes=10**18,
            hard_limit_bytes=max(1, nbytes - 1),
            on_hard="error",
        )


def test_dataset_to_pandas_respects_hard_limit() -> None:
    frame = pd.DataFrame({"x": list(range(50))})
    dataset = Dataset.from_pandas(frame)
    nbytes = estimate_dataframe_bytes(dataset.frame)
    with pytest.raises(ValidationError, match="hard limit"):
        dataset.to_pandas(hard_limit_bytes=max(1, nbytes - 1))
