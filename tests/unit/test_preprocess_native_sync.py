"""Preprocess transforms keep or rebuild Dataset.native for engine paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import prepare_design_frame
from buildml.ingest.detect import available_engines


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 8, n),
            "income": rng.normal(50, 12, n),
            "city": rng.choice(["a", "b", "c"], n),
            "y": ([0, 1] * (n // 2)),
        }
    )


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_impute_scale_encode_keep_native_usable() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"age": "feature", "income": "feature", "city": "feature", "y": "target"})
        .with_engine("polars")
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    assert session.dataset.has_native
    session.impute(strategy="median")
    assert session.dataset.has_native
    session.encode(columns=["city"], method="onehot")
    assert session.dataset.has_native
    session.scale(method="standard")
    assert session.dataset.has_native

    # Native handle matches the transformed Pandas frame.
    native_pd = session.dataset.to_engine("polars").to_pandas()
    frame = session.dataset.to_pandas()
    assert list(native_pd.columns) == list(frame.columns)
    assert len(native_pd) == len(frame)

    features = [c for c in session.dataset.columns if c != "y"]
    prep = prepare_design_frame(session.dataset, features, sample_rows=10, random_state=1)
    assert prep.used_native_handle is True


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_drop_and_select_rebuild_duckdb_native() -> None:
    session = (
        Session.ingest(_frame(24))
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .with_engine("duckdb")
        .split(test_size=0.25, stratify=True, random_state=1)
    )
    session.drop_columns(["city"])
    assert session.dataset.has_native
    assert "city" not in session.dataset.columns
    session.select_features(strategy="variance", threshold=0.0)
    # select_features may keep all numeric columns; native must remain attached.
    assert session.dataset.has_native
    projected = session.dataset.project(["age", "y"])
    assert projected.has_native
    assert projected.columns == ["age", "y"]


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_sync_native_session_helper() -> None:
    session = Session.ingest(_frame(12)).with_engine("polars")
    session.dataset.invalidate_native()
    assert not session.dataset.has_native
    session.sync_native()
    assert session.dataset.has_native
