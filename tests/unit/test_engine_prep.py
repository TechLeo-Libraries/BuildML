"""Engine-aware design-matrix projection and sampling disclosures."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.types import EngineName
from buildml.data.engines import prepare_design_frame
from buildml.ingest.detect import available_engines


def _wide_frame(n: int = 40, n_noise: int = 8) -> pd.DataFrame:
    data = {"y": ([0, 1] * (n // 2)), "signal": list(range(n))}
    for i in range(n_noise):
        data[f"noise_{i}"] = [float(i)] * n
    return pd.DataFrame(data)


def test_prepare_design_frame_projects_columns_pandas() -> None:
    frame = _wide_frame()
    session = Session.ingest(frame).set_roles(
        {"signal": "feature", "y": "target", **{f"noise_{i}": "ignore" for i in range(8)}}
    )
    result = prepare_design_frame(session.dataset, ["signal", "y"], sample_rows=10, random_state=0)
    assert list(result.columns_materialized) == ["signal", "y"]
    assert result.n_rows_materialized == 10
    assert result.sampled is True
    assert result.engine == "pandas"
    assert any("Projected" in tip for tip in result.disclosures)
    assert any("out-of-core" in tip for tip in result.limitations)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_prepare_design_frame_polars_projects_before_convert() -> None:
    frame = _wide_frame()
    session = (
        Session.ingest(frame)
        .set_roles({"signal": "feature", "y": "target"})
        .with_engine("polars")
    )
    result = session.prepare_design_matrix(columns=["signal", "y"], sample_rows=12, random_state=1)
    assert result.engine == "polars"
    assert result.used_native_handle is True
    assert list(result.columns_materialized) == ["signal", "y"]
    assert result.n_rows_materialized == 12
    assert any("native handle" in tip for tip in result.disclosures)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_prepare_design_frame_duckdb_projects() -> None:
    frame = _wide_frame()
    session = (
        Session.ingest(frame)
        .set_roles({"signal": "feature", "y": "target"})
        .with_engine("duckdb")
    )
    result = prepare_design_frame(session.dataset, ["signal", "y"])
    assert result.engine == "duckdb"
    assert result.sampled is False
    assert result.n_rows_materialized == len(frame)
    assert set(result.columns_materialized) == {"signal", "y"}


def test_session_prepare_design_matrix_partition() -> None:
    from sklearn.linear_model import LogisticRegression

    session = (
        Session.ingest(_wide_frame(60))
        .set_roles({"signal": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
    )
    prep = session.prepare_design_matrix(partition="train", sample_rows=None)
    train_n = len(session.split_plan.train_indices)  # type: ignore[union-attr]
    assert prep.n_rows_materialized == train_n
    assert "signal" in prep.columns_materialized
    assert "y" in prep.columns_materialized
    # Fit still works after prep disclosure (prep does not mutate dataset).
    session.fit(LogisticRegression(max_iter=200), task="classification")
    assert session.fit_result is not None
