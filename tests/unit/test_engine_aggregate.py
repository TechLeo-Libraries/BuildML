"""Dataset/engine projection and aggregation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.core.types import EngineName
from buildml.data.dataset import Dataset
from buildml.data.engines import get_engine
from buildml.data.engines.aggregate import canonicalize_agg_func, quantile_level
from buildml.ingest.detect import available_engines


def _sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["east", "east", "west", "west", "west"],
            "amount": [10.0, 20.0, 5.0, 15.0, 10.0],
            "qty": [1, 2, 1, 1, 3],
        }
    )


def test_pandas_project_and_aggregate() -> None:
    ds = Dataset.from_pandas(_sales())
    projected = ds.project(["region", "amount"])
    assert projected.columns == ["region", "amount"]
    summary = projected.aggregate(
        {"amount": ["mean", "sum"], "*": "count"},
        by=["region"],
    )
    frame = summary.to_pandas().sort_values("region").reset_index(drop=True)
    assert list(frame.columns) == ["region", "amount_mean", "amount_sum", "count"]
    east = frame.loc[frame["region"] == "east"].iloc[0]
    assert float(east["amount_mean"]) == pytest.approx(15.0)
    assert float(east["amount_sum"]) == pytest.approx(30.0)
    assert int(east["count"]) == 2


def test_pandas_global_aggregate() -> None:
    ds = Dataset.from_pandas(_sales())
    out = ds.aggregate({"amount": "mean", "qty": "sum", "*": "count"}).to_pandas()
    assert len(out) == 1
    assert float(out.loc[0, "amount_mean"]) == pytest.approx(12.0)
    assert float(out.loc[0, "qty_sum"]) == pytest.approx(8.0)
    assert int(out.loc[0, "count"]) == 5


def test_pandas_median_and_quantiles() -> None:
    ds = Dataset.from_pandas(_sales())
    out = ds.aggregate({"amount": ["median", "q25", "q75"]}).to_pandas()
    assert float(out.loc[0, "amount_median"]) == pytest.approx(10.0)
    assert float(out.loc[0, "amount_q25"]) == pytest.approx(10.0)
    assert float(out.loc[0, "amount_q75"]) == pytest.approx(15.0)
    assert canonicalize_agg_func("quantile_0.25") == "q25"
    assert canonicalize_agg_func("quantile_0.5") == "median"
    assert quantile_level("q90") == pytest.approx(0.9)


def test_aggregate_grouped_median() -> None:
    ds = Dataset.from_pandas(_sales())
    frame = (
        ds.aggregate({"amount": "median"}, by=["region"])
        .to_pandas()
        .sort_values("region")
        .reset_index(drop=True)
    )
    east = frame.loc[frame["region"] == "east"].iloc[0]
    west = frame.loc[frame["region"] == "west"].iloc[0]
    assert float(east["amount_median"]) == pytest.approx(15.0)
    assert float(west["amount_median"]) == pytest.approx(10.0)


def test_aggregate_rejects_unknown_function() -> None:
    ds = Dataset.from_pandas(_sales())
    with pytest.raises(ValidationError, match="Unsupported aggregate"):
        ds.aggregate({"amount": "mode"})
    with pytest.raises(ValidationError, match="Unsupported aggregate|integer percentiles"):
        ds.aggregate({"amount": "quantile_0.333"})


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_native_project_then_aggregate() -> None:
    session = Session.ingest(_sales()).with_engine("polars")
    projected = session.dataset.project(["region", "amount", "qty"])
    assert projected.has_native
    summary = projected.aggregate({"amount": "sum", "qty": "mean"}, by=["region"])
    assert summary.has_native
    frame = summary.to_pandas().sort_values("region").reset_index(drop=True)
    west = frame.loc[frame["region"] == "west"].iloc[0]
    assert float(west["amount_sum"]) == pytest.approx(30.0)
    assert float(west["qty_mean"]) == pytest.approx(5.0 / 3.0)


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_polars_native_median_quantile() -> None:
    session = Session.ingest(_sales()).with_engine("polars")
    summary = session.dataset.aggregate({"amount": ["median", "q25"]}, by=["region"])
    assert summary.has_native
    frame = summary.to_pandas().sort_values("region").reset_index(drop=True)
    east = frame.loc[frame["region"] == "east"].iloc[0]
    assert float(east["amount_median"]) == pytest.approx(15.0)
    assert float(east["amount_q25"]) == pytest.approx(12.5)


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_native_aggregate_pushdown() -> None:
    session = Session.ingest(_sales()).with_engine("duckdb")
    engine = get_engine("duckdb")
    summary = session.dataset.aggregate(
        {"amount": ["min", "max"], "*": "count"},
        by=["region"],
    )
    assert summary.has_native
    assert getattr(engine, "_last_pushdown", None) == "aggregate"
    frame = summary.to_pandas().sort_values("region").reset_index(drop=True)
    east = frame.loc[frame["region"] == "east"].iloc[0]
    assert float(east["amount_min"]) == pytest.approx(10.0)
    assert float(east["amount_max"]) == pytest.approx(20.0)
    assert int(east["count"]) == 2
    session.dataset.close_native()


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_duckdb_native_median_quantile() -> None:
    session = Session.ingest(_sales()).with_engine("duckdb")
    engine = get_engine("duckdb")
    summary = session.dataset.aggregate({"amount": ["median", "q75"]}, by=["region"])
    assert summary.has_native
    assert getattr(engine, "_last_pushdown", None) == "aggregate"
    frame = summary.to_pandas().sort_values("region").reset_index(drop=True)
    west = frame.loc[frame["region"] == "west"].iloc[0]
    assert float(west["amount_median"]) == pytest.approx(10.0)
    assert float(west["amount_q75"]) == pytest.approx(12.5)
    session.dataset.close_native()
