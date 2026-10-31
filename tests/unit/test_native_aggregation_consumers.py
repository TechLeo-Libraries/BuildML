"""Native summaries retain their inputs and count observed nonmissing data."""

import copy
import math
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

BACKINGS = [("pandas", "memory"), ("polars", "memory"), ("polars", "lazy"),
            ("duckdb", "memory"), ("duckdb", "lazy")]
GROUP = "region key"
AMOUNT = 'amount " net'
FUNCS = ["sum", "mean", "min", "max", "count", "n_unique", "std", "median", "q0", "q25", "q75", "q100"]
REQUEST = {AMOUNT: FUNCS, "missing metric": ["sum", "count", "n_unique"], "*": "count"}


def _frame(kind):
    raw = ["010", "001", "010", None, "001", "010", None, "001", "010", "001", None, "010"]
    groups = ({"integer": [None if v is None else {"010": 10, "001": 2}[v] for v in raw],
               "boolean": [None if v is None else v == "010" for v in raw]}.get(kind, raw))
    if kind == "categorical":
        groups = pd.Categorical(raw, categories=["001", "010", "unused"])
    frame = pd.DataFrame({GROUP: groups, "tier": [True, False, True, False]*3,
                          AMOUNT: pd.array([1, None, 1, None, 5, 3, None, None, 7, 9, None, 4], dtype="Float64"),
                          "missing metric": pd.array([None]*12, dtype="Float64"),
                          "row_id": range(12), "target": ["a", "b"]*6})
    frame.index = pd.Index(np.arange(12) % 5, name="duplicate_rows")
    return frame


@contextmanager
def _source(backing, kind, tmp_path):
    engine, mode = backing
    if engine != "pandas":
        pytest.importorskip(engine)
    frame = _frame(kind)
    if mode == "lazy":
        path = tmp_path / "source.parquet"
        frame.to_parquet(path, index=False)
        owner = Session.ingest(path, engine=engine, mode="lazy")
        assert owner.dataset.pandas_stale and owner.dataset.frame.empty
    else:
        owner = Session.ingest(frame).with_engine(engine)
    owner.set_roles({GROUP: "group", "tier": "group", "row_id": "id", "target": "target"})
    try:
        yield owner, frame
    finally:
        owner.close_native()


def _scalar(values, func):
    values = [float(v) for v in values if not pd.isna(v)]
    if func == "count":
        return len(values)
    if func == "n_unique":
        return len(set(values))
    if not values:
        return None
    if func == "sum":
        return math.fsum(values)
    if func == "mean":
        return math.fsum(values)/len(values)
    if func == "min":
        return min(values)
    if func == "max":
        return max(values)
    if func == "std":
        if len(values) < 2:
            return None
        mean = math.fsum(values)/len(values)
        return math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
    percentile = .5 if func == "median" else int(func[1:])/100
    ordered = sorted(values)
    position = (len(ordered)-1)*percentile
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right]-ordered[left])*(position-left)


def _key(value):
    return None if pd.isna(value) else value


def _expected(frame, by, request=REQUEST):
    columns = list(by or []) + ["count" if c == "*" else f"{c}_{f}"
                                 for c, funcs in request.items() for f in ([funcs] if isinstance(funcs, str) else funcs)]
    groups = {}
    for row in frame.to_dict("records"):
        key = tuple(_key(row[c]) for c in (by or []))
        groups.setdefault(key, []).append(row)
    if not by and not groups:
        groups[()] = []
    result = []
    for key, rows in groups.items():
        row = dict(zip(by or [], key))
        for column, funcs in request.items():
            for func in ([funcs] if isinstance(funcs, str) else funcs):
                name = "count" if column == "*" else f"{column}_{func}"
                row[name] = len(rows) if column == "*" else _scalar([r[column] for r in rows], func)
        result.append(row)
    return pd.DataFrame(result, columns=columns)


def _compare(actual, expected, by):
    assert list(actual.columns) == list(expected.columns)
    keys = list(by or [])
    def indexed(frame):
        return {tuple(_key(row[c]) for c in keys): row for row in frame.to_dict("records")}
    observed, reference = indexed(actual), indexed(expected)
    assert len(actual) == len(observed) == len(expected)
    assert set(observed) == set(reference)
    for key, row in reference.items():
        for column, value in row.items():
            got = observed[key][column]
            if pd.isna(value):
                assert pd.isna(got), (key, column, got)
            elif column in keys:
                assert got == value
            else:
                assert float(got) == pytest.approx(float(value), abs=1e-12)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("kind", ["string", "integer", "boolean", "categorical"])
@pytest.mark.parametrize("by", [None, [GROUP], [GROUP, "tier"]])
@pytest.mark.parametrize("population", ["all", "filtered", "empty"])
def test_scalar_summaries_nested_consumers_and_parquet_keep_their_own_population(backing, kind, by, population, tmp_path):
    with _source(backing, kind, tmp_path) as (owner, original):
        history, roles = copy.deepcopy(owner.history), dict(owner.dataset.roles)
        source, selected = owner.dataset, original
        if population != "all":
            mask = [population != "empty" and i % 3 != 1 for i in range(12)]
            source = source.filter_rows(mask)
            selected = selected.loc[mask]
        expected = _expected(selected, by)
        summary = source.aggregate(REQUEST, by=by)
        assert summary.roles == {}
        if backing[0] != "pandas":
            assert summary.has_native
        # A fresh projection reads the actual lazy plan even after a pandas cache exists.
        _compare(summary.project(summary.columns).to_pandas(), expected, by)
        reduction = {f"{AMOUNT}_sum": ["sum", "count", "n_unique"], "*": "count"}
        nested = summary.aggregate(reduction)
        nested_expected = _expected(expected, None, reduction)
        sibling = source.project(list(source.columns)[::-1]).aggregate({AMOUNT: "mean", "*": "count"}, by=by)
        _compare(sibling.to_pandas(), _expected(selected, by, {AMOUNT: "mean", "*": "count"}), by)
        _compare(nested.project(nested.columns).to_pandas(), nested_expected, None)
        _compare(summary.project(summary.columns).to_pandas(), expected, by)
        materialized = source.aggregate(REQUEST, by=by, materialize=True)
        _compare(materialized.to_pandas(), expected, by)
        path = Session(dataset=summary).to_parquet(tmp_path / "summary.parquet")
        _compare(pd.read_parquet(path), expected, by)
        assert owner.history == history and owner.dataset.roles == roles
        restored = owner.dataset.project(owner.dataset.columns).to_pandas()
        assert list(restored.columns) == list(original.columns) and len(restored) == len(original)
        for actual_row, expected_row in zip(restored.to_dict("records"), original.to_dict("records")):
            for column, value in expected_row.items():
                if pd.isna(value):
                    assert pd.isna(actual_row[column])
                else:
                    assert actual_row[column] == value
        if source is not owner.dataset:
            source.close_native()
        for dataset in [summary, nested, sibling, materialized]:
            dataset.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("aggregations,by", [({}, None), ({AMOUNT: []}, None), ({"*": "mean"}, None),
    ({"absent": "sum"}, None), ({AMOUNT: "count"}, ["absent"]),
    ({AMOUNT: "quantile_0.333"}, None), ({AMOUNT: "quantile_1.2"}, None), ({AMOUNT: "quantile_bad"}, None)])
def test_invalid_aggregation_requests_leave_the_source_and_prior_summary_usable(backing, aggregations, by, tmp_path):
    with _source(backing, "string", tmp_path) as (owner, frame):
        history, native = copy.deepcopy(owner.history), owner.dataset.native
        summary = owner.dataset.aggregate({AMOUNT: "count", "*": "count"}, by=[GROUP])
        expected = _expected(frame, [GROUP], {AMOUNT: "count", "*": "count"})
        with pytest.raises(ValidationError):
            owner.dataset.aggregate(aggregations, by=by)
        _compare(summary.project(summary.columns).to_pandas(), expected, [GROUP])
        assert owner.history == history and owner.dataset.native is native
        summary.close_native()
