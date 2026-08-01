"""Portable filter_expr helper for Polars / DuckDB."""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.core.types import EngineName
from buildml.data.filter_syntax import portable_filter_expr, quote_identifier, sql_literal
from buildml.ingest.detect import available_engines


def test_portable_filter_expr_renders_scalars() -> None:
    assert portable_filter_expr("a", ">", 2) == '"a" > 2'
    assert portable_filter_expr("score", ">=", 0.5) == '"score" >= 0.5'
    assert portable_filter_expr("flag", "==", True) == '"flag" = TRUE'
    assert portable_filter_expr("name", "!=", "x'y") == "\"name\" <> 'x''y'"
    assert portable_filter_expr('col"x', "<", None) == '"col""x" < NULL'


def test_portable_filter_expr_rejects_bad_op() -> None:
    with pytest.raises(ValidationError, match="op must be"):
        portable_filter_expr("a", "LIKE", 1)


def test_quote_and_literal_helpers() -> None:
    assert quote_identifier("a b") == '"a b"'
    assert sql_literal("hi") == "'hi'"
    with pytest.raises(ValidationError):
        sql_literal([1, 2])


@pytest.mark.skipif(EngineName.POLARS not in available_engines(), reason="polars not installed")
def test_portable_predicate_on_polars() -> None:
    session = Session.ingest(pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}), engine="polars")
    pred = portable_filter_expr("a", ">", 1)
    out = session.dataset.filter_expr(pred).to_pandas()
    assert list(out["a"]) == [2, 3]


@pytest.mark.skipif(EngineName.DUCKDB not in available_engines(), reason="duckdb not installed")
def test_portable_predicate_on_duckdb() -> None:
    with Session.ingest(pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}), engine="duckdb") as session:
        pred = portable_filter_expr("a", ">=", 2)
        out = session.dataset.filter_expr(pred).to_pandas()
        assert list(out["a"]) == [2, 3]
