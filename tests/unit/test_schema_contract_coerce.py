"""Schema contract coercion and role-aware required columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.types import ColumnRole
from buildml.ingest.detect import schema_from_dataframe
from buildml.pipeline.contract import (
    build_schema_contract,
    coerce_score_frame,
    families_compatible,
    validate_score_frame,
)


def test_families_compatible_bool_numeric_and_string_categorical() -> None:
    assert families_compatible("bool", "numeric")
    assert families_compatible("string", "categorical")
    assert not families_compatible("numeric", "string")


def test_coerce_numeric_from_string() -> None:
    frame = pd.DataFrame({"age": ["21", "35"], "income": ["40.5", "55"]})
    contract = build_schema_contract(
        schema=schema_from_dataframe(
            pd.DataFrame({"age": [21.0, 35.0], "income": [40.5, 55.0]})
        ),
        roles={"age": ColumnRole.FEATURE, "income": ColumnRole.FEATURE},
        feature_columns=("age", "income"),
        target_column=None,
        input_columns=("age", "income"),
    )
    coerced, result = coerce_score_frame(frame, contract, stage="input")
    assert result.ok
    assert "age" in result.coerced_columns
    assert pd.api.types.is_numeric_dtype(coerced["age"])


def test_required_roles_flag_missing_group() -> None:
    contract = build_schema_contract(
        schema=schema_from_dataframe(
            pd.DataFrame({"x": [1.0, 2.0], "g": ["a", "b"], "y": [0, 1]})
        ),
        roles={
            "x": ColumnRole.FEATURE,
            "g": ColumnRole.GROUP,
            "y": ColumnRole.TARGET,
        },
        feature_columns=("x",),
        target_column="y",
        input_columns=("x", "g"),
    )
    assert "group" in contract.required_roles
    result = validate_score_frame(
        pd.DataFrame({"x": [1.0, 2.0]}),
        contract,
        stage="input",
    )
    assert not result.ok
    assert "group" in result.missing_roles


def test_predict_coerces_compatible_string_numerics(tmp_path) -> None:
    rng = np.random.default_rng(4)
    n = 40
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 5, n),
            "income": rng.normal(50, 10, n),
            "y": ([0, 1] * (n // 2)),
        }
    )
    session = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe, evaluate_partition=None)

    from buildml.pipeline import predict_from_pipeline

    holdout = pd.DataFrame({"age": ["41.0", "39.5"], "income": ["51", "48"]})
    scored = predict_from_pipeline(pipe, holdout, apply_plans=False)
    assert scored.n_rows == 2
    assert scored.contract_validation is not None
    assert scored.contract_validation.ok
    assert scored.contract_validation.coerced_columns
