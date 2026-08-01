"""Score-time schema contract persistence and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import (
    SCHEMA_CONTRACT_FILENAME,
    load_pipeline_bundle,
    predict_from_pipeline,
    save_pipeline_bundle,
    validate_score_frame,
)
from buildml.pipeline.contract import SchemaContract


def _tiny_cls() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame(
        {
            "age": rng.normal(40, 5, n),
            "income": rng.normal(50, 10, n),
            "y": ([0, 1] * (n // 2)),
        }
    )


def test_save_pipeline_writes_schema_contract(tmp_path: Path) -> None:
    session = (
        Session.ingest(_tiny_cls())
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe, evaluate_partition=None)
    assert (pipe / SCHEMA_CONTRACT_FILENAME).exists()
    bundle = load_pipeline_bundle(pipe)
    assert bundle.schema_contract is not None
    assert "age" in bundle.schema_contract.columns
    assert "y" not in bundle.schema_contract.columns  # target not required at score time
    assert bundle.schema_contract.feature_columns


def test_predict_rejects_missing_and_wrong_type(tmp_path: Path) -> None:
    session = (
        Session.ingest(_tiny_cls())
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=1)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe, evaluate_partition=None)

    with pytest.raises(ValidationError, match="missing columns"):
        predict_from_pipeline(pipe, pd.DataFrame({"age": [1.0, 2.0]}), apply_plans=False)

    with pytest.raises(ValidationError, match="wrong-type"):
        predict_from_pipeline(
            pipe,
            pd.DataFrame({"age": ["a", "b"], "income": ["c", "d"]}),
            apply_plans=False,
        )


def test_predict_allows_extra_columns_with_warning(tmp_path: Path) -> None:
    session = (
        Session.ingest(_tiny_cls())
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=2)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    pipe = tmp_path / "pipe"
    session.save_pipeline(pipe, evaluate_partition=None)
    holdout = _tiny_cls().assign(extra=1.0)
    scored = predict_from_pipeline(pipe, holdout[["age", "income", "extra"]], apply_plans=False)
    assert scored.n_rows == len(holdout)
    assert any("extra columns" in w for w in scored.warnings)
    assert scored.contract_validation is not None
    assert scored.contract_validation.ok


def test_legacy_bundle_without_contract_still_scores(tmp_path: Path) -> None:
    session = (
        Session.ingest(_tiny_cls())
        .set_roles({"age": "feature", "income": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=3)
        .fit(LogisticRegression(max_iter=300), task="classification")
    )
    pipe = tmp_path / "legacy"
    # Save then strip contract to simulate an older bundle.
    save_pipeline_bundle(
        pipe,
        fit_result=session.fit_result,  # type: ignore[arg-type]
        dataset_schema=session.dataset.schema.to_dict(),
        roles={k: v.value for k, v in session.dataset.roles.items()},
        input_columns=session.dataset.columns,
    )
    (pipe / SCHEMA_CONTRACT_FILENAME).unlink()
    meta = json.loads((pipe / "meta.json").read_text(encoding="utf-8"))
    meta["has_schema_contract"] = False
    (pipe / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    bundle = load_pipeline_bundle(pipe)
    assert bundle.schema_contract is None
    scored = predict_from_pipeline(
        pipe,
        _tiny_cls()[["age", "income"]],
        apply_plans=False,
    )
    assert scored.n_rows > 0
    assert scored.contract_validation is not None
    assert scored.contract_validation.contract_present is False
    assert any("no schema_contract" in w for w in scored.warnings)


def test_validate_score_frame_reports_extra_and_wrong_type() -> None:
    contract = SchemaContract(
        columns=("age", "income"),
        dtypes={"age": "float64", "income": "float64"},
        dtype_families={"age": "numeric", "income": "numeric"},
        feature_columns=("age", "income"),
    )
    result = validate_score_frame(
        pd.DataFrame({"age": [1.0], "income": [2.0], "z": [3.0]}),
        contract,
        stage="input",
    )
    assert result.ok
    assert result.extra_columns == ["z"]

    bad = validate_score_frame(
        pd.DataFrame({"age": ["x"], "income": ["y"]}),
        contract,
        stage="input",
    )
    assert not bad.ok
    assert bad.wrong_type_columns
