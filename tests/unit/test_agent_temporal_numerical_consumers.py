"""Agent temporal preprocessing is checked against chronological and numeric oracles."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def run(session, tool, **arguments):
    registry = build_default_registry()
    result = execute_tool(
        session,
        propose_tool_execution(tool, arguments, registry),
        confirmed=True,
        registry=registry,
    )
    assert result.executed, result.error
    return result


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
@pytest.mark.parametrize("test_size", [0.2, 10])
@pytest.mark.parametrize("validation_size", [None, 0.25, 8])
@pytest.mark.parametrize("explicit_time", [False, True])
def test_agent_temporal_pipeline_matches_chronology_and_train_only_matrix(
    order, test_size, validation_size, explicit_time
):
    rng = np.random.default_rng(756)
    n = 80
    x = np.linspace(-4, 9, n) + rng.normal(0, 0.5, n)
    z = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n),
            "x": x,
            "z": z,
            "y": 2 * x - 0.8 * z + rng.normal(0, 0.3, n),
        }
    )
    frame.loc[::6, "x"] = np.nan
    if order == "reverse":
        frame = frame.iloc[::-1].copy()
    elif order == "shuffled":
        frame = frame.iloc[rng.permutation(n)].copy()
    frame.index = [5, 5, 2, 9] * 20
    original = frame.copy(deep=True)
    s = Session.ingest(frame)
    run(s, "set_roles", mapping={"time": "time", "x": "feature", "z": "feature", "y": "target"})
    arguments = dict(test_size=test_size, validation_size=validation_size)
    if explicit_time:
        arguments["time_column"] = "time"
    split = run(s, "time_split", **arguments)
    assert split.result["time_split_created"] is True
    chronological = np.argsort(frame.time.to_numpy(), kind="stable")
    nt = round(n * test_size) if isinstance(test_size, float) else test_size
    nv = (
        0
        if validation_size is None
        else (
            round((n - nt) * validation_size)
            if isinstance(validation_size, float)
            else validation_size
        )
    )
    expected_test = chronological[-nt:]
    expected_validation = chronological[n - nt - nv : n - nt] if nv else []
    expected_train = chronological[: n - nt - nv]
    assert list(s.split_plan.train_indices) == list(expected_train)
    assert list(s.split_plan.test_indices) == list(expected_test)
    assert list(s.split_plan.validation_indices or ()) == list(expected_validation)
    train_raw = frame.iloc[expected_train][["x", "z"]]
    medians = np.nanmedian(train_raw.to_numpy(), axis=0)
    filled = np.where(np.isnan(frame[["x", "z"]].to_numpy()), medians, frame[["x", "z"]].to_numpy())
    mean = filled[expected_train].mean(axis=0)
    std = filled[expected_train].std(axis=0)
    design = (filled - mean) / std
    run(s, "impute", columns=["x", "z"], strategy="median")
    run(s, "scale", columns=["x", "z"], method="standard")
    np.testing.assert_allclose(s.to_pandas()[["x", "z"]], design, atol=1e-12)
    run(s, "fit", estimator="Ridge", task="regression", hyperparameters={"alpha": 0.7})
    xt = design[expected_train]
    yt = frame.y.to_numpy()[expected_train]
    centered = xt - xt.mean(axis=0)
    coef = np.linalg.solve(centered.T @ centered + 0.7 * np.eye(2), centered.T @ (yt - yt.mean()))
    predicted = (design[expected_test] - xt.mean(axis=0)) @ coef + yt.mean()
    actual = s.fit_result.estimator.predict(s.partition("test")[["x", "z"]])
    np.testing.assert_allclose(actual, predicted, atol=1e-12)
    evaluation = run(s, "evaluate", partition="test").result
    assert evaluation.metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((frame.y.to_numpy()[expected_test] - predicted) ** 2))
    )
    assert s.to_pandas().index.equals(original.index)
    np.testing.assert_array_equal(s.to_pandas().y, original.y)
    pd.testing.assert_frame_equal(frame, original)
