"""Session-facing slice tests for probabilistic ML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def _session() -> Session:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(160, 2))
    y = 0.9 * x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.25, size=160)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def test_session_fit_interval_eval_bundle(tmp_path: Path) -> None:
    session = _session()
    fit = session.fit_probabilistic(
        estimator="bayesian_ridge",
        conformal=True,
        alpha=0.1,
    )
    assert session.probabilistic_plan is not None
    assert fit.n_conformal_calib_rows > 0

    interval = session.predict_interval(partition="test")
    assert interval.lower is not None
    assert len(interval.lower) == len(interval.upper)

    ev = session.evaluate_probabilistic(partition="validation")
    assert "rmse" in ev.metrics
    assert ev.interval_coverage is not None

    # Classical calibration still requires classical fit — unchanged.
    try:
        session.calibration()
        raised = False
    except Exception as exc:  # noqa: BLE001
        raised = True
        assert "fit" in str(exc).lower() or "fitted" in str(exc).lower()
    assert raised

    out = tmp_path / "prob_bundle"
    session.save_probabilistic_bundle(out)
    assert (out / "meta.json").is_file()
    assert (out / "probabilistic_plan.joblib").is_file()

    other = _session()
    other.load_probabilistic_bundle(out, trusted=True)
    assert other.probabilistic_plan is not None
    assert other.probabilistic_plan.estimator_name == "bayesian_ridge"
    reloaded = other.evaluate_probabilistic(partition="test")
    assert "mae" in reloaded.metrics


def test_walkthrough_exposes_probabilistic_status() -> None:
    session = _session()
    session.fit_probabilistic(estimator="bayesian_ridge", conformal=True)
    report = session.walkthrough()
    payload = report.to_dict()
    assert "probabilistic_status" in payload
    assert payload["probabilistic_status"]["enabled"] is True
