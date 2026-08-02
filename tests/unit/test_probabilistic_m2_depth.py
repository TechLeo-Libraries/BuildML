"""M2 depth coverage for probabilistic low-level APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.probabilistic.evaluate import evaluate_probabilistic
from buildml.probabilistic.fit import fit_probabilistic
from buildml.probabilistic.predict import predict_interval, predict_probabilistic


def _reg_frame(n: int = 180, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = 1.2 * x[:, 0] - 0.6 * x[:, 1] + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})


def _cls_frame(n: int = 180, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "label": y})


def _reg_session() -> Session:
    return (
        Session.ingest(_reg_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=1)
        .scale(method="standard")
    )


def _cls_session() -> Session:
    return (
        Session.ingest(_cls_frame())
        .set_roles({"a": "feature", "b": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=1, stratify=True)
        .scale(method="standard")
    )


def test_bayesian_ridge_conformal_intervals() -> None:
    session = _reg_session()
    plan, fit = fit_probabilistic(
        session.dataset,
        session.split_plan,
        estimator="bayesian_ridge",
        conformal=True,
        alpha=0.1,
        reduce_plan=session._reduce_plan,
    )
    assert fit.conformal is True
    assert fit.conformal_quantile is not None
    assert plan.supports_return_std
    pred = predict_probabilistic(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert pred.std is not None
    assert len(pred.predictions) == len(pred.std)
    interval = predict_interval(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert interval.lower is not None and interval.upper is not None
    assert all(u >= lo for lo, u in zip(interval.lower, interval.upper, strict=True))
    ev = evaluate_probabilistic(
        session.dataset, plan, session.split_plan, partition="validation"
    )
    assert "nll" in ev.metrics
    assert "interval_coverage" in ev.metrics
    assert 0.0 <= ev.metrics["interval_coverage"] <= 1.0


def test_gaussian_process_regressor_path() -> None:
    session = _reg_session()
    plan, fit = fit_probabilistic(
        session.dataset,
        session.split_plan,
        estimator="gaussian_process_regressor",
        conformal=True,
        n_restarts_optimizer=0,
        reduce_plan=session._reduce_plan,
    )
    assert fit.estimator_name == "gaussian_process_regressor"
    interval = predict_interval(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
        method="both",
    )
    assert interval.std is not None
    assert interval.method in {"both", "split_conformal", "posterior_std"}


def test_gaussian_nb_prediction_sets() -> None:
    session = _cls_session()
    plan, fit = fit_probabilistic(
        session.dataset,
        session.split_plan,
        estimator="gaussian_nb",
        conformal=True,
        alpha=0.2,
        reduce_plan=session._reduce_plan,
    )
    assert fit.task == "classification"
    assert plan.conformal_quantile_ is not None
    sets = predict_interval(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert sets.prediction_sets is not None
    assert all(len(s) >= 1 for s in sets.prediction_sets)
    ev = evaluate_probabilistic(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert "nll" in ev.metrics
    assert "set_coverage" in ev.metrics


def test_conformal_refuses_holdout_by_construction() -> None:
    session = _reg_session()
    plan, _ = fit_probabilistic(
        session.dataset,
        session.split_plan,
        estimator="bayesian_ridge",
        conformal=True,
        reduce_plan=session._reduce_plan,
    )
    train_set = set(session.split_plan.train_indices)
    assert set(plan.conformal_fit_indices_).issubset(train_set)
    assert set(plan.conformal_calib_indices_).issubset(train_set)
    assert set(plan.conformal_fit_indices_).isdisjoint(plan.conformal_calib_indices_)
    holdout = set(session.split_plan.test_indices) | set(
        session.split_plan.validation_indices
    )
    assert set(plan.conformal_calib_indices_).isdisjoint(holdout)


def test_unknown_estimator_rejected() -> None:
    session = _reg_session()
    with pytest.raises(ValidationError, match="Unknown probabilistic"):
        fit_probabilistic(
            session.dataset,
            session.split_plan,
            estimator="pymc_nuts",  # type: ignore[arg-type]
            reduce_plan=session._reduce_plan,
        )
