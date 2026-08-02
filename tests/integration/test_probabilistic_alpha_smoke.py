"""Alpha smoke: Session probabilistic BayesianRidge + conformal + bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_probabilistic_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(220, 3))
    y = (
        1.1 * x[:, 0]
        - 0.4 * x[:, 1]
        + 0.3 * x[:, 2]
        + rng.normal(scale=0.4, size=220)
    )
    frame = pd.DataFrame(
        {"f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "y": y}
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_probabilistic(
        estimator="bayesian_ridge",
        conformal=True,
        alpha=0.1,
        interval_method="both",
    )
    assert fit.estimator_name == "bayesian_ridge"
    assert fit.conformal_quantile is not None

    preds = session.predict_probabilistic(partition="test", return_std=True)
    assert preds.std is not None
    assert len(preds.predictions) == preds.n_rows

    intervals = session.predict_interval(partition="test")
    assert intervals.lower is not None and intervals.upper is not None

    ev = session.evaluate_probabilistic(partition="validation")
    assert "nll" in ev.metrics
    assert "interval_coverage" in ev.metrics

    bundle = tmp_path / "probabilistic_bundle"
    session.save_probabilistic_bundle(bundle)
    clone = (
        Session.ingest(frame)
        .set_roles(
            {"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    clone.load_probabilistic_bundle(bundle)
    again = clone.predict_interval(partition="test")
    assert again.n_rows == intervals.n_rows
