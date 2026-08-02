"""Probabilistic example: BayesianRidge + split conformal → eval → bundle.

Honesty: sklearn BayesianRidge with train-only split conformal — not a
PyMC/Stan MCMC platform and not Bayesian deep nets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(240, 2))
    y = 1.4 * x[:, 0] - 0.8 * x[:, 1] + rng.normal(scale=0.35, size=240)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_probabilistic(
        estimator="bayesian_ridge",
        alpha=0.1,
        conformal=True,
        interval_method="both",
    )
    print(
        f"estimator={fit.estimator_name} n_fit={fit.n_fit_rows} "
        f"n_calib={fit.n_conformal_calib_rows} q={fit.conformal_quantile}"
    )

    preds = session.predict_probabilistic(partition="test", return_std=True)
    print(f"n_pred={len(preds.predictions)} has_std={preds.std is not None}")

    intervals = session.predict_interval(partition="test")
    print(
        f"interval method={intervals.method} "
        f"width0={intervals.upper[0] - intervals.lower[0]:.4f}"
    )

    ev = session.evaluate_probabilistic(partition="validation")
    print(f"metrics={ev.metrics}")

    out = Path("artifacts") / "probabilistic_bayesian_ridge_bundle"
    session.save_probabilistic_bundle(out)
    print(f"saved bundle -> {out}")


if __name__ == "__main__":
    main()
