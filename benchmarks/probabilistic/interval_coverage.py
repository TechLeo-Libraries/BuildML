"""Interval coverage benchmark across native / MAPIE / NGBoost backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.probabilistic.catalog import probabilistic_capability_matrix
from buildml.probabilistic.extras import mapie_available, ngboost_available


def _reference_frame(n: int = 200, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 1.1 * x[:, 0] - 0.7 * x[:, 1] + 0.3 * x[:, 2] + rng.normal(scale=0.35, size=n)
    return pd.DataFrame(
        {"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y},
    )


def _run_backend(
    backend: str,
    estimator: str,
    *,
    task: str = "regression",
) -> dict[str, object]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({"x0": "feature", "x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_probabilistic(
        backend=backend,
        estimator=estimator,  # type: ignore[arg-type]
        task=task if backend == "mapie" else None,
        conformal=True,
        alpha=0.1,
        n_estimators=50,
        learning_rate=0.08,
    )
    ev_val = session.evaluate_probabilistic(partition="validation")
    ev_test = session.evaluate_probabilistic(partition="test")
    interval = session.predict_interval(partition="test")
    return {
        "backend": backend,
        "estimator": estimator,
        "task": task,
        "n_fit_rows": fit.n_fit_rows,
        "n_conformal_calib_rows": fit.n_conformal_calib_rows,
        "validation_interval_coverage": ev_val.metrics.get("interval_coverage"),
        "validation_nll": ev_val.metrics.get("nll"),
        "validation_crps": ev_val.metrics.get("crps"),
        "test_interval_coverage": ev_test.metrics.get("interval_coverage"),
        "test_mean_interval_width": ev_test.metrics.get("mean_interval_width"),
        "interval_method": interval.method,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML probabilistic interval coverage benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/probabilistic/results/interval_coverage.json"),
    )
    args = parser.parse_args(argv)

    matrix = probabilistic_capability_matrix()
    runs: list[dict[str, object]] = []
    runs.append(_run_backend("native", "bayesian_ridge"))
    runs.append(_run_backend("native", "gaussian_process_regressor"))

    if mapie_available():
        for method in ("split", "cv_plus"):
            runs.append(_run_backend("mapie", method, task="regression"))
    else:
        runs.append(
            {
                "backend": "mapie",
                "skipped": True,
                "reason": "mapie not installed",
            }
        )

    if ngboost_available():
        runs.append(_run_backend("ngboost", "ngboost_regressor"))
    else:
        runs.append(
            {
                "backend": "ngboost",
                "skipped": True,
                "reason": "ngboost not installed",
            }
        )

    payload = {
        "capability_matrix": matrix,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
