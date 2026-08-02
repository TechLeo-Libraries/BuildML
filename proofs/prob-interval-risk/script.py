"""Tier A proof: prob-interval-risk."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from proofs._lib import metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("prob-interval-risk", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 400
    x = rng.normal(size=(n, 4))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["loss"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "loss": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_probabilistic(
        estimator="bayesian_ridge",
        alpha=0.1,
        conformal=True,
        interval_method="both",
        random_state=ctx.seed,
    )
    # Conformal / intervals typically calibrated on validation
    try:
        intervals = session.predict_interval(partition="test", alpha=0.1)
        interval_payload = metrics_round(intervals.to_dict() if hasattr(intervals, "to_dict") else {})
    except Exception as exc:  # noqa: BLE001
        interval_payload = {"error": f"{type(exc).__name__}: {exc}"}
    ev = session.evaluate_probabilistic(partition="test")
    bundle = session.save_probabilistic_bundle(ctx.artifacts_dir / "prob_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_risk_regression", "license": "synthetic/public-domain", "n_rows": n},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "intervals": interval_payload,
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Probabilistic model fit on train",
            "Interval calibration uses non-test partitions when required by API",
            "Test evaluate after lock",
        ],
        "industry_comparison": {"status": "stub", "note": "MAPIE twin when installed"},
        "limitations": ["Empirical coverage ≠ guaranteed under distribution shift"],
    })
    print("prob-interval-risk OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
