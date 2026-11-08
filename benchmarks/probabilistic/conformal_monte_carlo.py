"""Independent-trial conformal calibration and shift diagnostics; no universal guarantee."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def run_trials(trials: int, alpha: float = .1) -> dict:
    """Report independent-trial marginal coverage, interval width and shifted-noise coverage."""
    rows = []
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(600, 2))
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=600)
        frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
        s = Session.ingest(frame).set_roles({"a": "feature", "b": "feature", "y": "target"}).split(
            test_size=.3, random_state=seed)
        s.probabilistic.fit(alpha=alpha, random_state=seed)
        interval = s.probabilistic.predict_interval(partition="test")
        actual = s.partition("test").y.to_numpy()
        lower, upper = np.asarray(interval.lower), np.asarray(interval.upper)
        # Deliberate violation: deployment noise is wider than calibration noise.
        shifted = actual + rng.normal(scale=3, size=len(actual))
        rows.append({"seed": seed, "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                     "shifted_coverage": float(np.mean((shifted >= lower) & (shifted <= upper))),
                     "width": float(np.mean(upper - lower))})
    coverage = np.asarray([r["coverage"] for r in rows])
    return {"protocol": "New independent IID dataset per trial; unprocessed features; fixed alpha; untouched test",
            "trials": trials, "alpha": alpha, "nominal_marginal_coverage": 1-alpha,
            "mean_coverage": float(coverage.mean()),
            "trial_mean_standard_error": float(coverage.std(ddof=1)/np.sqrt(trials)) if trials > 1 else None,
            "mean_shifted_coverage": float(np.mean([r["shifted_coverage"] for r in rows])),
            "mean_width": float(np.mean([r["width"] for r in rows])), "per_trial": rows,
            "limitations": "Simulation regression evidence; shifted rows violate calibration assumptions. No subgroup or arbitrary-population guarantee."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmarks/conformal-monte-carlo.json"))
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    payload = run_trials(args.trials)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k != "per_trial"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
