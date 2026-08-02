"""ATE estimation benchmark on synthetic data with known ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.causal.catalog import causal_capability_matrix
from buildml.causal.extras import dowhy_available, econml_available
from buildml.causal.types import CausalAssumptions


def _synthetic_ate_frame(
    n: int = 500,
    *,
    true_ate: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 3))
    logit = 0.7 * w[:, 0] - 0.4 * w[:, 1] + 0.2 * w[:, 2]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = (
        true_ate * t
        + 0.5 * w[:, 0]
        - 0.3 * w[:, 1]
        + 0.1 * w[:, 2]
        + rng.normal(scale=0.4, size=n)
    )
    return pd.DataFrame(
        {"x1": w[:, 0], "x2": w[:, 1], "x3": w[:, 2], "t": t, "y": y}
    )


def _run_backend(
    backend: str,
    method: str,
    *,
    true_ate: float,
    bootstrap_samples: int = 40,
) -> dict[str, object]:
    frame = _synthetic_ate_frame(true_ate=true_ate)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "x1": "feature",
                "x2": "feature",
                "x3": "feature",
                "t": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    session.declare_causal_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2", "x3"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.fit_causal(
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        bootstrap_samples=0 if method == "causal_forest" else bootstrap_samples,
        random_state=0,
    )
    ev = session.evaluate_causal(partition="validation", bootstrap_samples=10)
    refute = session.refute_causal(kind="placebo_treatment", random_state=0)
    return {
        "backend": backend,
        "method": method,
        "true_ate": true_ate,
        "estimated_ate": fit.ate,
        "ate_error": abs(float(fit.ate) - true_ate),
        "ate_ci_low": fit.ate_ci_low,
        "ate_ci_high": fit.ate_ci_high,
        "validation_ate": ev.ate,
        "refute_ate": refute.refute_ate,
        "cate_std": getattr(fit, "cate_std", None),
        "n_train": fit.n_train_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML causal ATE estimation benchmark (synthetic ground truth)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/causal/results/ate_estimation.json"),
    )
    parser.add_argument("--true-ate", type=float, default=1.5)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_backend("native", "aipw", true_ate=args.true_ate))
    runs.append(_run_backend("native", "t_learner", true_ate=args.true_ate))
    if dowhy_available():
        runs.append(
            _run_backend("dowhy", "backdoor_linear", true_ate=args.true_ate)
        )
        runs.append(
            _run_backend(
                "dowhy", "backdoor_propensity_weighting", true_ate=args.true_ate
            )
        )
    if econml_available():
        runs.append(_run_backend("econml", "dml", true_ate=args.true_ate))
        runs.append(
            _run_backend("econml", "causal_forest", true_ate=args.true_ate)
        )

    payload = {
        "true_ate": args.true_ate,
        "capability_matrix": causal_capability_matrix(),
        "runs": runs,
        "summary": {
            "n_runs": len(runs),
            "best_native_error": min(
                r["ate_error"] for r in runs if r["backend"] == "native"
            ),
            "mean_error": float(np.mean([float(r["ate_error"]) for r in runs])),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
