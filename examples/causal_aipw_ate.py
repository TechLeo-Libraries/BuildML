"""Causal example: declared assumptions → AIPW ATE → eval → refute → bundle.

Honesty: backdoor ATE under caller-declared CausalAssumptions — not causal
discovery, not DoWhy/EconML, and not causality from EDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(11)
    n = 420
    w = rng.normal(size=(n, 2))
    logit = 0.9 * w[:, 0] - 0.6 * w[:, 1]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = 1.8 * t + 0.6 * w[:, 0] - 0.5 * w[:, 1] + rng.normal(scale=0.45, size=n)
    frame = pd.DataFrame({"x1": w[:, 0], "x2": w[:, 1], "t": t, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles(
            {"x1": "feature", "x2": "feature", "t": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )

    session.causal.declare_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.causal.fit(method="aipw", bootstrap_samples=40)
    print(
        f"method={fit.method} ate={fit.ate:.4f} "
        f"ci=[{fit.ate_ci_low:.4f}, {fit.ate_ci_high:.4f}]"
    )

    ev = session.causal.evaluate(partition="validation", bootstrap_samples=20)
    print(f"eval ate={ev.ate:.4f} metrics={ev.metrics}")

    ref = session.causal.refute(kind="placebo_treatment")
    print(f"placebo refute_ate={ref.refute_ate:.4f} shift={ref.ate_shift:.4f}")

    out = Path("artifacts") / "causal_aipw_bundle"
    session.causal.save_bundle(out)
    print(f"saved bundle -> {out}")


if __name__ == "__main__":
    main()
