"""Tier A proof: causal-treatment-effect."""

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
    ctx = new_proof_context("causal-treatment-effect", seed=11)
    rng = np.random.default_rng(ctx.seed)
    n = 480
    w = rng.normal(size=(n, 2))
    logit = 0.9 * w[:, 0] - 0.6 * w[:, 1]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = 1.8 * t + 0.6 * w[:, 0] - 0.5 * w[:, 1] + rng.normal(scale=0.45, size=n)
    frame = pd.DataFrame({"x1": w[:, 0], "x2": w[:, 1], "t": t, "y": y})
    session = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "t": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
    )
    session.declare_causal_assumptions(
        treatment="t", outcome="y", confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True, acknowledge_positivity=True,
    )
    fit = session.fit_causal(method="aipw", bootstrap_samples=40)
    ev = session.evaluate_causal(partition="validation", bootstrap_samples=20)
    ref = session.refute_causal(kind="placebo_treatment")
    bundle = session.save_causal_bundle(ctx.artifacts_dir / "causal_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_backdoor", "license": "synthetic/public-domain", "n_rows": n, "true_ate_approx": 1.8},
        "assumptions": {
            "treatment": "t", "outcome": "y", "confounders": ["x1", "x2"],
            "acknowledged": ["unconfoundedness", "positivity"],
        },
        "fit": {
            "method": fit.method,
            "ate": float(fit.ate),
            "ate_ci_low": float(fit.ate_ci_low),
            "ate_ci_high": float(fit.ate_ci_high),
        },
        "eval": {
            "ate": float(ev.ate),
            "metrics": metrics_round(dict(ev.metrics)),
        },
        "refute": {
            "kind": "placebo_treatment",
            "refute_ate": float(ref.refute_ate),
            "ate_shift": float(ref.ate_shift),
        },
        "bundle_path": str(bundle),
        "leakage_controls": ["Assumptions declared before fit", "Train-only nuisance models", "Holdout eval disclosed"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn AIPW-style twin on the same split; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": ["Synthetic DGP; assumptions are declared not proven", "EDA remains non-causal"],
    })
    print("causal-treatment-effect OK", float(fit.ate))


if __name__ == "__main__":
    main()
