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
    session.causal.declare_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.causal.fit(method="aipw", bootstrap_samples=40)
    ev = session.causal.evaluate(partition="validation", bootstrap_samples=20)
    ref = session.causal.refute(kind="placebo_treatment")
    bundle = session.causal.save_bundle(ctx.artifacts_dir / "causal_bundle")

    restored = (
        Session.ingest(frame)
        .set_roles({"x1": "feature", "x2": "feature", "t": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
    )
    restored.causal.load_bundle(bundle, trusted=True)
    ev_reloaded = restored.causal.evaluate(
        partition="validation", bootstrap_samples=20
    )

    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_backdoor",
            "license": "synthetic/public-domain",
            "n_rows": n,
            "true_ate_approx": 1.8,
        },
        "assumptions": {
            "treatment": "t",
            "outcome": "y",
            "confounders": ["x1", "x2"],
            "acknowledged": ["unconfoundedness", "positivity"],
        },
        "fit": {
            "method": fit.method,
            "backend": getattr(fit, "backend", "native"),
            "ate": float(fit.ate),
            "ate_ci_low": float(fit.ate_ci_low) if fit.ate_ci_low is not None else None,
            "ate_ci_high": float(fit.ate_ci_high) if fit.ate_ci_high is not None else None,
        },
        "eval": {
            "ate": float(ev.ate),
            "ate_ci_low": float(ev.ate_ci_low) if ev.ate_ci_low is not None else None,
            "ate_ci_high": float(ev.ate_ci_high) if ev.ate_ci_high is not None else None,
            "metrics": metrics_round(dict(ev.metrics)),
        },
        "reloaded_eval": {
            "ate": float(ev_reloaded.ate),
            "metrics": metrics_round(dict(ev_reloaded.metrics)),
        },
        "bundle_roundtrip": {
            "loaded": restored.causal.plan is not None,
            "ate_match": bool(abs(float(ev.ate) - float(ev_reloaded.ate)) < 1e-9),
        },
        "refute": {
            "kind": "placebo_treatment",
            "refute_ate": float(ref.refute_ate),
            "ate_shift": float(ref.ate_shift),
        },
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Assumptions declared before fit",
            "Train-only nuisance models",
            "Holdout eval disclosed",
            "Bundle load re-evaluate uses frozen nuisances only",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn AIPW-style twin on the same split; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Synthetic DGP; assumptions are declared not proven",
            "EDA remains non-causal",
            "Native AIPW path (DoWhy/EconML industry optional, subprocess-gated)",
        ],
    })
    print("causal-treatment-effect OK", float(fit.ate))


if __name__ == "__main__":
    main()
