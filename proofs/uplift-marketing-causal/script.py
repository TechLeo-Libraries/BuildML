"""Tier A proof: uplift-marketing-causal — marketing treatment ATE via AIPW."""

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
    ctx = new_proof_context("uplift-marketing-causal", seed=33)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    # Marketing uplift DGP: promo treatment with known ATE ≈ 1.2 on spend.
    w = rng.normal(size=(n, 3))
    logit = 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.3 * w[:, 2]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = 1.2 * t + 0.5 * w[:, 0] - 0.4 * w[:, 1] + 0.25 * w[:, 2] + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({
        "recency_z": w[:, 0],
        "freq_z": w[:, 1],
        "monetary_z": w[:, 2],
        "promo": t,
        "spend": y,
    })
    session = (
        Session.ingest(frame)
        .set_roles({
            "recency_z": "feature",
            "freq_z": "feature",
            "monetary_z": "feature",
            "promo": "feature",
            "spend": "target",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
    )
    session.declare_causal_assumptions(
        treatment="promo", outcome="spend",
        confounders=["recency_z", "freq_z", "monetary_z"],
        acknowledge_unconfoundedness=True, acknowledge_positivity=True,
    )
    fit = session.fit_causal(method="aipw", bootstrap_samples=40)
    ev = session.evaluate_causal(partition="validation", bootstrap_samples=20)
    ref = session.refute_causal(kind="placebo_treatment")
    bundle = session.save_causal_bundle(ctx.artifacts_dir / "causal_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_marketing_uplift",
            "license": "synthetic/public-domain",
            "n_rows": n,
            "true_ate_approx": 1.2,
        },
        "assumptions": {
            "treatment": "promo",
            "outcome": "spend",
            "confounders": ["recency_z", "freq_z", "monetary_z"],
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
        "leakage_controls": [
            "Assumptions declared before fit",
            "Train-only nuisance models",
            "Holdout eval disclosed",
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
            "Marketing uplift narrative distinct from causal-treatment-effect",
        ],
    })
    print("uplift-marketing-causal OK", float(fit.ate))


if __name__ == "__main__":
    main()
