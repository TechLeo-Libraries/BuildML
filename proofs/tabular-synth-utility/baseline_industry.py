"""Tier C: independent-marginal resampling twin for tabular-synth-utility."""

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
from scipy.stats import ks_2samp

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    pa = np.array([(a == c).mean() for c in cats])
    pb = np.array([(b == c).mean() for c in cats])
    return float(0.5 * np.abs(pa - pb).sum())


def main() -> None:
    ctx = new_proof_context("tabular-synth-utility", seed=30)
    rng = np.random.default_rng(ctx.seed)
    n = 520
    frame = pd.DataFrame({
        "unit_price": rng.lognormal(3.2, 0.55, n).clip(1.0, 500.0),
        "units_sold": rng.poisson(12, n).astype(float) + 1.0,
        "margin_pct": rng.beta(3, 4, n),
        "category": rng.choice(["electronics", "apparel", "grocery", "home"], size=n),
    })
    session = (
        Session.ingest(frame.copy())
        .set_roles({
            "unit_price": "feature",
            "units_sold": "feature",
            "margin_pct": "feature",
            "category": "feature",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr = list(plan.train_indices)
    te = list(plan.test_indices)
    train = frame.loc[tr]
    test = frame.loc[te]

    n_syn = len(tr)
    syn = pd.DataFrame({
        "unit_price": rng.choice(train["unit_price"].to_numpy(), size=n_syn, replace=True),
        "units_sold": rng.choice(train["units_sold"].to_numpy(), size=n_syn, replace=True),
        "margin_pct": rng.choice(train["margin_pct"].to_numpy(), size=n_syn, replace=True),
        "category": rng.choice(train["category"].to_numpy(), size=n_syn, replace=True),
    })

    cont = ["unit_price", "units_sold", "margin_pct"]
    ks_vals = [ks_2samp(syn[c], test[c]).statistic for c in cont]
    corr_syn = np.corrcoef(syn[cont].to_numpy().T)
    corr_real = np.corrcoef(test[cont].to_numpy().T)
    corr_l1 = float(np.mean(np.abs(corr_syn - corr_real)))
    industry_metrics = metrics_round(
        {
            "mean_ks": float(np.mean(ks_vals)),
            "mean_tv": _tv_distance(syn["category"].to_numpy(), test["category"].to_numpy()),
            "corr_l1": corr_l1,
            "n_columns_scored": 4.0,
        }
    )

    bml = load_buildml_results(ctx.project_dir)
    bml_eval = dict(bml.get("eval", {}).get("metrics", {}))
    bml_metrics = metrics_round(
        {
            "mean_ks": float(bml_eval.get("mean_ks", float("nan"))),
            "mean_tv": float(bml_eval.get("mean_tv", float("nan"))),
            "corr_l1": float(bml_eval.get("corr_l1", float("nan"))),
            "n_columns_scored": float(bml_eval.get("n_columns_scored", 4.0)),
        }
    )

    write_comparison(
        ctx,
        buildml={"backend": "buildml.synthetic/gaussian_copula", "test_metrics": bml_metrics},
        industry={
            "backend": "independent column bootstrap from train",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Same split (seed=30)",
                "Generator = train-only column bootstrap; fidelity vs holdout test",
            ],
            "disclosures": [
                "NO differential privacy claims — utility metrics ≠ anonymity",
            ],
        },
        split_counts={
            "train": len(tr),
            "validation": len(plan.validation_indices),
            "test": len(te),
        },
        delta_keys=("mean_ks", "mean_tv", "corr_l1"),
        extra={"note": "Lower KS/TV/corr_l1 is better fidelity; deltas = buildml - industry"},
    )
    print("tabular-synth-utility Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
