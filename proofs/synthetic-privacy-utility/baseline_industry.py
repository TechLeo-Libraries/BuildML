"""Tier C: independent-marginal resampling twin for synthetic-privacy-utility."""

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
    ctx = new_proof_context("synthetic-privacy-utility", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 12, n).clip(18, 90),
            "income": rng.lognormal(10.5, 0.5, n),
            "score": rng.beta(2, 5, n),
            "segment": rng.choice(["A", "B", "C"], size=n),
        }
    )
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "age": "feature",
                "income": "feature",
                "score": "feature",
                "segment": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    tr = list(plan.train_indices)
    te = list(plan.test_indices)
    train = frame.loc[tr]
    test = frame.loc[te]

    # Industry twin: independent column bootstrap from train (no joint copula).
    n_syn = len(tr)
    syn = pd.DataFrame(
        {
            "age": rng.choice(train["age"].to_numpy(), size=n_syn, replace=True),
            "income": rng.choice(train["income"].to_numpy(), size=n_syn, replace=True),
            "score": rng.choice(train["score"].to_numpy(), size=n_syn, replace=True),
            "segment": rng.choice(train["segment"].to_numpy(), size=n_syn, replace=True),
        }
    )

    ks_vals = [
        ks_2samp(syn[c], test[c]).statistic for c in ("age", "income", "score")
    ]
    # Correlation L1 on continuous cols vs test
    cont = ["age", "income", "score"]
    corr_syn = np.corrcoef(syn[cont].to_numpy().T)
    corr_real = np.corrcoef(test[cont].to_numpy().T)
    corr_l1 = float(np.mean(np.abs(corr_syn - corr_real)))
    industry_metrics = metrics_round(
        {
            "mean_ks": float(np.mean(ks_vals)),
            "mean_tv": _tv_distance(syn["segment"].to_numpy(), test["segment"].to_numpy()),
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
                "Same split (seed=0)",
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
    print("synthetic-privacy-utility Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
