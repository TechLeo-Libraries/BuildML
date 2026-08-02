"""Tier C: sklearn AIPW-style twin for causal-treatment-effect."""

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _aipw_ate(x, t, y):
    """Doubly robust ATE with sklearn nuisance models (train-only caller)."""
    prop = LogisticRegression(max_iter=1000)
    prop.fit(x, t)
    e = np.clip(prop.predict_proba(x)[:, 1], 0.05, 0.95)
    mu1 = GradientBoostingRegressor(random_state=0)
    mu0 = GradientBoostingRegressor(random_state=0)
    mu1.fit(x[t == 1], y[t == 1])
    mu0.fit(x[t == 0], y[t == 0])
    m1 = mu1.predict(x)
    m0 = mu0.predict(x)
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return float(np.mean(psi)), float(np.std(psi) / np.sqrt(len(psi)))


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
        Session.ingest(frame.copy())
        .set_roles({"x1": "feature", "x2": "feature", "t": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, ["x1", "x2"]])
    t_train = frame.loc[train_idx, "t"].to_numpy()
    y_train = frame.loc[train_idx, "y"].to_numpy()
    ate, se = _aipw_ate(x_train, t_train, y_train)

    # Placebo refute on train (shuffle treatment).
    t_placebo = rng.permutation(t_train)
    ate_p, _ = _aipw_ate(x_train, t_placebo, y_train)

    # Holdout evaluation disclosure: estimate on validation covariates with train nuisances.
    x_val = scaler.transform(frame.loc[val_idx, ["x1", "x2"]])
    t_val = frame.loc[val_idx, "t"].to_numpy()
    y_val = frame.loc[val_idx, "y"].to_numpy()
    prop = LogisticRegression(max_iter=1000).fit(x_train, t_train)
    e = np.clip(prop.predict_proba(x_val)[:, 1], 0.05, 0.95)
    mu1 = GradientBoostingRegressor(random_state=0).fit(x_train[t_train == 1], y_train[t_train == 1])
    mu0 = GradientBoostingRegressor(random_state=0).fit(x_train[t_train == 0], y_train[t_train == 0])
    m1, m0 = mu1.predict(x_val), mu0.predict(x_val)
    psi = m1 - m0 + t_val * (y_val - m1) / e - (1 - t_val) * (y_val - m0) / (1 - e)
    ate_val = float(np.mean(psi))

    industry_metrics = metrics_round(
        {
            "ate": ate,
            "ate_se": se,
            "ate_validation": ate_val,
            "placebo_ate": ate_p,
            "abs_error_vs_true": abs(ate - 1.8),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_ate = float(bml_raw.get("fit", {}).get("ate", float("nan")))
    bml_metrics = metrics_round(
        {
            "ate": bml_ate,
            "ate_validation": float(bml_raw.get("eval", {}).get("ate", float("nan"))),
            "placebo_ate": float(bml_raw.get("refute", {}).get("refute_ate", float("nan"))),
            "abs_error_vs_true": abs(bml_ate - 1.8) if bml_ate == bml_ate else None,
        }
    )

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_causal(aipw)",
            "test_metrics": bml_metrics,
            "true_ate_approx": 1.8,
        },
        industry={
            "backend": "sklearn LogisticRegression+GBM AIPW",
            "test_metrics": industry_metrics,
            "true_ate_approx": 1.8,
            "leakage_controls": [
                "Nuisance models fit on train only",
                "Validation used for holdout ATE disclosure only",
                "Assumptions declared (unconfoundedness/positivity) — not proven",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(plan.test_indices),
        },
        delta_keys=("ate", "ate_validation", "placebo_ate", "abs_error_vs_true"),
    )
    print("causal-treatment-effect Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
