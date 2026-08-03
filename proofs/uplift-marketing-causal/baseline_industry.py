"""Tier C: sklearn AIPW-style twin for uplift-marketing-causal."""

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

TRUE_ATE = 1.2
CONFOUNDERS = ["recency_z", "freq_z", "monetary_z"]


def _aipw_ate(x, t, y):
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
    ctx = new_proof_context("uplift-marketing-causal", seed=33)
    rng = np.random.default_rng(ctx.seed)
    n = 500
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
        Session.ingest(frame.copy())
        .set_roles({
            **{c: "feature" for c in CONFOUNDERS},
            "promo": "feature",
            "spend": "target",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, CONFOUNDERS])
    t_train = frame.loc[train_idx, "promo"].to_numpy()
    y_train = frame.loc[train_idx, "spend"].to_numpy()
    ate, se = _aipw_ate(x_train, t_train, y_train)

    t_placebo = rng.permutation(t_train)
    ate_p, _ = _aipw_ate(x_train, t_placebo, y_train)

    x_val = scaler.transform(frame.loc[val_idx, CONFOUNDERS])
    t_val = frame.loc[val_idx, "promo"].to_numpy()
    y_val = frame.loc[val_idx, "spend"].to_numpy()
    prop = LogisticRegression(max_iter=1000).fit(x_train, t_train)
    e = np.clip(prop.predict_proba(x_val)[:, 1], 0.05, 0.95)
    mu1 = GradientBoostingRegressor(random_state=0).fit(
        x_train[t_train == 1], y_train[t_train == 1]
    )
    mu0 = GradientBoostingRegressor(random_state=0).fit(
        x_train[t_train == 0], y_train[t_train == 0]
    )
    m1, m0 = mu1.predict(x_val), mu0.predict(x_val)
    psi = m1 - m0 + t_val * (y_val - m1) / e - (1 - t_val) * (y_val - m0) / (1 - e)
    ate_val = float(np.mean(psi))

    industry_metrics = metrics_round(
        {
            "ate": ate,
            "ate_se": se,
            "ate_validation": ate_val,
            "placebo_ate": ate_p,
            "abs_error_vs_true": abs(ate - TRUE_ATE),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_ate = float(bml_raw.get("fit", {}).get("ate", float("nan")))
    bml_metrics = metrics_round(
        {
            "ate": bml_ate,
            "ate_validation": float(bml_raw.get("eval", {}).get("ate", float("nan"))),
            "placebo_ate": float(bml_raw.get("refute", {}).get("refute_ate", float("nan"))),
            "abs_error_vs_true": abs(bml_ate - TRUE_ATE) if bml_ate == bml_ate else None,
        }
    )

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_causal(aipw)",
            "test_metrics": bml_metrics,
            "true_ate_approx": TRUE_ATE,
        },
        industry={
            "backend": "sklearn LogisticRegression+GBM AIPW",
            "test_metrics": industry_metrics,
            "true_ate_approx": TRUE_ATE,
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
    print("uplift-marketing-causal Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
