"""Tier A proof: compliance-neuro-symbolic — KYC/AML rule-ish decisions."""

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
from buildml.core.errors import MissingExtraError
from proofs._lib import TORCH_STATUS, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("compliance-neuro-symbolic", seed=27)
    rng = np.random.default_rng(ctx.seed)
    n = 420
    account_age_days = rng.uniform(1, 4000, size=n)
    wire_amount = rng.lognormal(9.2, 0.85, size=n)
    pep_score = rng.beta(2.5, 4.0, size=n)
    jurisdiction_risk = rng.beta(2.5, 3.5, size=n)
    # Softened escalation score + Bernoulli labels (not exact Boolean rules).
    score = (
        1.6 * ((wire_amount > 8000) & (jurisdiction_risk > 0.35)).astype(float)
        + 1.4 * ((account_age_days < 365) & (pep_score > 0.30)).astype(float)
        + rng.normal(0, 0.55, size=n)
    )
    escalate = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    frame = pd.DataFrame({
        "account_age_days": account_age_days,
        "wire_amount": wire_amount,
        "pep_score": pep_score,
        "jurisdiction_risk": jurisdiction_risk,
        "escalate": escalate,
    })
    session = (
        Session.ingest(frame)
        .set_roles({
            "account_age_days": "feature",
            "wire_amount": "feature",
            "pep_score": "feature",
            "jurisdiction_risk": "feature",
            "escalate": "target",
        })
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    fit = session.symbolic.fit(method="decision_tree", random_state=ctx.seed)
    val = session.symbolic.evaluate(partition="validation")
    test = session.symbolic.evaluate(partition="test")
    test_metrics = metrics_round(dict(getattr(test, "metrics", {}) or {}))
    for key in ("accuracy", "f1", "f1_macro", "f1_weighted"):
        value = test_metrics.get(key)
        if isinstance(value, (int, float)) and float(value) >= 0.97:
            raise SystemExit(
                "compliance-neuro-symbolic refused perfect-score theater: "
                f"{key}={float(value):.4f} >= 0.97 on noisy KYC labels."
            )
    neuro = {"ran": False, "skip_torch_paths": TORCH_STATUS.get("skip_torch_paths", True)}
    if not TORCH_STATUS.get("skip_torch_paths"):
        try:
            nf = session.symbolic.fit_neuro(method="nam", random_state=ctx.seed, epochs=5)
            ne = session.symbolic.evaluate_neuro(partition="validation")
            neuro = {
                "ran": True,
                "fit": metrics_round(nf.to_dict() if hasattr(nf, "to_dict") else {}),
                "validation_metrics": metrics_round(dict(getattr(ne, "metrics", {}) or {})),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            neuro = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    bundle = session.symbolic.save_bundle(ctx.artifacts_dir / "symbolic_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_kyc_aml_escalation",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": test_metrics,
        "neuro_symbolic": neuro,
        "torch": TORCH_STATUS,
        "bundle_path": str(bundle),
        "leakage_controls": ["Stratified split", "Symbolic fit on train", "Test after lock"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn DecisionTree twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Not legal advice; rule fidelity ≠ compliance certification",
            "Distinct KYC/AML narrative from policy-rules-neuro-symbolic",
            "Labels are probabilistic soft-rule draws, not exact Boolean rules",
        ],
    })
    print("compliance-neuro-symbolic OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
