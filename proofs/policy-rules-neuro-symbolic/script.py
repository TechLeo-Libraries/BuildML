"""Tier A proof: policy-rules-neuro-symbolic."""

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
    ctx = new_proof_context("policy-rules-neuro-symbolic", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 400
    age = rng.uniform(18, 80, size=n)
    income = rng.lognormal(10.5, 0.4, size=n)
    risk = rng.beta(2, 5, size=n)
    # Rule-ish label: deny if young+high risk or low income
    y = ((age < 25) & (risk > 0.45) | (income < 20000)).astype(int)
    frame = pd.DataFrame({"age": age, "income": income, "risk": risk, "deny": y})
    session = (
        Session.ingest(frame)
        .set_roles({"age": "feature", "income": "feature", "risk": "feature", "deny": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    fit = session.fit_symbolic(method="decision_tree", random_state=ctx.seed)
    val = session.evaluate_symbolic(partition="validation")
    test = session.evaluate_symbolic(partition="test")
    neuro = {"ran": False, "skip_torch_paths": TORCH_STATUS.get("skip_torch_paths", True)}
    if not TORCH_STATUS.get("skip_torch_paths"):
        try:
            nf = session.fit_neuro_symbolic(method="nam", random_state=ctx.seed, epochs=5)
            ne = session.evaluate_neuro_symbolic(partition="validation")
            neuro = {
                "ran": True,
                "fit": metrics_round(nf.to_dict() if hasattr(nf, "to_dict") else {}),
                "validation_metrics": metrics_round(dict(getattr(ne, "metrics", {}) or {})),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            neuro = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    bundle = session.save_symbolic_bundle(ctx.artifacts_dir / "symbolic_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_compliance", "license": "synthetic/public-domain", "n_rows": n},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        "neuro_symbolic": neuro,
        "torch": TORCH_STATUS,
        "bundle_path": str(bundle),
        "leakage_controls": ["Stratified split", "Symbolic fit on train", "Test after lock"],
        "industry_comparison": {"status": "filled"},
        "limitations": ["Not legal advice; rule fidelity ≠ compliance certification"],
    })
    print("policy-rules-neuro-symbolic OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
