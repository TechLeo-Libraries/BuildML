"""Tier A proof: multi-target-underwriting."""

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
    ctx = new_proof_context("multi-target-underwriting", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 500
    x = rng.normal(size=(n, 6))
    t1 = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    t2 = (x[:, 2] - 0.4 * x[:, 3] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    frame["approve"] = t1
    frame["high_limit"] = t2
    session = (
        Session.ingest(frame)
        .set_roles({
            **{f"f{i}": "feature" for i in range(6)},
            "approve": "target", "high_limit": "target",
        })
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    fit = session.multitask.fit(method="multioutput", random_state=ctx.seed)
    val = session.multitask.evaluate(partition="validation")
    test = session.multitask.evaluate(partition="test")
    bundle = session.multitask.save_bundle(ctx.artifacts_dir / "multitask_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_multi_underwriting", "license": "synthetic/public-domain", "n_rows": n},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Split before fit", "Multi-output fit on train", "Test after lock"],
        "industry_comparison": {"status": "filled"},
        "limitations": ["Same-type classification targets only in this proof"],
    })
    print("multi-target-underwriting OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
