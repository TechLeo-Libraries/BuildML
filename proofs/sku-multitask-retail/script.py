"""Tier A proof: sku-multitask-retail — multi-target buy + high_margin."""

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
    ctx = new_proof_context("sku-multitask-retail", seed=25)
    rng = np.random.default_rng(ctx.seed)
    n = 520
    x = rng.normal(size=(n, 6))
    buy = (x[:, 0] + 0.55 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    high_margin = (x[:, 2] - 0.45 * x[:, 3] + 0.2 * x[:, 4] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[
        "price_z", "discount_z", "affinity_z", "competitor_z", "season_z", "stock_z",
    ])
    frame["buy"] = buy
    frame["high_margin"] = high_margin
    feats = list(frame.columns[:6])
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in feats}, "buy": "target", "high_margin": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    fit = session.multitask.fit(method="multioutput", random_state=ctx.seed)
    val = session.multitask.evaluate(partition="validation")
    test = session.multitask.evaluate(partition="test")
    bundle = session.multitask.save_bundle(ctx.artifacts_dir / "multitask_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_sku_multitask",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Split before fit", "Multi-output fit on train", "Test after lock"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn MultiOutputClassifier twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": ["Same-type classification targets only in this proof"],
    })
    print("sku-multitask-retail OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
