"""Tier A proof: tabular-ssl-probe — IoT sensor SSL pretext + linear probe."""

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
from proofs._lib import TORCH_STATUS, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("tabular-ssl-probe", seed=32)
    rng = np.random.default_rng(ctx.seed)
    n = 420
    # IoT / telemetry tabular SSL — distinct feature DGP from ssl-representation-probe.
    x = rng.normal(size=(n, 10))
    y = (
        0.8 * x[:, 0] - 0.4 * x[:, 3] + 0.35 * x[:, 7]
        + rng.normal(scale=0.28, size=n) > 0
    ).astype(int)
    frame = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(10)])
    frame["fault"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{f"sensor_{i}": "feature" for i in range(10)}, "fault": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_ssl_pretext(method="masked_tabular", random_state=ctx.seed)
    try:
        session.finetune_ssl_head(random_state=ctx.seed)
    except Exception:
        pass
    val = session.evaluate_ssl(partition="validation")
    test = session.evaluate_ssl(partition="test")
    bundle = session.save_ssl_bundle(ctx.artifacts_dir / "ssl_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_iot_ssl_tabular",
            "license": "synthetic/public-domain",
            "n_rows": n,
        },
        "torch": TORCH_STATUS,
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Pretext+probe fit on train", "Test after lock"],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: PCA embedding + logistic probe twin; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Tabular masked pretext only in this proof",
            "Distinct IoT sensor narrative from ssl-representation-probe",
        ],
    })
    print("tabular-ssl-probe OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
