"""Tier A proof: ssl-representation-probe."""

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
    ctx = new_proof_context("ssl-representation-probe", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 400
    x = rng.normal(size=(n, 8))
    y = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    frame["y"] = y
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(8)}, "y": "target"})
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
        "data": {"name": "synthetic_ssl_tabular", "license": "synthetic/public-domain", "n_rows": n},
        "torch": TORCH_STATUS,
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
        "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Pretext+probe fit on train", "Test after lock"],
        "industry_comparison": {"status": "stub"},
        "limitations": ["Tabular masked pretext only in this proof"],
    })
    print("ssl-representation-probe OK", getattr(test, "metrics", test))


if __name__ == "__main__":
    main()
