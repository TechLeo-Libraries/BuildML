"""Tier A proof: credit-tda-shape."""

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
from proofs._lib import extra_available, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("credit-tda-shape", seed=0)
    tda_ok = extra_available("ripser") and extra_available("persim")
    if not tda_ok:
        write_results(ctx, {
            "status": "skipped_missing_extra",
            "extra": "tda",
            "error": "ripser/persim not importable",
            "torch": None,
        })
        print("credit-tda-shape SKIPPED (no tda extra)")
        return
    rng = np.random.default_rng(ctx.seed)
    a = rng.normal(size=(160, 4))
    b = rng.normal(size=(160, 4)) * 1.6 + np.array([2.5, 0.0, 0.0, 0.0])
    frame = pd.DataFrame(np.vstack([a, b]), columns=[f"f{i}" for i in range(4)])
    frame["approved"] = [0] * 160 + [1] * 160
    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "approved": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed, stratify=True)
        .scale(method="standard")
    )
    try:
        fit = session.fit_tda(
            vectorization="persistence_image", knn=12, n_bins=12,
            head="logistic_regression", random_state=ctx.seed,
        )
    except MissingExtraError as exc:
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("credit-tda-shape SKIPPED", exc)
        return
    val = session.evaluate_tda(partition="validation")
    test = session.evaluate_tda(partition="test")
    bundle = session.save_tda_bundle(ctx.artifacts_dir / "tda_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_credit_clouds", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(val.metrics)),
        "test_metrics": metrics_round(dict(test.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": ["Stratified split", "Scale+TDA fit on train", "Test after lock"],
        "industry_comparison": {
            "status": "see_comparison_json",
            "note": "Run baseline_industry.py for sklearn logistic twin on same split",
        },
        "limitations": ["Synthetic clouds; TDA features are shape descriptors not FICO"],
    })
    print("credit-tda-shape OK", dict(test.metrics))


if __name__ == "__main__":
    main()
