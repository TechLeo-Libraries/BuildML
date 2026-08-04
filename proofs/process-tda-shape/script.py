"""Tier A proof: process-tda-shape — manufacturing process clouds via TDA."""

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
    ctx = new_proof_context("process-tda-shape", seed=21)
    tda_ok = extra_available("ripser") and extra_available("persim")
    if not tda_ok:
        write_results(ctx, {
            "status": "skipped_missing_extra",
            "extra": "tda",
            "error": "ripser/persim not importable",
            "torch": None,
        })
        print("process-tda-shape SKIPPED (no tda extra)")
        return
    rng = np.random.default_rng(ctx.seed)
    # In-spec process cloud vs drifted (hotter / noisier) cloud — manufacturing, not credit.
    ok = rng.normal(size=(170, 5)) * np.array([1.0, 0.8, 0.6, 0.5, 0.4])
    drift = rng.normal(size=(170, 5)) * np.array([1.8, 1.4, 1.1, 0.9, 0.7]) + np.array(
        [1.8, -0.6, 0.4, 0.0, 0.0]
    )
    frame = pd.DataFrame(
        np.vstack([ok, drift]),
        columns=["temp_z", "pressure_z", "vibration_z", "flow_z", "torque_z"],
    )
    frame["pass_fail"] = [1] * 170 + [0] * 170
    feats = ["temp_z", "pressure_z", "vibration_z", "flow_z", "torque_z"]
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in feats}, "pass_fail": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed, stratify=True)
        .scale(method="standard")
    )
    try:
        fit = session.tda.fit(
            vectorization="persistence_image", knn=12, n_bins=12,
            head="logistic_regression", random_state=ctx.seed,
        )
    except MissingExtraError as exc:
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        print("process-tda-shape SKIPPED", exc)
        return
    val = session.tda.evaluate(partition="validation")
    test = session.tda.evaluate(partition="test")
    bundle = session.tda.save_bundle(ctx.artifacts_dir / "tda_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_process_clouds",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "validation_metrics": metrics_round(dict(val.metrics)),
        "test_metrics": metrics_round(dict(test.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Stratified split before any fit",
            "Scale+TDA fit on train only",
            "Test session.tda.evaluate after lock",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn logistic on raw scaled process "
                "features (no TDA); run script then baseline_industry.py for "
                "results/comparison.json."
            ),
        },
        "limitations": [
            "Synthetic sensor clouds; TDA descriptors are not plant SPC charts",
            "Distinct from credit-tda-shape (manufacturing process narrative)",
        ],
    })
    print("process-tda-shape OK", dict(test.metrics))


if __name__ == "__main__":
    main()
