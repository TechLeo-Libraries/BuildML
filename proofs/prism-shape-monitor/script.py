"""Tier B product: Prism Shape Monitor.

Composes TDA shape descriptors + unsupervised anomaly + classical supervised
pass/fail scoring for process monitoring. TDA stage skips if extras missing.
"""

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
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = ["temp_z", "pressure_z", "vibration_z", "flow_z", "torque_z"]
TARGET = "pass_fail"


def _process_clouds(n_per: int = 200, seed: int = 43) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    ok = rng.normal(size=(n_per, 5)) * np.array([1.0, 0.8, 0.6, 0.5, 0.4])
    drift = rng.normal(size=(n_per, 5)) * np.array([1.8, 1.4, 1.1, 0.9, 0.7]) + np.array(
        [1.8, -0.6, 0.4, 0.0, 0.0]
    )
    frame = pd.DataFrame(np.vstack([ok, drift]), columns=FEATURES)
    frame[TARGET] = [1] * n_per + [0] * n_per
    meta = {
        "name": "prism_process_clouds",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "positive_rate": float(frame[TARGET].mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("prism-shape-monitor", seed=43)
    frame, data_meta = _process_clouds(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: TDA (optional extras) ---
    tda_ok = extra_available("ripser") and extra_available("persim")
    if not tda_ok:
        stages["tda"] = {
            "status": "skipped",
            "error": "ripser/persim not importable",
        }
        skip_notes.append("tda: ripser/persim not importable")
    else:
        try:
            t_fit = session.tda.fit(
                vectorization="persistence_image",
                knn=12,
                n_bins=12,
                head="logistic_regression",
                random_state=ctx.seed,
            )
            t_val = session.tda.evaluate(partition="validation")
            t_test = session.tda.evaluate(partition="test")
            stages["tda"] = {
                "status": "ok",
                "fit": metrics_round(t_fit.to_dict() if hasattr(t_fit, "to_dict") else {}),
                "validation_metrics": metrics_round(dict(t_val.metrics)),
                "test_metrics": metrics_round(dict(t_test.metrics)),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            stages["tda"] = {
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skip_notes.append(f"tda: {exc}")
    write_results(ctx, stages["tda"], filename="tda.json")

    # --- Stage 2: unsupervised anomaly ---
    try:
        # Fresh session so TDA head does not own the anomaly path.
        a_session = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        if extra_available("pyod"):
            a_fit = a_session.anomaly.fit(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = a_session.anomaly.fit(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        a_tune = a_session.anomaly.tune_threshold(
            partition="validation",
            label_column=TARGET,
            positive_label=0,  # fail / drift is the rare class of interest
            metric="f1",
        )
        a_ev = a_session.anomaly.evaluate(partition="test", positive_label=0)
        stages["anomaly"] = {
            "status": "ok",
            "backend": a_backend,
            "fit_threshold": float(getattr(a_fit, "threshold", float("nan"))),
            "tune": metrics_round(a_tune.to_dict() if hasattr(a_tune, "to_dict") else {}),
            "test_labeled_metrics": metrics_round(
                dict(getattr(a_ev, "labeled_metrics", {}) or {})
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["anomaly"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"anomaly: {exc}")
    write_results(ctx, stages["anomaly"], filename="anomaly.json")

    # --- Stage 3: classical supervised pass/fail ---
    c_session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    c_session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    c_val = c_session.evaluate(partition="validation")
    c_test = c_session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(c_val.metrics)),
        "test_metrics": metrics_round(dict(c_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    summary = {
        "status": "completed",
        "product": "Prism Shape Monitor",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before TDA / anomaly / supervised fit",
            "Scale + TDA fit on train only",
            "Anomaly threshold tuned on validation only",
            "Test used once per stage after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting persistence images on the full cloud leaks holdout geometry",
            "Tuning anomaly thresholds on test inflates F1 for drift alerts",
            "Supervised pass/fail trained with test rows overstates SPC readiness",
        ],
        "limitations": [
            "Synthetic process clouds — not plant SPC charts",
            "TDA stage skipped when ripser/persim extras are missing",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "prism-shape-monitor OK",
        {
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
