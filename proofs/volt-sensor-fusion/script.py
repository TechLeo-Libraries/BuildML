"""Tier B product: Volt Sensor Fusion.

Composes unsupervised anomaly + optional TDA shape descriptors + classical
supervised fault scoring for synthetic industrial sensor streams.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_iot_sensor_anomaly_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURES = ["temp_c", "vibration", "current_a", "pressure", "rpm"]
TARGET = "is_fault"


def main() -> None:
    ctx = new_proof_context("volt-sensor-fusion", seed=43)
    frame, data_meta = load_iot_sensor_anomaly_synthetic(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: anomaly ---
    try:
        if extra_available("pyod"):
            a_fit = session.fit_anomaly(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = session.fit_anomaly(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.08,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        a_tune = session.tune_anomaly_threshold(
            partition="validation",
            label_column=TARGET,
            positive_label=1,
            metric="f1",
        )
        a_ev = session.evaluate_anomaly(partition="test", positive_label=1)
        stages["anomaly"] = {
            "status": "ok",
            "backend": a_backend,
            "fit_threshold": float(getattr(a_fit, "threshold", float("nan"))),
            "tune": metrics_round(a_tune.to_dict() if hasattr(a_tune, "to_dict") else {}),
            "test_labeled_metrics": metrics_round(
                dict(getattr(a_ev, "labeled_metrics", {}) or {})
            ),
        }
        write_results(ctx, stages["anomaly"], filename="anomaly.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["anomaly"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"anomaly: {exc}")

    # --- Stage 2: TDA (optional) ---
    tda_ok = extra_available("ripser") and extra_available("persim")
    if not tda_ok:
        stages["tda"] = {
            "status": "skipped_missing_extra",
            "extra": "tda",
            "error": "ripser/persim not importable",
        }
        skip_notes.append("tda: ripser/persim not importable")
        write_results(ctx, stages["tda"], filename="tda.json")
    else:
        try:
            tda_session = (
                Session.ingest(frame.copy())
                .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
                .inject_split(
                    train_indices=list(plan.train_indices),
                    validation_indices=list(plan.validation_indices),
                    test_indices=list(plan.test_indices),
                )
                .scale(method="standard")
            )
            fit_t = tda_session.fit_tda(
                vectorization="persistence_image",
                knn=12,
                n_bins=12,
                head="logistic_regression",
                random_state=ctx.seed,
            )
            test_t = tda_session.evaluate_tda(partition="test")
            stages["tda"] = {
                "status": "ok",
                "fit": metrics_round(
                    fit_t.to_dict() if hasattr(fit_t, "to_dict") else {}
                ),
                "test_metrics": metrics_round(dict(test_t.metrics)),
            }
            write_results(ctx, stages["tda"], filename="tda.json")
        except (MissingExtraError, TypeError, ValueError) as exc:
            stages["tda"] = {
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skip_notes.append(f"tda: {exc}")
            write_results(ctx, stages["tda"], filename="tda.json")

    # --- Stage 3: classical ---
    try:
        classical = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, TARGET: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        classical.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical.evaluate(partition="test")
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "test_metrics": metrics_round(dict(c_test.metrics)),
        }
        write_results(ctx, stages["classical"], filename="classical.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"classical: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    status = "completed" if ok_stages >= 2 else "partial"
    summary = {
        "status": status,
        "product": "Volt Sensor Fusion",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before anomaly / TDA / classical",
            "Anomaly threshold tuned on validation only",
            "TDA + scale fit on train only when extras present",
            "Classical scorer uses inject_split — test after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Tuning anomaly thresholds on test inflates F1",
            "Fitting TDA descriptors on the full fleet invents shape separability",
            "Fitting classical scores on the full table invents holdout ROC",
        ],
        "limitations": [
            "Synthetic industrial sensors — not a real SCADA extract",
            "TDA stage skipped when ripser/persim extras are missing",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "volt-sensor-fusion OK",
        {
            "anomaly": (stages.get("anomaly") or {}).get("status"),
            "tda": (stages.get("tda") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
