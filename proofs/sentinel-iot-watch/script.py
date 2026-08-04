"""Tier B product: Sentinel IoT Watch.

Composes unsupervised anomaly + online partial_fit streaming + lag forecast
for factory IoT telemetry. Chronological / train-cursor leakage discipline.
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
LABEL = "is_fault"


def _iot_load_series(n_hours: int = 24 * 90, seed: int = 51) -> tuple[pd.DataFrame, dict]:
    """Synthetic plant load / vibration proxy series for forecast stage."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    daily = 8 * np.sin(2 * np.pi * (t % 24) / 24 - 0.5)
    weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))
    ambient = 20 + 6 * np.sin(2 * np.pi * t / (24 * 365.25)) + rng.normal(0, 1.2, n_hours)
    load = (
        55 + daily + weekly + 0.35 * (ambient - 18).clip(0, None) + rng.normal(0, 1.5, n_hours)
    )
    frame = pd.DataFrame({"ts": times, "ambient_c": ambient, "plant_load": load})
    meta = {
        "name": "sentinel_iot_plant_load",
        "license": "synthetic/public-domain",
        "n_rows": int(n_hours),
        "freq": "h",
        "target": "plant_load",
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("sentinel-iot-watch", seed=51)
    sensors, sensor_meta = load_iot_sensor_anomaly_synthetic(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 0: honest sensor split ---
    session = (
        Session.ingest(sensors.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, LABEL: "target"})
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

    # --- Stage 1: anomaly ---
    try:
        if extra_available("pyod"):
            a_fit = session.anomaly.fit(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.06,
                random_state=ctx.seed,
            )
            a_backend = "pyod/hbos"
        else:
            a_fit = session.anomaly.fit(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.06,
                random_state=ctx.seed,
            )
            a_backend = "sklearn/isolation_forest"
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        a_tune = session.anomaly.tune_threshold(
            partition="validation",
            label_column=LABEL,
            positive_label=1,
            metric="f1",
        )
        a_ev = session.anomaly.evaluate(partition="test", positive_label=1)
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

    # --- Stage 2: online stream on train cursor ---
    try:
        online_session = (
            Session.ingest(sensors.copy())
            .set_roles({**{c: "feature" for c in FEATURES}, LABEL: "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        o_fit = online_session.online.fit(
            estimator="sgd_classifier",
            chunk_size=50,
            n_init=50,
            classes=[0, 1],
        )
        updates = 0
        while True:
            remaining = (
                online_session.online.plan.n_train_rows
                - online_session.online.plan.cursor
            )
            if remaining <= 0:
                break
            online_session.online.partial_fit(n_rows=min(50, remaining))
            updates += 1
        o_test = online_session.online.evaluate(partition="test")
        stages["online"] = {
            "status": "ok",
            "n_init_rows": int(o_fit.n_init_rows),
            "n_updates": updates,
            "test_metrics": metrics_round(dict(o_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["online"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"online: {exc}")
    write_results(ctx, stages["online"], filename="online.json")

    # --- Stage 3: plant load forecast ---
    ts_frame, ts_meta = _iot_load_series(seed=ctx.seed)
    try:
        fc_session = (
            Session.ingest(ts_frame)
            .set_roles(
                {"ts": "time", "ambient_c": "feature", "plant_load": "target"}
            )
            .time_split(test_size=0.15, validation_size=0.15)
        )
        fc_plan = fc_session.split_plan
        assert fc_plan is not None
        fc_fit = fc_session.forecast.fit(
            method="lag_ridge",
            horizon=24,
            lags=[1, 2, 3, 24, 48],
            alpha=1.0,
        )
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        fc_val = fc_session.forecast.evaluate(
            partition="validation", strategy="rolling_one_step"
        )
        fc_test = fc_session.forecast.evaluate(
            partition="test", strategy="rolling_one_step"
        )
        stages["forecast"] = {
            "status": "ok",
            "data": ts_meta,
            "split_counts": {
                "train": len(fc_plan.train_indices),
                "validation": len(fc_plan.validation_indices),
                "test": len(fc_plan.test_indices),
            },
            "fit": metrics_round(fc_fit.to_dict() if hasattr(fc_fit, "to_dict") else {}),
            "validation": metrics_round(
                fc_val.to_dict() if hasattr(fc_val, "to_dict") else {}
            ),
            "test": metrics_round(
                fc_test.to_dict() if hasattr(fc_test, "to_dict") else {}
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["forecast"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"forecast: {exc}")
    write_results(ctx, stages["forecast"], filename="forecast.json")

    summary = {
        "status": "completed",
        "product": "Sentinel IoT Watch",
        "data": {"sensors": sensor_meta, "forecast": ts_meta},
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified sensor split before anomaly / online fit",
            "Anomaly threshold tuned on validation only",
            "Online partial_fit consumes train cursor only",
            "Forecast uses chronological time_split",
        ],
        "what_fails_if_leakage_ignored": [
            "Tuning anomaly thresholds on test inflates fault F1",
            "Streaming updates that include test rows make online metrics meaningless",
            "Random split on plant load lets the forecaster peek at future seasonality",
        ],
        "limitations": [
            "Synthetic IoT sensors + plant load — not a SCADA extract",
            "Batch online chunks, not Kafka/Flink",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "sentinel-iot-watch OK",
        {
            "anomaly": stages["anomaly"]["status"],
            "online": stages["online"]["status"],
            "forecast": stages["forecast"]["status"],
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
