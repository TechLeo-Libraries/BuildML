"""Anomaly detector comparison benchmark across sklearn / PyOD / torch backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.anomaly.catalog import anomaly_capability_matrix
from buildml.anomaly.extras import pyod_available
from buildml.dl.extras import torch_available


def _reference_frame(n_normal: int = 300, n_fraud: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(n_normal, 4))
    fraud = rng.normal(4.5, 0.5, size=(n_fraud, 4))
    frame = pd.DataFrame(
        np.vstack([normal, fraud]),
        columns=[f"x{i}" for i in range(4)],
    )
    frame["is_fraud"] = [0] * n_normal + [1] * n_fraud
    return frame


def _run_detector(
    backend: str,
    method: str,
    *,
    mode: str = "unsupervised",
) -> dict[str, object]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({f"x{i}": "feature" for i in range(4)} | {"is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_anomaly(
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        contamination=0.1,
        ae_epochs=15,
        prefer_reduce_components=False,
    )
    session.tune_anomaly_threshold(partition="validation", metric="f1")
    ev_val = session.evaluate_anomaly(partition="validation")
    ev_test = session.evaluate_anomaly(partition="test")
    return {
        "backend": backend,
        "method": method,
        "mode": mode,
        "train_alert_rate": fit.train_alert_rate,
        "threshold": fit.threshold,
        "validation_average_precision": ev_val.labeled_metrics.get("average_precision"),
        "validation_f1": ev_val.labeled_metrics.get("f1"),
        "test_average_precision": ev_test.labeled_metrics.get("average_precision"),
        "test_alert_rate": ev_test.alert_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML anomaly detector comparison benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/anomaly/results/detector_comparison.json"),
    )
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_detector("sklearn", "isolation_forest"))
    runs.append(_run_detector("sklearn", "lof"))
    if pyod_available():
        for method in ("hbos", "copod", "ecod"):
            runs.append(_run_detector("pyod", method))
    if torch_available():
        try:
            runs.append(_run_detector("torch", "autoencoder"))
        except Exception as exc:
            runs.append({"backend": "torch", "method": "autoencoder", "error": str(exc)})

    supervised_session = (
        Session.ingest(_reference_frame())
        .set_roles({f"x{i}": "feature" for i in range(4)} | {"is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    supervised_session.fit_anomaly(method="supervised_hgb", mode="supervised")
    ev = supervised_session.evaluate_anomaly(partition="test", k=15)
    runs.append(
        {
            "backend": "sklearn",
            "method": "supervised_hgb",
            "mode": "supervised",
            "test_average_precision": ev.labeled_metrics.get("average_precision"),
            "test_precision_at_k": ev.labeled_metrics.get("precision_at_k"),
        }
    )

    payload = {
        "capability_matrix": anomaly_capability_matrix(),
        "runs": runs,
        "pyod_available": pyod_available(),
        "torch_available": torch_available(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
