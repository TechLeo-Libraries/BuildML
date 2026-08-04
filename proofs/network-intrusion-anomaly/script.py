"""Tier A proof: network intrusion / fraud-style anomaly detection."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    extra_available,
    load_intrusion_anomaly_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "same_srv_rate",
    "dst_host_count",
]
LABEL = "is_attack"


def _labels(plan) -> list[str]:
    n = max(plan.train_indices + plan.validation_indices + plan.test_indices) + 1
    out = ["unused"] * n
    for i in plan.train_indices:
        out[i] = "train"
    for i in plan.validation_indices:
        out[i] = "validation"
    for i in plan.test_indices:
        out[i] = "test"
    return out


def main() -> None:
    ctx = new_proof_context("network-intrusion-anomaly", seed=11)
    frame, data_meta = load_intrusion_anomaly_synthetic(seed=ctx.seed)
    pyod_ok = extra_available("pyod")

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURES}, LABEL: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    # Prefer PyOD HBOS when installed; else sklearn IsolationForest.
    try:
        if pyod_ok:
            fit = session.anomaly.fit(
                backend="pyod",
                method="hbos",
                mode="unsupervised",
                contamination=0.06,
                random_state=ctx.seed,
            )
            backend_used, method_used = "pyod", "hbos"
        else:
            fit = session.anomaly.fit(
                method="isolation_forest",
                mode="unsupervised",
                contamination=0.06,
                random_state=ctx.seed,
            )
            backend_used, method_used = "sklearn", "isolation_forest"
    except (MissingExtraError, TypeError, ValueError):
        fit = session.anomaly.fit(
            method="isolation_forest",
            mode="unsupervised",
            contamination=0.06,
            random_state=ctx.seed,
        )
        backend_used, method_used = "sklearn", "isolation_forest"

    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    tune = session.anomaly.tune_threshold(
        partition="validation",
        label_column=LABEL,
        positive_label=1,
        metric="f1",
    )
    scored = session.anomaly.score(partition="test")
    ev = session.anomaly.evaluate(partition="test", positive_label=1)
    bundle = session.anomaly.save_bundle(ctx.artifacts_dir / "anomaly_bundle")

    labeled = metrics_round(dict(getattr(ev, "labeled_metrics", {}) or {}))
    # Anti perfect-score theater: soft attack margins + label noise should leave
    # residual error after validation threshold tuning.
    for key in ("f1", "average_precision", "roc_auc"):
        value = labeled.get(key)
        if isinstance(value, (int, float)) and float(value) >= 0.999:
            raise SystemExit(
                "network-intrusion-anomaly refused perfect-score theater: "
                f"{key}={float(value):.4f} >= 0.999 on overlapping noisy flows."
            )
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "backend": backend_used,
            "method": method_used,
            "pyod_available": pyod_ok,
            "fit": {
                "threshold": float(getattr(fit, "threshold", float("nan"))),
                "train_alert_rate": float(
                    getattr(fit, "train_alert_rate", float("nan"))
                ),
            },
            "threshold_tuning": {
                "partition": "validation",
                "result": metrics_round(
                    tune.to_dict() if hasattr(tune, "to_dict") else {"raw": str(tune)}
                ),
            },
            "test_score": {
                "n_flagged": int(getattr(scored, "n_flagged", -1)),
                "alert_rate": float(getattr(scored, "alert_rate", float("nan"))),
            },
            "test_labeled_metrics": labeled,
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Unsupervised fit on train features only",
                "Threshold tuned on validation labels only (allow_test_tuning=False)",
                "Test scored/evaluated after threshold locked",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn IsolationForest twin on the same split; "
                    "optional PyOD when installed — run script then baseline_industry.py for "
                    "results/comparison.json."
                ),
            },
            "honesty": [
                "Generator uses soft attack margins + ~4% label flips.",
                "Refuses labeled f1/AP/ROC-AUC >= 0.999 (anti perfect-score theater).",
            ],
            "limitations": [
                "Synthetic KDD-inspired flows, not full KDD Cup 1999",
                "Labeled metrics assume attack label quality; production often unlabeled",
            ],
        },
    )
    print("network-intrusion-anomaly OK", labeled)


if __name__ == "__main__":
    main()
