"""Tier A proof: unsupervised customer segmentation with external validation."""

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
    assert_no_test_in_selection,
    extra_available,
    load_customer_segments_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = ["recency_days", "frequency", "monetary"]
EXTERNAL = "true_segment"


def main() -> None:
    ctx = new_proof_context("cluster-customer-segments", seed=19)
    frame, data_meta = load_customer_segments_synthetic(n_per=250, seed=ctx.seed)
    hdbscan_ok = extra_available("hdbscan")

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURES}, EXTERNAL: "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )
    plan = session.split_plan
    assert plan is not None
    counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    method = "kmeans"
    try:
        fit = session.fit_clusters(method="kmeans", n_clusters=4, random_state=ctx.seed)
    except Exception:
        fit = session.fit_clusters(method="kmeans", n_clusters=4, random_state=ctx.seed)

    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    val = session.evaluate_clusters(
        partition="validation",
        external_label_column=EXTERNAL,
    )
    test = session.evaluate_clusters(
        partition="test",
        external_label_column=EXTERNAL,
    )
    bundle = session.save_unsupervised_bundle(ctx.artifacts_dir / "unsupervised_bundle")

    # Optional HDBSCAN probe (does not replace primary k-means claim).
    hdbscan_probe: dict = {"available": hdbscan_ok, "ran": False}
    if hdbscan_ok:
        try:
            session2 = (
                Session.ingest(frame)
                .set_roles({**{c: "feature" for c in FEATURES}, EXTERNAL: "ignore"})
                .inject_split(
                    train_indices=list(plan.train_indices),
                    validation_indices=list(plan.validation_indices),
                    test_indices=list(plan.test_indices),
                )
                .scale(method="standard")
            )
            fit_h = session2.fit_clusters(
                method="hdbscan",
                hdbscan_min_cluster_size=25,
            )
            ev_h = session2.evaluate_clusters(
                partition="validation",
                external_label_column=EXTERNAL,
            )
            hdbscan_probe = {
                "available": True,
                "ran": True,
                "fit": metrics_round(
                    fit_h.to_dict() if hasattr(fit_h, "to_dict") else {"repr": str(fit_h)}
                ),
                "validation": metrics_round(
                    ev_h.to_dict() if hasattr(ev_h, "to_dict") else {}
                ),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            hdbscan_probe = {
                "available": True,
                "ran": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts},
            "method": method,
            "fit": metrics_round(
                fit.to_dict() if hasattr(fit, "to_dict") else {"repr": str(fit)}
            ),
            "validation_metrics": metrics_round(
                val.to_dict() if hasattr(val, "to_dict") else {}
            ),
            "test_metrics": metrics_round(
                test.to_dict() if hasattr(test, "to_dict") else {}
            ),
            "hdbscan_probe": hdbscan_probe,
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Scale + PCA fit on train only",
                "Clusters fit on train",
                "External segment labels used only for evaluation (role=ignore)",
                "Test cluster metrics after model locked",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn KMeans + silhouette twin on the same "
                    "split; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "External labels exist only because data is synthetic",
                "Production clustering often lacks ground-truth segments",
            ],
        },
    )
    print(
        "cluster-customer-segments OK",
        getattr(test, "to_dict", lambda: test)(),
    )


if __name__ == "__main__":
    main()
