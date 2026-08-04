"""Tier A proof: unsupervised clustering on sklearn wine with external ARI."""

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
    load_sklearn_wine,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    write_results,
)


EXTERNAL = "cultivar"


def main() -> None:
    ctx = new_proof_context("wine-cluster-segments", seed=19)
    frame, data_meta = load_sklearn_wine()
    features = list(data_meta["feature_columns"])
    n_clusters = int(data_meta.get("n_classes") or frame[EXTERNAL].nunique())

    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in features}, EXTERNAL: "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=min(5, len(features)), prefix="pc")
    )
    plan = session.split_plan
    assert plan is not None
    counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    fit = session.unsupervised.fit(
        method="kmeans",
        n_clusters=n_clusters,
        random_state=ctx.seed,
    )
    assert_no_test_in_selection(
        selection_partition="validation",
        evaluation_partition="test",
    )
    val = session.unsupervised.evaluate(
        partition="validation",
        external_label_column=EXTERNAL,
    )
    test = session.unsupervised.evaluate(
        partition="test",
        external_label_column=EXTERNAL,
    )
    test_metrics = metrics_round(
        test.to_dict() if hasattr(test, "to_dict") else {}
    )
    external = dict(test_metrics.get("external_metrics") or {})
    refuse_perfect_scores(
        external,
        keys=("adjusted_rand_index", "normalized_mutual_info"),
        ceiling=1.0,
        proof_slug="wine-cluster-segments",
        context="sklearn wine external cluster validation",
    )
    # Also refuse near-perfect ARI theater on this small but real table.
    ari = external.get("adjusted_rand_index")
    if isinstance(ari, (int, float)) and float(ari) >= 0.98:
        raise SystemExit(
            "wine-cluster-segments refused near-perfect ARI theater: "
            f"adjusted_rand_index={float(ari):.4f} >= 0.98 on real wine cultivars."
        )

    bundle = session.unsupervised.save_bundle(ctx.artifacts_dir / "unsupervised_bundle")
    write_results(
        ctx,
        {
            "status": "completed",
            "evidence_tier": "REAL_PUBLIC_DATASET",
            "data": data_meta,
            "split": {
                "kind": plan.kind,
                "protocol": "random_train_validation_test_0.6_0.2_0.2",
                "counts": counts,
                "random_state": ctx.seed,
            },
            "method": "kmeans",
            "n_clusters": n_clusters,
            "fit": metrics_round(
                fit.to_dict() if hasattr(fit, "to_dict") else {"repr": str(fit)}
            ),
            "validation_metrics": metrics_round(
                val.to_dict() if hasattr(val, "to_dict") else {}
            ),
            "test_metrics": test_metrics,
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Scale + PCA fit on train only",
                "Clusters fit on train",
                "Cultivar labels role=ignore (external validation only)",
                "Test cluster metrics after model locked",
            ],
            "limitations": [
                "Wine n=178 is small; cultivar labels are chemistry classes",
                "Production clustering often lacks ground-truth segments",
                "Refuses ARI/NMI >= 1.0 and ARI >= 0.98",
            ],
        },
    )
    print("wine-cluster-segments OK", external)


if __name__ == "__main__":
    try:
        main()
    except MissingExtraError as exc:
        ctx = new_proof_context("wine-cluster-segments", seed=19)
        write_results(ctx, {"status": "skipped_missing_extra", "error": str(exc)})
        raise
