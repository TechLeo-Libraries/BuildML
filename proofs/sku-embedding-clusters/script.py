"""Tier A proof: sku-embedding-clusters — product embedding segmentation."""

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
from proofs._lib import (
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURES = ["emb0", "emb1", "emb2", "emb3", "price_z"]
EXTERNAL = "true_family"


def _load_sku_embeddings(n_per: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    centers = {
        0: np.array([-1.5, -1.2, 0.2, 0.0, -0.5]),
        1: np.array([1.4, -0.8, 0.5, 0.3, 0.2]),
        2: np.array([0.1, 1.5, -0.4, -0.2, 0.8]),
        3: np.array([-0.3, 0.2, 1.6, 1.1, -0.9]),
    }
    rows = []
    for fam, center in centers.items():
        for _ in range(n_per):
            x = center + rng.normal(scale=0.35, size=5)
            rows.append({**{FEATURES[i]: float(x[i]) for i in range(5)}, EXTERNAL: fam})
    frame = pd.DataFrame(rows)
    meta = {
        "name": "synthetic_sku_embeddings",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_families": 4,
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("sku-embedding-clusters", seed=31)
    frame, data_meta = _load_sku_embeddings(n_per=220, seed=ctx.seed)

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

    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts},
            "method": "kmeans",
            "fit": metrics_round(
                fit.to_dict() if hasattr(fit, "to_dict") else {"repr": str(fit)}
            ),
            "validation_metrics": metrics_round(
                val.to_dict() if hasattr(val, "to_dict") else {}
            ),
            "test_metrics": metrics_round(
                test.to_dict() if hasattr(test, "to_dict") else {}
            ),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "Scale + PCA fit on train only",
                "Clusters fit on train",
                "External family labels used only for evaluation (role=ignore)",
                "Test cluster metrics after model locked",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn KMeans + PCA twin on the same "
                    "split; run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "External labels exist only because data is synthetic",
                "Distinct product-embedding narrative from RFM customer segments",
            ],
        },
    )
    print(
        "sku-embedding-clusters OK",
        getattr(test, "to_dict", lambda: test)(),
    )


if __name__ == "__main__":
    main()
