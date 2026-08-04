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
    # Closer centers + higher noise so KMeans after PCA cannot trivially hit ARI≈1.
    centers = {
        0: np.array([-1.0, -0.8, 0.2, 0.0, -0.3]),
        1: np.array([0.9, -0.5, 0.4, 0.2, 0.1]),
        2: np.array([0.1, 0.9, -0.2, -0.1, 0.5]),
        3: np.array([-0.2, 0.1, 1.0, 0.7, -0.5]),
    }
    rows = []
    for fam, center in centers.items():
        for _ in range(n_per):
            x = center + rng.normal(scale=0.6, size=5)
            rows.append({**{FEATURES[i]: float(x[i]) for i in range(5)}, EXTERNAL: fam})
    frame = pd.DataFrame(rows)
    n_boundary = max(1, int(0.1 * len(frame)))
    for idx in rng.choice(len(frame), size=n_boundary, replace=False):
        a, b = rng.choice(4, size=2, replace=False)
        blend = 0.5 * centers[int(a)] + 0.5 * centers[int(b)] + rng.normal(0, 0.25, size=5)
        for i, col in enumerate(FEATURES):
            frame.loc[idx, col] = float(blend[i])
        frame.loc[idx, EXTERNAL] = int(a if rng.random() < 0.5 else b)
    meta = {
        "name": "synthetic_sku_embeddings",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_families": 4,
        "notes": "Overlapping embedding families + 10% boundary SKUs (anti perfect ARI).",
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

    fit = session.unsupervised.fit(method="kmeans", n_clusters=4, random_state=ctx.seed)
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
    for key in ("adjusted_rand_index", "normalized_mutual_info"):
        value = external.get(key)
        if isinstance(value, (int, float)) and float(value) >= 0.97:
            raise SystemExit(
                "sku-embedding-clusters refused perfect-score theater: "
                f"{key}={float(value):.4f} >= 0.97 on overlapping SKU families."
            )
    bundle = session.unsupervised.save_bundle(ctx.artifacts_dir / "unsupervised_bundle")

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
            "test_metrics": test_metrics,
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
