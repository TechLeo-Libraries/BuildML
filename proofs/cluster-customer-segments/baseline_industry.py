"""Tier C: sklearn KMeans + silhouette twin for cluster-customer-segments."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_customer_segments_synthetic,
    metrics_round,
    new_proof_context,
    write_comparison,
)

FEATURES = ["recency_days", "frequency", "monetary"]
EXTERNAL = "true_segment"


def main() -> None:
    ctx = new_proof_context("cluster-customer-segments", seed=19)
    frame, _ = load_customer_segments_synthetic(n_per=250, seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATURES}, EXTERNAL: "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    test_idx = list(plan.test_indices)
    val_idx = list(plan.validation_indices)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_idx, FEATURES])
    x_test = scaler.transform(frame.loc[test_idx, FEATURES])
    pca = PCA(n_components=2, random_state=ctx.seed)
    z_train = pca.fit_transform(x_train)
    z_test = pca.transform(x_test)

    km = KMeans(n_clusters=4, random_state=ctx.seed, n_init=10)
    km.fit(z_train)
    pred_test = km.predict(z_test)
    y_ext = frame.loc[test_idx, EXTERNAL].to_numpy()
    industry_metrics = metrics_round(
        {
            "silhouette": float(silhouette_score(z_test, pred_test)),
            "ari": float(adjusted_rand_score(y_ext, pred_test)),
            "nmi": float(normalized_mutual_info_score(y_ext, pred_test)),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(bml_raw, prefer=("test_metrics",))
    # Flatten nested cluster metrics if needed.
    tm = bml_raw.get("test_metrics", {})
    if isinstance(tm, dict):
        for src, dst in (
            ("silhouette", "silhouette"),
            ("adjusted_rand_score", "ari"),
            ("ari", "ari"),
            ("normalized_mutual_info", "nmi"),
            ("nmi", "nmi"),
        ):
            if src in tm and dst not in bml_metrics:
                bml_metrics[dst] = tm[src]
        nested = tm.get("external_metrics") or tm.get("metrics") or {}
        if isinstance(nested, dict):
            for src, dst in (
                ("silhouette", "silhouette"),
                ("adjusted_rand_score", "ari"),
                ("ari", "ari"),
                ("normalized_mutual_info", "nmi"),
                ("nmi", "nmi"),
            ):
                if src in nested:
                    bml_metrics[dst] = nested[src]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session.fit_clusters",
            "method": bml_raw.get("method", "kmeans"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.KMeans+PCA",
            "test_metrics": industry_metrics,
            "leakage_controls": [
                "Scaler+PCA fit on train only",
                "KMeans fit on train only",
                "External labels used only for evaluation",
                "Same SplitPlan as BuildML Session",
            ],
            "validation_rows_reserved": len(val_idx),
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=("silhouette", "ari", "nmi"),
    )
    print("cluster-customer-segments Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
