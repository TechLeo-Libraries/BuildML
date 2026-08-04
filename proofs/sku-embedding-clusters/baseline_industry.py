"""Tier C: sklearn KMeans + PCA twin for sku-embedding-clusters."""

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
    metrics_round,
    new_proof_context,
    write_comparison,
)

FEATURES = ["emb0", "emb1", "emb2", "emb3", "price_z"]
EXTERNAL = "true_family"


def _load_sku_embeddings(n_per: int, seed: int) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("sku-embedding-clusters", seed=31)
    frame = _load_sku_embeddings(n_per=220, seed=ctx.seed)
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
            "backend": "buildml.session.unsupervised.fit",
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
    print("sku-embedding-clusters Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
