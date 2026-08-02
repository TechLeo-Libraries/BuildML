"""Cluster quality benchmark: modern methods vs legacy 3-method baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.dl.extras import torch_available
from buildml.unsupervised.extras import hdbscan_available


def _reference_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    labels = []
    for i, center in enumerate([(0.0, 0.0), (3.0, 3.0), (0.0, 3.5)]):
        pts = rng.normal(center, 0.45, size=(n // 3, 2))
        chunks.append(pts)
        labels.extend([i] * (n // 3))
    frame = pd.DataFrame(np.vstack(chunks), columns=["x", "y"])
    frame["segment"] = labels
    return frame


def _run_method(method: str, *, epochs: int = 30) -> dict[str, float | str | None]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    kwargs: dict = {"method": method, "random_state": 0, "prefer_reduce_components": False}
    if method in {"kmeans", "agglomerative", "spectral", "gmm", "dec", "idec"}:
        kwargs["n_clusters"] = 3
    if method == "dbscan":
        kwargs["eps"] = 0.8
        kwargs["min_samples"] = 5
        kwargs["n_clusters"] = None
    if method == "hdbscan":
        kwargs["hdbscan_min_cluster_size"] = 8
        kwargs["n_clusters"] = None
    if method in {"dec", "idec"}:
        kwargs["pretrain_epochs"] = epochs
        kwargs["finetune_epochs"] = epochs
        kwargs["batch_size"] = 64
    session.fit_clusters(**kwargs)
    ev = session.evaluate_clusters(
        partition="test",
        external_label_column="segment",
        compute_stability=method == "kmeans",
        stability_runs=5,
    )
    return {
        "method": method,
        "silhouette": ev.metrics.get("silhouette"),
        "davies_bouldin": ev.metrics.get("davies_bouldin"),
        "calinski_harabasz": ev.metrics.get("calinski_harabasz"),
        "adjusted_rand_index": ev.external_metrics.get("adjusted_rand_index"),
        "n_clusters_observed": ev.n_clusters_observed,
        "stability_ari_mean": ev.metrics.get("stability_ari_mean"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML unsupervised cluster quality benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/unsupervised/results/cluster_quality.json"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)

    baseline = ["kmeans", "agglomerative", "dbscan"]
    extended = ["gmm", "spectral", "optics", "mean_shift"]
    if hdbscan_available():
        extended.append("hdbscan")
    if torch_available():
        extended.extend(["dec", "idec"])

    methods = baseline + extended
    rows: list[dict[str, float | str | None]] = []
    for method in methods:
        try:
            rows.append(_run_method(method, epochs=args.epochs))
        except Exception as exc:  # noqa: BLE001 — optional torch / hdbscan paths
            rows.append({"method": method, "skipped": True, "error": str(exc)})
    baseline_best = max(
        (r for r in rows if r.get("method") in baseline and r.get("skipped") is not True),
        key=lambda r: float(r.get("adjusted_rand_index") or 0.0),
    )
    modern_best = max(
        (r for r in rows if r.get("skipped") is not True),
        key=lambda r: float(r.get("adjusted_rand_index") or 0.0),
    )
    payload = {
        "benchmark": "unsupervised_cluster_quality",
        "hdbscan_available": hdbscan_available(),
        "torch_available": torch_available(),
        "results": rows,
        "baseline_methods": baseline,
        "baseline_best_method": baseline_best["method"],
        "baseline_best_ari": baseline_best.get("adjusted_rand_index"),
        "best_method": modern_best["method"],
        "best_ari": modern_best.get("adjusted_rand_index"),
        "floor_note": "Extended methods should meet or exceed legacy 3-method baseline ARI on reference blobs.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if (
        baseline_best.get("adjusted_rand_index") is not None
        and modern_best.get("adjusted_rand_index") is not None
        and float(modern_best["adjusted_rand_index"]) < float(baseline_best["adjusted_rand_index"]) - 0.05
    ):
        print(
            "WARN: best extended method trails legacy baseline by >5 pts ARI — investigate.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
