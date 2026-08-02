"""TDA persistence pipeline benchmark (native vs giotto backends)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.tda.catalog import tda_capability_matrix
from buildml.tda.extras import giotto_available, tda_available


def _reference_frame(n_per_class: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_per_class, 4))
    b = rng.normal(size=(n_per_class, 4)) * 1.5 + np.array([2.5, 0.0, 0.0, 0.0])
    x = np.vstack([a, b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y
    return frame


def _run_backend(
    backend: str,
    vectorization: str,
    *,
    compare_diagram_distances: bool = True,
) -> dict[str, object]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    fit = session.fit_tda(
        backend=backend,  # type: ignore[arg-type]
        vectorization=vectorization,  # type: ignore[arg-type]
        knn=10,
        n_bins=12,
        head="logistic_regression",
        random_state=0,
        mapper=backend == "giotto",
    )
    ev = session.evaluate_tda(
        partition="validation",
        compare_diagram_distances=compare_diagram_distances,
        diagram_distance_metric="wasserstein",
    )
    return {
        "backend": backend,
        "vectorization": vectorization,
        "feature_dim": fit.feature_dim,
        "train_score": fit.train_score,
        "validation_accuracy": ev.metrics.get("accuracy"),
        "validation_macro_f1": ev.metrics.get("macro_f1"),
        "diagram_distances": dict(ev.diagram_distances),
        "n_disclosures": len(fit.disclosures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML TDA persistence pipeline benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/tda/results/persistence_pipeline.json"),
    )
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    matrix = tda_capability_matrix()

    if tda_available():
        for vec in ("persistence_image", "landscape", "silhouette"):
            runs.append(_run_backend("native", vec))
    if giotto_available():
        for vec in ("betti_curve", "persistence_image", "persistence_landscape"):
            runs.append(_run_backend("giotto", vec))

    payload = {
        "capability_matrix": matrix,
        "runs": runs,
        "tda_available": tda_available(),
        "giotto_available": giotto_available(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_runs": len(runs)}, indent=2))
    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
