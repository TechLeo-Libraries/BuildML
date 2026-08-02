"""CBR retrieval accuracy / latency benchmark (k vs accuracy and latency)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.cbr.catalog import cbr_capability_matrix
from buildml.cbr.extras import cbr_industry_available, hnswlib_available
from buildml.dl.extras import torch_available


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -0.5], 0.6, size=(n // 2, 2))
    x1 = rng.normal([1.0, 0.8], 0.6, size=(n - n // 2, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _run_backend(
    backend: str,
    *,
    k_values: list[int],
    repeats: int = 3,
) -> dict[str, object]:
    session = (
        Session.ingest(_synthetic_frame())
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_cbr(
        backend=backend,  # type: ignore[arg-type]
        task="classification",
        metric="euclidean",
        reuse="distance_weighted",
        prefer_reduce_components=False,
        k=max(k_values),
    )
    points: list[dict[str, object]] = []
    for k in k_values:
        latencies: list[float] = []
        acc = 0.0
        for _ in range(repeats):
            t0 = time.perf_counter()
            ev = session.evaluate_cbr(partition="test", k=k)
            latencies.append(time.perf_counter() - t0)
            acc = float(ev.metrics.get("accuracy", 0.0))
        points.append(
            {
                "k": k,
                "accuracy": acc,
                "latency_ms_mean": float(np.mean(latencies) * 1000.0),
                "latency_ms_std": float(np.std(latencies) * 1000.0),
            }
        )
    return {
        "backend": backend,
        "n_cases": fit.n_cases,
        "points": points,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CBR k vs accuracy/latency benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cbr/results/retrieval_accuracy.json"),
    )
    args = parser.parse_args(argv)

    matrix = cbr_capability_matrix()
    backends: list[str] = ["sklearn"]
    if cbr_industry_available():
        backends.append("industry")
    if torch_available():
        backends.append("torch")

    k_values = [1, 3, 5, 11, 21]
    results = {
        "capability_matrix_summary": {
            "default_backend": matrix["default_backend_when_installed"],
            "hnswlib_present": hnswlib_available(),
            "torch_present": torch_available(),
        },
        "runs": [_run_backend(b, k_values=k_values) for b in backends],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    # Floor: sklearn k=5 should beat random on separable synthetic data.
    sklearn_run = next(r for r in results["runs"] if r["backend"] == "sklearn")
    acc_k5 = next(p["accuracy"] for p in sklearn_run["points"] if p["k"] == 5)
    if float(acc_k5) < 0.75:
        print(f"FAIL: sklearn k=5 accuracy {acc_k5} below floor 0.75", file=sys.stderr)
        return 1
    print("CBR retrieval_accuracy benchmark passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
