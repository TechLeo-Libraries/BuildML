"""CBR retrieval accuracy / latency benchmark (k vs accuracy and latency)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


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


def _industry_spec_present() -> bool:
    """Cheap install check — avoid importing hnswlib/torch in the smoke process."""
    return (
        importlib.util.find_spec("hnswlib") is not None
        or importlib.util.find_spec("faiss") is not None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CBR k vs accuracy/latency benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cbr/results/retrieval_accuracy.json"),
    )
    args = parser.parse_args(argv)

    k_values = [1, 3, 5, 11, 21]
    # Sklearn-only first. Never import buildml.cbr.extras / hnswlib / torch before
    # the floor is recorded — native extension DLL init can process-kill on Windows.
    runs: list[dict[str, object]] = [_run_backend("sklearn", k_values=k_values)]

    core = {
        "capability_matrix_summary": {
            "default_backend": "sklearn",
            "hnswlib_spec_present": importlib.util.find_spec("hnswlib") is not None,
            "torch_probe": "deferred",
        },
        "runs": list(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(core, indent=2), encoding="utf-8")

    enable_industry = os.environ.get("BUILDML_BENCH_CBR_INDUSTRY", "") == "1" or (
        sys.platform != "win32" and _industry_spec_present()
    )
    if enable_industry and _industry_spec_present():
        try:
            runs.append(_run_backend("industry", k_values=k_values))
        except Exception as exc:  # noqa: BLE001
            runs.append({"backend": "industry", "skipped": True, "error": str(exc)})
    elif _industry_spec_present():
        runs.append(
            {
                "backend": "industry",
                "skipped": True,
                "reason": (
                    "Windows smoke skips in-process hnswlib import by default; "
                    "set BUILDML_BENCH_CBR_INDUSTRY=1 to enable."
                ),
            }
        )

    if sys.platform != "win32" or os.environ.get("BUILDML_BENCH_TORCH", "") == "1":
        try:
            from buildml.dl.extras import torch_available

            if torch_available():
                runs.append(_run_backend("torch", k_values=k_values))
            else:
                runs.append(
                    {
                        "backend": "torch",
                        "skipped": True,
                        "reason": "torch_available() is False",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            runs.append({"backend": "torch", "skipped": True, "error": str(exc)})
    else:
        runs.append(
            {
                "backend": "torch",
                "skipped": True,
                "reason": (
                    "Windows torch path skipped by default; "
                    "set BUILDML_BENCH_TORCH=1 to enable."
                ),
            }
        )

    results = {
        "capability_matrix_summary": {
            "default_backend": "sklearn",
            "hnswlib_spec_present": importlib.util.find_spec("hnswlib") is not None,
            "industry_enabled": enable_industry,
            "torch_probe": (
                "enabled"
                if (sys.platform != "win32" or os.environ.get("BUILDML_BENCH_TORCH") == "1")
                else "skipped_on_windows"
            ),
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    sklearn_run = next(r for r in results["runs"] if r["backend"] == "sklearn")
    if "error" in sklearn_run:
        print(f"FAIL: sklearn backend error: {sklearn_run['error']}", file=sys.stderr)
        return 1
    acc_k5 = next(p["accuracy"] for p in sklearn_run["points"] if p["k"] == 5)
    if float(acc_k5) < 0.75:
        print(f"FAIL: sklearn k=5 accuracy {acc_k5} below floor 0.75", file=sys.stderr)
        return 1
    print("CBR retrieval_accuracy benchmark passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
