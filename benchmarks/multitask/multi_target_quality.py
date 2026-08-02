"""Multi-target quality benchmark across sklearn/industry/torch backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import importlib.util

import numpy as np
import pandas as pd

from buildml import Session
from buildml.multitask.catalog import multitask_capability_matrix
from buildml.multitask.extras import xgboost_available
from buildml.dl.extras import torch_spec_available


def _torch_spec_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cls_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    frame = pd.DataFrame(x, columns=["f0", "f1", "f2"])
    frame["t1"] = (x[:, 0] > 0).astype(int)
    frame["t2"] = (x[:, 1] > 0).astype(int)
    return frame


def _reg_frame(n: int = 240, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    frame = pd.DataFrame(x, columns=["f0", "f1", "f2"])
    frame["t1"] = x[:, 0] * 1.4 + rng.normal(0, 0.08, size=n)
    frame["t2"] = x[:, 1] * -0.9 + rng.normal(0, 0.08, size=n)
    return frame


def _run(
    frame: pd.DataFrame,
    *,
    backend: str,
    method: str,
    task: str,
    epochs: int = 30,
) -> dict[str, object]:
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "f0": "feature",
                "f1": "feature",
                "f2": "feature",
                "t1": "target",
                "t2": "target",
            }
        )
        .split(test_size=0.25, random_state=0)
        .scale(method="standard")
    )
    kwargs: dict[str, object] = {
        "backend": backend,
        "method": method,
        "task": task,
        "prefer_reduce_components": False,
        "epochs": epochs,
        "batch_size": 32,
    }
    if backend == "sklearn" and task == "regression":
        kwargs["base_estimator"] = "ridge"
    fit = session.fit_multitask(**kwargs)  # type: ignore[arg-type]
    ev = session.evaluate_multitask(partition="test")
    row: dict[str, object] = {
        "backend": backend,
        "method": method,
        "task": task,
        "n_tasks": fit.n_tasks,
        "metrics": ev.metrics,
    }
    if task == "classification":
        row["score"] = ev.metrics.get("mean_accuracy")
    else:
        row["score"] = ev.metrics.get("mean_r2")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML multi-task multi-target quality benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/multitask/results/multi_target_quality.json"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(
        _run(_cls_frame(), backend="sklearn", method="multi_output", task="classification")
    )
    runs.append(
        _run(_reg_frame(), backend="sklearn", method="multi_output", task="regression")
    )
    if xgboost_available():
        runs.append(
            _run(
                _cls_frame(),
                backend="industry",
                method="multi_output_xgb",
                task="classification",
            )
        )
        runs.append(
            _run(
                _reg_frame(),
                backend="industry",
                method="multi_output_xgb",
                task="regression",
            )
        )
    if torch_spec_available():
        try:
            runs.append(
                _run(
                    _cls_frame(),
                    backend="torch",
                    method="shared_trunk_multihead",
                    task="classification",
                    epochs=args.epochs,
                )
            )
            runs.append(
                _run(
                    _reg_frame(),
                    backend="torch",
                    method="shared_trunk_multihead",
                    task="regression",
                    epochs=args.epochs,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if "torch" not in str(exc).lower():
                raise

    cls_runs = [r for r in runs if r["task"] == "classification"]
    reg_runs = [r for r in runs if r["task"] == "regression"]
    sklearn_cls = next((r for r in cls_runs if r["backend"] == "sklearn"), None)
    sklearn_reg = next((r for r in reg_runs if r["backend"] == "sklearn"), None)
    payload = {
        "benchmark": "multitask_multi_target_quality",
        "capability_matrix": multitask_capability_matrix(),
        "results": runs,
        "sklearn_cls_accuracy": None if sklearn_cls is None else sklearn_cls["score"],
        "sklearn_reg_r2": None if sklearn_reg is None else sklearn_reg["score"],
        "floor_note": (
            "Industry/torch backends should meet or exceed sklearn baseline on "
            "these synthetic multi-target tasks when extras are installed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    for task_kind, baseline in (
        ("classification", sklearn_cls),
        ("regression", sklearn_reg),
    ):
        if baseline is None:
            continue
        base_score = float(baseline["score"] or 0.0)
        industry = [
            r
            for r in runs
            if r["task"] == task_kind and r["backend"] in {"industry", "torch"}
        ]
        if not industry:
            continue
        best = max(industry, key=lambda r: float(r["score"] or 0.0))
        if task_kind == "classification" and float(best["score"] or 0.0) < base_score - 0.12:
            print(
                f"WARN: best {task_kind} industry/torch trails sklearn by >12 pts.",
                file=sys.stderr,
            )
        if task_kind == "regression" and float(best["score"] or 0.0) < base_score - 0.15:
            print(
                f"WARN: best {task_kind} industry/torch trails sklearn by >0.15 R².",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
