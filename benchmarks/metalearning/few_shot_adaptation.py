"""Meta-learning few-shot adaptation benchmark (sklearn vs torch vs industry)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.metalearning.catalog import metalearning_capability_matrix
from buildml.metalearning.extras import (
    metalearning_industry_available,
    metalearning_torch_available,
)


def _episodic_frame(n_tasks: int = 10, n_per_task: int = 36, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for task in range(n_tasks):
        shift = rng.normal(0, 0.9, size=3)
        for i in range(n_per_task):
            label = i % 2
            center = shift + (1.1 if label else -1.1)
            x = rng.normal(center, 0.35, size=3)
            rows.append(
                {
                    "f0": float(x[0]),
                    "f1": float(x[1]),
                    "f2": float(x[2]),
                    "label": int(label),
                    "task_id": f"t{task}",
                }
            )
    return pd.DataFrame(rows)


def _run_backend(
    backend: str,
    method: str,
    *,
    k_shot: int = 3,
    n_episodes: int = 10,
    meta_epochs: int = 20,
) -> dict[str, object]:
    session = (
        Session.ingest(_episodic_frame())
        .set_roles(
            {
                "f0": "feature",
                "f1": "feature",
                "f2": "feature",
                "label": "target",
                "task_id": "group",
            }
        )
        .split(test_size=0.25, validation_size=0.15, random_state=1)
        .scale(method="standard")
    )
    fit = session.fit_metalearning(
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        k_shot=k_shot,
        n_query=6,
        n_episodes=n_episodes,
        meta_epochs=meta_epochs,
        task_holdout_fraction=0.2,
        prefer_reduce_components=False,
    )
    ev = session.evaluate_metalearning(
        partition="test",
        k_shot=k_shot,
        prefer_novel_tasks=True,
    )
    return {
        "backend": backend,
        "method": method,
        "meta_train_accuracy": fit.meta_train_accuracy,
        "n_meta_train_tasks": fit.n_meta_train_tasks,
        "n_held_out_tasks": fit.n_held_out_tasks,
        "held_out_task_ids": list(session.metalearning_plan.held_out_task_ids),
        "eval_mean_accuracy": ev.metrics.get("mean_accuracy"),
        "eval_mean_f1_macro": ev.metrics.get("mean_f1_macro"),
        "n_tasks_evaluated": ev.n_tasks_evaluated,
        "novel_task_ids": list(ev.novel_task_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML meta-learning few-shot adaptation benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/metalearning/results/few_shot_adaptation.json"),
    )
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--meta-epochs", type=int, default=20)
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_backend("sklearn", "prototypical", k_shot=args.k_shot, n_episodes=args.n_episodes))
    runs.append(_run_backend("sklearn", "warm_start", k_shot=args.k_shot, n_episodes=args.n_episodes))
    if metalearning_torch_available():
        try:
            runs.append(
                _run_backend(
                    "torch",
                    "prototypical_torch",
                    k_shot=args.k_shot,
                    n_episodes=args.n_episodes,
                    meta_epochs=args.meta_epochs,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if "torch" not in str(exc).lower():
                raise
    if metalearning_industry_available():
        try:
            runs.append(
                _run_backend(
                    "industry",
                    "maml",
                    k_shot=args.k_shot,
                    n_episodes=args.n_episodes,
                    meta_epochs=args.meta_epochs,
                )
            )
            runs.append(
                _run_backend(
                    "industry",
                    "reptile",
                    k_shot=args.k_shot,
                    n_episodes=args.n_episodes,
                    meta_epochs=args.meta_epochs,
                )
            )
        except Exception as exc:  # noqa: BLE001
            if "learn2learn" not in str(exc).lower() and "torch" not in str(exc).lower():
                raise

    sklearn_runs = [r for r in runs if r["backend"] == "sklearn"]
    sklearn_best = max(
        sklearn_runs,
        key=lambda r: float(r.get("eval_mean_accuracy") or 0.0),
        default=None,
    )
    payload = {
        "benchmark": "metalearning_few_shot_adaptation",
        "capability_matrix": metalearning_capability_matrix(),
        "k_shot": args.k_shot,
        "results": runs,
        "sklearn_baseline_eval_accuracy": (
            None if sklearn_best is None else sklearn_best.get("eval_mean_accuracy")
        ),
        "floor_note": (
            "Torch/industry backends should track sklearn prototypical/warm_start "
            "on this synthetic episodic benchmark when extras are installed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if sklearn_best:
        base = float(sklearn_best.get("eval_mean_accuracy") or 0.0)
        for run in runs:
            if run["backend"] == "sklearn":
                continue
            final = float(run.get("eval_mean_accuracy") or 0.0)
            if final < base - 0.15:
                print(
                    f"WARN: {run['backend']}/{run['method']} trails sklearn baseline "
                    f"by >15 pts on holdout episodic accuracy.",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
