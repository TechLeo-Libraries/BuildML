"""Tabular AutoML search benchmark: native vs optuna vs industry backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.automl.catalog import automl_capability_matrix
from buildml.automl.extras import (
    autogluon_available,
    flaml_available,
    optuna_available,
)
from buildml.automl.search import export_comparison_metrics
from buildml.automl.types import AutoMLBudget


def _reference_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = (0.85 * x1 - 0.45 * x2 + rng.normal(scale=0.35, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def _run_backend(
    backend: str,
    *,
    n_trials: int,
    time_budget: float | None,
) -> dict[str, object]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
    )
    kwargs: dict[str, object] = {
        "backend": backend,
        "n_trials": n_trials,
        "cv": 3,
        "include_recipe_search": True,
        "families": ("logistic", "random_forest", "gradient_boosting"),
        "random_state": 0,
        "budget": AutoMLBudget(max_trials=n_trials, max_time_seconds=time_budget),
    }
    if backend == "optuna":
        kwargs["method"] = "optuna"
    if time_budget is not None:
        kwargs["time_budget"] = time_budget

    result = session.run_automl(**kwargs)
    test = session.evaluate_automl(partition="test")
    return {
        "backend": backend,
        "best_family": result.best_family,
        "best_recipe_strategy": result.best_recipe_strategy,
        "selection_score": result.best_score,
        "test_accuracy": test.metrics.get("accuracy"),
        "test_f1_weighted": test.metrics.get("f1_weighted"),
        "n_trials": len(result.trials),
        "disclosures_head": list(result.disclosures[:3]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML AutoML tabular search benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/automl/results/tabular_search.json"),
    )
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args(argv)

    matrix = automl_capability_matrix()
    backends = ["native"]
    if optuna_available():
        backends.append("optuna")
    if flaml_available():
        backends.append("flaml")
    if autogluon_available():
        backends.append("autogluon")

    runs: list[dict[str, object]] = []
    for backend in backends:
        try:
            runs.append(
                _run_backend(
                    backend,
                    n_trials=args.n_trials,
                    time_budget=args.time_budget,
                )
            )
        except Exception as exc:  # noqa: BLE001 — benchmark disclosure
            runs.append({"backend": backend, "error": str(exc)})

    payload = {
        "capability_matrix": matrix,
        "backends_tested": backends,
        "n_trials": args.n_trials,
        "time_budget": args.time_budget,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    # Export last native trial comparison when available.
    native_runs = [r for r in runs if r.get("backend") == "native" and "error" not in r]
    if native_runs:
        session = (
            Session.ingest(_reference_frame())
            .set_roles({"x1": "feature", "x2": "feature", "cat": "feature", "y": "target"})
            .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        )
        result = session.run_automl(
            backend="native",
            n_trials=args.n_trials,
            cv=3,
            include_recipe_search=True,
            random_state=0,
        )
        export_comparison_metrics(result, args.out.parent / "tabular_search_trials.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
