"""Active-learning query-efficiency benchmark (label budget vs accuracy curve)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.activelearning.catalog import activelearning_capability_matrix
from buildml.activelearning.extras import activelearning_industry_available
from buildml.data.dataset import Dataset
from buildml.dl.extras import torch_spec_available
from buildml.ingest.detect import schema_from_dataframe


def _synthetic_frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.2, -0.8, 0.3], 0.7, size=(n // 2, 3))
    x1 = rng.normal([1.1, 0.9, -0.4], 0.7, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["f0", "f1", "f2"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask_train(session: Session, fraction: float = 0.85) -> tuple[Session, pd.Series]:
    rng = np.random.default_rng(2)
    full = session.to_pandas().copy()
    truth = full["y"].copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session, truth


def _run_curve(
    backend: str,
    strategy: str,
    *,
    budgets: list[int],
    batch_size: int = 5,
    epochs: int = 25,
) -> dict[str, object]:
    session = (
        Session.ingest(_synthetic_frame())
        .set_roles({"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session, truth = _mask_train(session)
    max_budget = max(budgets)
    session.fit_active_learner(
        backend=backend,  # type: ignore[arg-type]
        strategy=strategy,  # type: ignore[arg-type]
        prefer_reduce_components=False,
        label_budget=max_budget,
        batch_size=batch_size,
        epochs=epochs,
    )
    points: list[dict[str, object]] = []
    spent = 0
    for budget in budgets:
        while spent < budget:
            remaining = budget - spent
            q = session.suggest_query(batch_size=min(batch_size, remaining))
            if not q.indices:
                break
            labels = [int(truth.loc[i]) for i in q.indices]
            session.label_rows(indices=q.indices, labels=labels)
            spent += len(q.indices)
        ev = session.evaluate_active_learning(partition="test")
        points.append(
            {
                "label_budget": budget,
                "labels_used": session.activelearning_plan.n_queries_used,
                "test_accuracy": ev.metrics.get("accuracy"),
                "test_f1_macro": ev.metrics.get("f1_macro"),
            }
        )
    return {
        "backend": backend,
        "strategy": strategy,
        "curve": points,
        "final_accuracy": points[-1]["test_accuracy"] if points else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML active-learning query-efficiency benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/activelearning/results/query_efficiency.json"),
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0,5,10,20,30",
        help="Comma-separated label budgets to sample along the curve.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args(argv)
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    runs: list[dict[str, object]] = []
    runs.append(_run_curve("sklearn", "margin", budgets=budgets))
    runs.append(_run_curve("sklearn", "entropy", budgets=budgets))
    if activelearning_industry_available():
        runs.append(_run_curve("industry", "core_set", budgets=budgets))
        runs.append(_run_curve("industry", "qbc_kl", budgets=budgets))
    if torch_spec_available():
        try:
            runs.append(_run_curve("torch", "bald", budgets=budgets, epochs=args.epochs))
            runs.append(_run_curve("torch", "mc_dropout", budgets=budgets, epochs=args.epochs))
        except Exception as exc:  # noqa: BLE001
            if "torch" not in str(exc).lower():
                raise

    sklearn_runs = [r for r in runs if r["backend"] == "sklearn"]
    sklearn_best = max(
        sklearn_runs,
        key=lambda r: float((r.get("final_accuracy") or 0.0)),
        default=None,
    )
    payload = {
        "benchmark": "activelearning_query_efficiency",
        "capability_matrix": activelearning_capability_matrix(),
        "budgets": budgets,
        "results": runs,
        "sklearn_baseline_final_accuracy": (
            None if sklearn_best is None else sklearn_best.get("final_accuracy")
        ),
        "floor_note": (
            "Industry/torch backends should track or beat sklearn margin/entropy "
            "on this synthetic query-efficiency curve when extras are installed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if sklearn_best:
        base = float(sklearn_best.get("final_accuracy") or 0.0)
        for run in runs:
            if run["backend"] == "sklearn":
                continue
            final = float(run.get("final_accuracy") or 0.0)
            if final < base - 0.12:
                print(
                    f"WARN: {run['backend']}/{run['strategy']} trails sklearn baseline "
                    f"by >12 pts at max budget.",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
