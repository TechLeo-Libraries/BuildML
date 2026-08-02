"""Partial-label semi-supervised benchmark across sklearn/industry/torch backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.dl.extras import torch_available, torch_spec_available
from buildml.ingest.detect import schema_from_dataframe
from buildml.semisupervised.catalog import semisupervised_capability_matrix
from buildml.semisupervised.extras import xgboost_available


def _synthetic_frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.2, -0.8, 0.3], 0.7, size=(n // 2, 3))
    x1 = rng.normal([1.1, 0.9, -0.4], 0.7, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["f0", "f1", "f2"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _mask_train(session: Session, fraction: float = 0.65) -> Session:
    rng = np.random.default_rng(2)
    full = session.to_pandas().copy()
    idx = list(session.split_plan.train_indices)
    blank = rng.choice(idx, size=max(1, int(fraction * len(idx))), replace=False)
    full.loc[blank, "y"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def _run_method(
    backend: str,
    method: str,
    *,
    epochs: int = 20,
) -> dict[str, object]:
    session = (
        Session.ingest(_synthetic_frame())
        .set_roles({"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask_train(session)
    fit = session.fit_semisupervised(
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        prefer_reduce_components=False,
        epochs=epochs,
        max_self_train_iter=8,
        threshold=0.7,
    )
    ev = session.evaluate_semisupervised(partition="test")
    return {
        "backend": backend,
        "method": method,
        "n_labeled_train": fit.n_labeled_train,
        "n_unlabeled_train": fit.n_unlabeled_train,
        "test_accuracy": ev.metrics.get("accuracy"),
        "test_f1_macro": ev.metrics.get("f1_macro"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML semi-supervised partial-label benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/semisupervised/results/partial_labels.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_method("sklearn", "label_propagation"))
    runs.append(_run_method("sklearn", "self_training"))
    if xgboost_available():
        runs.append(_run_method("industry", "pseudo_label_xgb"))
    if torch_spec_available():
        try:
            runs.append(_run_method("torch", "fixmatch_tabular", epochs=args.epochs))
            runs.append(_run_method("torch", "mixmatch_tabular", epochs=args.epochs))
        except Exception as exc:  # noqa: BLE001 — broken torch wheels on some hosts
            if "torch" not in str(exc).lower():
                raise

    sklearn_best = max(
        (r for r in runs if r["backend"] == "sklearn"),
        key=lambda r: float(r["test_accuracy"] or 0.0),
        default=None,
    )
    industry_runs = [r for r in runs if r["backend"] in {"industry", "torch"}]
    best = max(runs, key=lambda r: float(r["test_accuracy"] or 0.0))
    payload = {
        "benchmark": "semisupervised_partial_labels",
        "capability_matrix": semisupervised_capability_matrix(),
        "results": runs,
        "best_method": best["method"],
        "best_backend": best["backend"],
        "sklearn_baseline_accuracy": None if sklearn_best is None else sklearn_best["test_accuracy"],
        "floor_note": (
            "Industry/torch backends should meet or exceed sklearn baseline on this "
            "synthetic partial-label task when extras are installed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if sklearn_best and industry_runs:
        base_acc = float(sklearn_best["test_accuracy"] or 0.0)
        best_ind = max(industry_runs, key=lambda r: float(r["test_accuracy"] or 0.0))
        if float(best_ind["test_accuracy"] or 0.0) < base_acc - 0.08:
            print(
                "WARN: best industry/torch method trails sklearn baseline by >8 pts.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
