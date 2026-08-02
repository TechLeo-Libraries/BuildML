"""Online / continual stream-accuracy benchmark (chunks vs full refit)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import frame_for_partition
from buildml.ingest.detect import schema_from_dataframe
from buildml.online.catalog import online_capability_matrix
from buildml.online.extras import online_industry_available
from buildml.dl.extras import torch_spec_available


def _synthetic_frame(n: int = 360, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -0.6, 0.2], 0.8, size=(n // 2, 3))
    x1 = rng.normal([1.0, 0.8, -0.3], 0.8, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["f0", "f1", "f2"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _chunk_curve(
    backend: str,
    estimator: str,
    *,
    chunk_size: int = 40,
    n_init: int = 40,
) -> dict[str, object]:
    session = (
        Session.ingest(_synthetic_frame())
        .set_roles({"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.fit_online(
        backend=backend,  # type: ignore[arg-type]
        estimator=estimator,
        chunk_size=chunk_size,
        n_init=n_init,
        classes=[0, 1],
        prefer_reduce_components=False,
    )
    points: list[dict[str, object]] = []
    while True:
        plan = session.online_plan
        assert plan is not None
        ev = session.evaluate_online(partition="test")
        points.append(
            {
                "n_seen_rows": plan.n_seen_rows,
                "n_updates": plan.n_updates,
                "accuracy": ev.metrics.get("accuracy"),
                "drift_detected": ev.drift_detected,
            }
        )
        remaining = max(0, len(session.split_plan.train_indices) - plan.cursor)
        if remaining <= 0:
            break
        session.partial_fit_online(n_rows=min(chunk_size, remaining))
    return {
        "backend": backend,
        "estimator": estimator,
        "chunk_size": chunk_size,
        "points": points,
        "final_accuracy": points[-1]["accuracy"] if points else None,
    }


def _full_refit_baseline(session_seed_frame: pd.DataFrame) -> float:
    session = (
        Session.ingest(session_seed_frame.copy())
        .set_roles({"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    train = frame_for_partition(session.dataset, session.split_plan, "train")
    test = frame_for_partition(session.dataset, session.split_plan, "test")
    x_train = train[["f0", "f1", "f2"]].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=int)
    x_test = test[["f0", "f1", "f2"]].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=int)
    model = SGDClassifier(loss="log_loss", random_state=0)
    model.fit(x_train, y_train)
    return float((model.predict(x_test) == y_test).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/online/results/stream_accuracy.json"),
    )
    args = parser.parse_args()

    frame = _synthetic_frame()
    baseline = _full_refit_baseline(frame)
    runs: list[dict[str, object]] = [
        _chunk_curve("sklearn", "sgd_classifier"),
    ]
    if online_industry_available():
        try:
            runs.append(_chunk_curve("industry", "river_logistic", chunk_size=40))
        except (MissingExtraError, ValidationError, OSError):
            pass
    if torch_spec_available():
        try:
            runs.append(
                _chunk_curve(
                    "torch",
                    "replay_mlp",
                    chunk_size=40,
                )
            )
        except (MissingExtraError, ValidationError, OSError):
            pass

    payload = {
        "benchmark": "online_stream_accuracy",
        "capability_matrix": online_capability_matrix(),
        "full_refit_test_accuracy": baseline,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
