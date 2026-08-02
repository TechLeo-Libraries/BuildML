"""FedAvg convergence benchmark on synthetic heterogeneous clients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.federated.catalog import federated_capability_matrix
from buildml.federated.extras import flwr_available
from buildml.federated.results import export_round_history


def _heterogeneous_frame(
    n_clients: int = 8,
    n_per: int = 50,
    *,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for client in range(n_clients):
        shift = rng.normal(0, 1.0, size=2)
        for i in range(n_per):
            label = i % 2
            center = shift + (1.2 if label else -1.2)
            x = rng.normal(center, 0.35, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "client_id": f"c{client}",
                }
            )
    return pd.DataFrame(rows)


def _run_backend(backend: str, *, n_rounds: int = 8) -> dict[str, object]:
    session = (
        Session.ingest(_heterogeneous_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "client_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    fit = session.fit_federated(
        backend=backend,  # type: ignore[arg-type]
        method="fedavg",
        estimator="sgd_classifier",
        n_rounds=n_rounds,
        local_epochs=2,
        client_fraction=0.75,
        random_state=0,
    )
    ev = session.evaluate_federated(partition="validation", per_client=True)
    metrics_over_rounds = [
        float(r.get("mean_client_train_metric") or 0.0)
        for r in fit.round_history
    ]
    return {
        "backend": backend,
        "n_clients": fit.n_clients,
        "n_rounds_completed": len(fit.round_history),
        "final_train_metric": fit.final_train_metric,
        "validation_accuracy": ev.metrics.get("accuracy"),
        "metrics_over_rounds": metrics_over_rounds,
        "monotone_non_decreasing_last": (
            metrics_over_rounds[-1] >= metrics_over_rounds[0]
            if len(metrics_over_rounds) >= 2
            else True
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML federated FedAvg convergence benchmark"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/federated/results/fedavg_convergence.json"),
    )
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_backend("native", n_rounds=args.rounds))
    if flwr_available():
        runs.append(_run_backend("flower", n_rounds=args.rounds))

    payload = {
        "capability_matrix": federated_capability_matrix(),
        "rounds_requested": args.rounds,
        "runs": runs,
        "summary": {
            "n_runs": len(runs),
            "native_final_train_metric": runs[0].get("final_train_metric"),
            "native_validation_accuracy": runs[0].get("validation_accuracy"),
        },
    }
    if len(runs) > 1:
        payload["summary"]["flower_validation_accuracy"] = runs[1].get(
            "validation_accuracy"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
