"""SSL linear-probe benchmark: Torch methods vs legacy sklearn masked tabular."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.dl.extras import torch_available


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.0, -0.5, 0.2], 0.8, size=(n // 2, 3))
    x1 = rng.normal([1.2, 0.9, -0.3], 0.8, size=(n - n // 2, 3))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["f0", "f1", "f2"])
    frame["y"] = [0] * (n // 2) + [1] * (n - n // 2)
    return frame


def _linear_probe_accuracy(method: str, *, epochs: int = 25) -> dict[str, float | str | None]:
    session = (
        Session.ingest(_synthetic_frame())
        .set_roles({"f0": "feature", "f1": "feature", "f2": "feature", "y": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    kwargs: dict = {
        "method": method,
        "latent_dim": 8,
        "random_state": 0,
        "prefer_reduce_components": False,
    }
    if method == "masked_tabular":
        kwargs["max_iter"] = 80
    else:
        kwargs["epochs"] = epochs
        kwargs["batch_size"] = 32
    session.fit_ssl_pretext(**kwargs)
    session.finetune_ssl_head(estimator="logistic_regression", random_state=0)
    ev = session.evaluate_ssl(partition="test")
    return {
        "method": method,
        "accuracy": float(ev.metrics.get("accuracy", 0.0)),
        "f1_macro": float(ev.metrics.get("f1_macro", 0.0)),
        "pretext_loss": getattr(session.ssl_fit_result, "pretext_loss", None),
        "reconstruction_mae": getattr(session.ssl_fit_result, "reconstruction_mae", None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML SSL linear-probe benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/ssl/results/linear_probe_tabular.json"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args(argv)

    methods = ["masked_tabular"]
    if torch_available():
        methods = [
            "simclr_tabular",
            "byol_tabular",
            "vicreg_tabular",
            "mae_tabular",
            "masked_tabular",
        ]
    rows = [_linear_probe_accuracy(m, epochs=args.epochs) for m in methods]
    best = max(rows, key=lambda r: float(r["accuracy"]))
    legacy = next((r for r in rows if r["method"] == "masked_tabular"), None)
    payload = {
        "benchmark": "ssl_linear_probe_tabular",
        "torch_available": torch_available(),
        "results": rows,
        "best_method": best["method"],
        "legacy_sklearn_accuracy": None if legacy is None else legacy["accuracy"],
        "floor_note": "Torch SSL should meet or exceed legacy sklearn masked_tabular.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if legacy and float(best["accuracy"]) < float(legacy["accuracy"]) - 0.05:
        print(
            "WARN: best Torch method trails legacy sklearn by >5 pts — investigate.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
