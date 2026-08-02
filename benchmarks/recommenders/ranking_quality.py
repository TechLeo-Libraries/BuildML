"""Recommender ranking quality benchmark across sklearn / implicit / LightFM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.recommenders.catalog import recommender_capability_matrix
from buildml.recommenders.extras import implicit_available, lightfm_available


def _interaction_frame(*, n_users: int = 50, n_items: int = 40, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        liked = rng.choice(n_items, size=8, replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": f"i{item}",
                    "rating": float(rng.integers(3, 6)),
                    "f1": float(item % 5),
                    "f2": float(item // 5),
                    "age": float(18 + (user % 30)),
                }
            )
    return pd.DataFrame(rows)


def _run_method(
    method: str | None = None,
    *,
    feedback: str = "explicit",
    backend: str | None = None,
    **kwargs: object,
) -> dict[str, object]:
    session = (
        Session.ingest(_interaction_frame())
        .set_roles(
            {
                "user_id": "id",
                "item_id": "id",
                "rating": "target",
                "f1": "feature",
                "f2": "feature",
                "age": "feature",
            }
        )
        .split(test_size=0.2, validation_size=0.15, random_state=0)
    )
    fit_kwargs: dict[str, object] = {
        "user_column": "user_id",
        "item_column": "item_id",
        "feedback": feedback,
        "n_factors": 16,
        "random_state": 0,
    }
    if method is not None:
        fit_kwargs["method"] = method
    if backend is not None:
        fit_kwargs["backend"] = backend
    fit_kwargs.update(kwargs)
    fit = session.fit_recommender(**fit_kwargs)
    ev = session.evaluate_recommender(partition="test", k=10)
    return {
        "method": fit.method,
        "backend": fit.backend,
        "feedback": feedback,
        "n_users": fit.n_users,
        "n_items": fit.n_items,
        "metrics": dict(ev.metrics),
        "n_users_scored": ev.n_users_scored,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML recommender ranking quality benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/recommenders/results/ranking_quality.json"),
    )
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    for method in ("item_knn", "svd", "nmf"):
        runs.append(_run_method(method=method))

    runs.append(
        _run_method(
            method="content",
            item_feature_columns=["f1", "f2"],
        )
    )

    if implicit_available():
        for method in ("als", "bpr"):
            runs.append(_run_method(method=method, feedback="implicit"))
        runs.append(_run_method(feedback="implicit"))
    else:
        runs.append(
            {
                "method": "als",
                "backend": "implicit",
                "skipped": True,
                "reason": "implicit not installed",
            }
        )

    if lightfm_available():
        runs.append(
            _run_method(
                method="lightfm",
                item_feature_columns=["f1", "f2"],
                user_feature_columns=["age"],
                lightfm_epochs=5,
            )
        )
    else:
        runs.append(
            {
                "method": "lightfm",
                "backend": "lightfm",
                "skipped": True,
                "reason": "lightfm not installed",
            }
        )

    payload = {
        "capability_matrix": recommender_capability_matrix(),
        "runs": runs,
        "implicit_available": implicit_available(),
        "lightfm_available": lightfm_available(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
