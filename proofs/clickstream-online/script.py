"""Tier A proof: clickstream-online — streaming conversion with partial_fit."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from proofs._lib import metrics_round, new_proof_context, write_results


def _make_clickstream(*, n: int = 480, seed: int = 24) -> pd.DataFrame:
    """Overlapping conversion stream — not two well-separated Gaussians."""
    rng = np.random.default_rng(seed)
    pages = rng.poisson(4.0, size=n).astype(float) + rng.normal(0, 0.7, size=n)
    dwell = rng.lognormal(2.5, 0.7, size=n)
    cart_adds = rng.poisson(0.8, size=n).astype(float)
    referral = rng.choice([0.0, 1.0], size=n, p=[0.65, 0.35])
    logit = (
        -1.8
        + 0.22 * pages
        + 0.35 * np.log1p(dwell)
        + 0.55 * cart_adds
        + 0.4 * referral
        + rng.normal(0, 0.85, size=n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    converted = (rng.random(n) < prob).astype(int)
    flip = rng.random(n) < 0.07
    converted = np.where(flip, 1 - converted, converted)
    return pd.DataFrame(
        {
            "pages_z": (pages - pages.mean()) / (pages.std() + 1e-9),
            "dwell_z": (dwell - dwell.mean()) / (dwell.std() + 1e-9),
            "cart_adds_z": (cart_adds - cart_adds.mean()) / (cart_adds.std() + 1e-9),
            "referral": referral,
            "converted": converted,
        }
    )


def main() -> None:
    ctx = new_proof_context("clickstream-online", seed=24)
    frame = _make_clickstream(n=500, seed=ctx.seed)
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "pages_z": "feature",
                "dwell_z": "feature",
                "cart_adds_z": "feature",
                "referral": "feature",
                "converted": "target",
            }
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.fit_online(
        estimator="sgd_classifier",
        chunk_size=40,
        n_init=40,
        classes=[0, 1],
    )
    updates = []
    while True:
        plan = session.online_plan
        remaining = plan.n_train_rows - plan.cursor
        if remaining <= 0:
            break
        u = session.partial_fit_online(n_rows=min(40, remaining))
        updates.append(
            {
                "n_updates": int(u.n_updates),
                "n_chunk_rows": int(u.n_chunk_rows),
                "n_seen_rows": int(u.n_seen_rows),
            }
        )
    val = session.evaluate_online(partition="validation")
    test = session.evaluate_online(partition="test")
    test_metrics = dict(test.metrics)
    acc = float(test_metrics.get("accuracy", float("nan")))
    if acc == acc and acc >= 0.99:
        raise SystemExit(
            "clickstream-online refused perfect-score theater: "
            f"test accuracy={acc:.4f} >= 0.99 on overlapping noisy stream."
        )
    bundle = session.save_online_bundle(ctx.artifacts_dir / "online_bundle")
    write_results(
        ctx,
        {
            "status": "completed",
            "data": {
                "name": "synthetic_clickstream_overlapping",
                "license": "synthetic/public-domain",
                "n_rows": int(len(frame)),
                "difficulty": "overlapping_classes_with_label_noise",
            },
            "fit": {
                "n_init_rows": int(fit.n_init_rows),
                "n_remaining_train": int(fit.n_remaining_train),
                "classes": list(fit.classes),
            },
            "updates": updates,
            "validation_metrics": metrics_round(dict(val.metrics)),
            "test_metrics": metrics_round(test_metrics),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "partial_fit consumes train cursor only",
                "Validation/test never enter online updates",
            ],
            "honesty": [
                "Refuses test accuracy >= 0.99 (anti perfect-score theater).",
            ],
            "limitations": ["Batch chunks, not Kafka/Flink"],
        },
    )
    print("clickstream-online OK", test_metrics)


if __name__ == "__main__":
    main()
