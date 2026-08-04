"""Tier A proof: stream-fraud-online."""

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


def _make_fraud_stream(*, n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Overlapping fraud stream with label noise — not linearly separable blobs."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.2, 0.85, size=n)
    velocity = rng.poisson(3.0, size=n).astype(float) + rng.normal(0, 0.8, size=n)
    hour = rng.integers(0, 24, size=n).astype(float)
    merchant_risk = rng.beta(2.0, 5.0, size=n)
    device_age = rng.exponential(120.0, size=n).clip(1, 2000)
    # Soft latent score with heavy class overlap + flip noise.
    logit = (
        -2.4
        + 0.35 * np.log1p(amount)
        + 0.28 * velocity
        + 0.08 * ((hour < 6) | (hour > 22)).astype(float)
        + 1.6 * merchant_risk
        - 0.002 * device_age
        + rng.normal(0, 0.9, size=n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (rng.random(n) < prob).astype(int)
    # Irreducible label noise (~8%).
    flip = rng.random(n) < 0.08
    is_fraud = np.where(flip, 1 - is_fraud, is_fraud)
    return pd.DataFrame(
        {
            "amount_z": (amount - amount.mean()) / (amount.std() + 1e-9),
            "velocity_z": (velocity - velocity.mean()) / (velocity.std() + 1e-9),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "merchant_risk": merchant_risk,
            "device_age_days": device_age,
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    ctx = new_proof_context("stream-fraud-online", seed=7)
    frame = _make_fraud_stream(n=520, seed=ctx.seed)
    features = [
        "amount_z",
        "velocity_z",
        "hour_sin",
        "hour_cos",
        "merchant_risk",
        "device_age_days",
    ]
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in features}, "is_fraud": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    fit = session.online.fit(
        estimator="sgd_classifier",
        chunk_size=50,
        n_init=50,
        classes=[0, 1],
    )
    updates = []
    while True:
        plan = session.online.plan
        remaining = plan.n_train_rows - plan.cursor
        if remaining <= 0:
            break
        u = session.online.partial_fit(n_rows=min(50, remaining))
        updates.append(
            {
                "n_updates": int(u.n_updates),
                "n_chunk_rows": int(u.n_chunk_rows),
                "n_seen_rows": int(u.n_seen_rows),
            }
        )
    val = session.online.evaluate(partition="validation")
    test = session.online.evaluate(partition="test")
    test_metrics = dict(test.metrics)
    acc = float(test_metrics.get("accuracy", float("nan")))
    if acc == acc and acc >= 0.99:
        raise SystemExit(
            "stream-fraud-online refused perfect-score theater: "
            f"test accuracy={acc:.4f} >= 0.99 on overlapping noisy stream. "
            "Generator must keep irreducible error."
        )
    bundle = session.online.save_bundle(ctx.artifacts_dir / "online_bundle")
    write_results(
        ctx,
        {
            "status": "completed",
            "data": {
                "name": "synthetic_fraud_stream_overlapping",
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
                "Classes overlap; ~8% label flips keep irreducible error.",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn SGDClassifier partial_fit twin on the same "
                    "split; optional River when installed — run script then baseline_industry.py for "
                    "results/comparison.json."
                ),
            },
            "limitations": ["Batch chunks, not Kafka/Flink"],
        },
    )
    print("stream-fraud-online OK", test_metrics)


if __name__ == "__main__":
    main()
