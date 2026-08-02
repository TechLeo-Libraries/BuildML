"""Tier A proof: active-labeling-budget."""

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
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from proofs._lib import metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("active-labeling-budget", seed=0)
    rng = np.random.default_rng(ctx.seed)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(160, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(160, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 160 + [1] * 160
    truth = frame["label"].copy()
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=int(0.85 * len(train_idx)), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset, full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    fit = session.fit_active_learner(strategy="margin", batch_size=8, label_budget=32)
    curve = []
    for round_i in range(4):
        q = session.suggest_query(batch_size=8)
        if not q.indices:
            break
        human = [int(truth.loc[i]) for i in q.indices]
        labeled = session.label_rows(indices=q.indices, labels=human)
        curve.append({
            "round": round_i,
            "n_newly_labeled": int(labeled.n_newly_labeled),
            "n_labeled_now": int(labeled.n_labeled_now),
            "budget_remaining": int(labeled.budget_remaining),
        })
    test = session.evaluate_active_learning(partition="test")
    bundle = session.save_active_learning_bundle(ctx.artifacts_dir / "al_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_al_pool", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "fit": {
            "strategy": fit.strategy,
            "n_labeled_train": int(fit.n_labeled_train),
            "n_unlabeled_pool": int(fit.n_unlabeled_pool),
        },
        "label_curve": curve,
        "test_metrics": metrics_round(dict(test.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Queries drawn from train unlabeled pool only",
            "Oracle used only to simulate humans (not library API inventing labels)",
            "Test evaluated after budget loop",
        ],
        "industry_comparison": {"status": "stub"},
        "limitations": ["Simulated oracle; not a real annotation UI"],
    })
    print("active-labeling-budget OK", dict(test.metrics))


if __name__ == "__main__":
    main()
