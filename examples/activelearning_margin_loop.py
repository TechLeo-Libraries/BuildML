"""Runnable active-learning loop: seed labels → query → human labels → eval → bundle.

Requires a GitHub / editable BuildML 2.x install (core sklearn; no extra).

The simulated oracle below is for the example only — library core never invents
labels. Production code must supply human annotations to ``label_rows``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


def main() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(140, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(140, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 140 + [1] * 140
    truth = frame["label"].copy()  # example-only oracle

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )

    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=int(0.85 * len(train_idx)), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )

    fit = session.fit_active_learner(
        strategy="margin",
        batch_size=8,
        label_budget=24,
    )
    print(
        "fit:",
        fit.strategy,
        "labeled=",
        fit.n_labeled_train,
        "pool=",
        fit.n_unlabeled_pool,
    )

    for round_i in range(3):
        q = session.suggest_query(batch_size=8)
        if not q.indices:
            print("round", round_i, "empty query (budget or pool exhausted)")
            break
        human_labels = [int(truth.loc[i]) for i in q.indices]
        labeled = session.label_rows(indices=q.indices, labels=human_labels)
        print(
            "round",
            round_i,
            "newly=",
            labeled.n_newly_labeled,
            "labeled_now=",
            labeled.n_labeled_now,
            "budget_remaining=",
            labeled.budget_remaining,
        )

    ev = session.evaluate_active_learning(partition="test")
    print("eval:", {k: round(v, 4) for k, v in ev.metrics.items()})

    out = Path(".buildml-artifacts") / "activelearning_bundle"
    path = session.save_active_learning_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
