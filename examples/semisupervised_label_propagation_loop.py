"""Runnable semi-supervised loop: scarce train labels → fit → eval → bundle.

Requires a GitHub / editable BuildML 2.x install (core sklearn; no extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


def _mask_train_labels(session: Session, fraction: float = 0.7, seed: int = 0) -> Session:
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    n_blank = max(1, int(fraction * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def main() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(120, 2))
    x1 = rng.normal([1.2, 1.0], 0.6, size=(120, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 120 + [1] * 120

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session = _mask_train_labels(session, fraction=0.7, seed=0)

    fit = session.semisupervised.fit(method="label_propagation", n_neighbors=7)
    print(
        "fit:",
        fit.method,
        "labeled=",
        fit.n_labeled_train,
        "unlabeled=",
        fit.n_unlabeled_train,
    )

    ev = session.semisupervised.evaluate(partition="test")
    print("eval:", {k: round(v, 4) for k, v in ev.metrics.items()})

    out = Path(".buildml-artifacts") / "semisupervised_bundle"
    path = session.semisupervised.save_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
