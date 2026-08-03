"""Semi-supervised alpha-gate smoke: split → mask train → fit → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


def test_semisupervised_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(130, 2))
    x1 = rng.normal([1.3, 1.1], 0.6, size=(130, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 130 + [1] * 130

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    blank = rng.choice(train_idx, size=int(0.65 * len(train_idx)), replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )

    fit = session.fit_semisupervised(method="label_propagation", n_neighbors=7)
    assert fit.n_unlabeled_train > 0
    assert session.semisupervised_plan is not None

    ev = session.evaluate_semisupervised(partition="validation")
    assert ev.partition == "validation"
    assert "accuracy" in ev.metrics
    assert session.semisupervised_eval_result is not None

    before = session.explain("fit_semisupervised", moment="before")
    assert before.prerequisite_status.get("split") is True

    first = session.predict_semisupervised(partition="validation")
    bundle = session.save_semisupervised_bundle(tmp_path / "semi_bundle")
    # Restore into a Session that keeps the same masked frame + SplitPlan
    # (stratify cannot run on NaN targets).
    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_semisupervised_bundle(bundle, trusted=True)
    again = restored.predict_semisupervised(partition="validation")
    assert again.predictions == first.predictions
