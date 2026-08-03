"""Online-learning alpha-gate smoke: split → fit_online → partial_fit → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_online_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(150, 2))
    x1 = rng.normal([1.3, 1.1], 0.6, size=(150, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 150 + [1] * 150

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )

    fit = session.fit_online(
        estimator="sgd_classifier",
        chunk_size=50,
        n_init=50,
    )
    assert fit.n_init_rows == 50
    assert session.online_plan is not None

    updates = 0
    while True:
        plan = session.online_plan
        assert plan is not None
        remaining = plan.n_train_rows - plan.cursor
        if remaining <= 0:
            break
        upd = session.partial_fit_online(n_rows=min(50, remaining))
        assert upd.update_mode == "partial_fit"
        updates += 1
        if updates > 20:
            break
    assert updates >= 1

    ev = session.evaluate_online(partition="validation")
    assert ev.partition == "validation"
    assert "accuracy" in ev.metrics
    assert session.online_eval_result is not None

    before = session.explain("partial_fit_online", moment="before")
    assert before.prerequisite_status.get("online-plan") is True

    wt = session.walkthrough()
    assert wt.online_status.get("has_online_plan") is True

    bundle = session.save_online_bundle(tmp_path / "online_bundle")
    restored = Session.ingest(session.to_pandas()).set_roles(
        {"x": "feature", "y": "feature", "label": "target"}
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_online_bundle(bundle, trusted=True)
    assert restored.online_plan is not None
    assert restored.online_plan.n_updates == session.online_plan.n_updates
    again = restored.evaluate_online(partition="test")
    assert "accuracy" in again.metrics
