"""End-to-end smoke for case-based reasoning Session path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_cbr_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(180, 2))
    y = (x[:, 0] - 0.2 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame({"f1": x[:, 0], "f2": x[:, 1], "label": y})

    session = (
        Session.ingest(frame)
        .set_roles({"f1": "feature", "f2": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=7, stratify=True)
        .scale(method="standard")
    )
    fit = session.fit_cbr(
        task="classification",
        metric="manhattan",
        reuse="majority",
        k=7,
    )
    assert fit.n_cases == session.cbr_plan.case_base.n_cases

    pred = session.predict_cbr(partition="test", return_traces=True)
    assert pred.n_rows > 0
    assert all(t.reuse_mode == "majority" for t in pred.traces)

    ev = session.evaluate_cbr(partition="validation")
    assert 0.0 <= ev.metrics["accuracy"] <= 1.0

    bundle = tmp_path / "cbr_alpha"
    session.save_cbr_bundle(bundle)

    report = session.walkthrough()
    assert report.cbr_status["enabled"] is True
    assert "case" in report.cbr_status["boundary"].lower() or report.cbr_status[
        "has_cbr_plan"
    ]

    other = (
        Session.ingest(frame)
        .set_roles({"f1": "feature", "f2": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=7, stratify=True)
        .scale(method="standard")
    )
    other.load_cbr_bundle(bundle, trusted=True)
    assert other.cbr_plan is not None
    assert other.evaluate_cbr(partition="test").n_rows > 0
