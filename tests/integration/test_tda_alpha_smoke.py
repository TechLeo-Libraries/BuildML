"""End-to-end Session smoke for TDA (+ skip without buildml[tda])."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.tda.extras import tda_available


@pytest.mark.skipif(not tda_available(), reason="buildml[tda] / ripser+persim missing")
def test_tda_end_to_end_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(110, 4))
    b = rng.normal(size=(110, 4)) * 1.7 + np.array([2.8, 0, 0, 0])
    x = np.vstack([a, b])
    y = np.array([0] * 110 + [1] * 110)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["y"] = y

    session = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    fit = session.fit_tda(
        vectorization="silhouette",
        knn=10,
        n_bins=10,
        head="logistic_regression",
        random_state=0,
    )
    assert fit.n_train_rows > 0
    assert session.transform_tda(partition="test").n_rows > 0
    ev = session.evaluate_tda(partition="validation")
    assert "accuracy" in ev.metrics
    session.save_tda_bundle(tmp_path / "tda")

    wt = session.walkthrough()
    assert wt.tda_status.get("has_tda_plan") is True

    other = (
        Session.ingest(frame)
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    other.load_tda_bundle(tmp_path / "tda", trusted=True)
    assert other.evaluate_tda(partition="test").n_rows > 0
