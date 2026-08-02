"""Self-supervised alpha-gate smoke: split → pretext → head → eval → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session


def test_selfsupervised_alpha_gate_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    x0 = rng.normal([-1.0, -1.0], 0.7, size=(110, 2))
    x1 = rng.normal([1.5, 1.2], 0.7, size=(110, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 110 + [1] * 110

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )

    pre = session.fit_ssl_pretext(
        method="masked_tabular",
        latent_dim=8,
        mask_ratio=0.2,
        max_iter=100,
        random_state=0,
    )
    assert pre.method == "masked_tabular"
    assert session.ssl_plan is not None

    head = session.finetune_ssl_head(estimator="logistic_regression", random_state=0)
    assert head.n_labeled_train > 0
    assert session.ssl_head_plan is not None

    ev = session.evaluate_ssl(partition="validation")
    assert ev.partition == "validation"
    assert "accuracy" in ev.metrics
    assert session.ssl_eval_result is not None

    before = session.explain("fit_ssl_pretext", moment="before")
    assert before.prerequisite_status.get("split") is True

    bundle = session.save_ssl_bundle(tmp_path / "ssl_bundle")
    restored = (
        Session.ingest(session.to_pandas())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    restored.load_ssl_bundle(bundle)
    again = restored.evaluate_ssl(partition="validation")
    assert again.metrics["accuracy"] == pytest.approx(ev.metrics["accuracy"])
