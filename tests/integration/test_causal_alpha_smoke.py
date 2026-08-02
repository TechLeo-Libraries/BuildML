"""Alpha smoke: Session causal path end-to-end."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buildml import Session


def test_causal_session_smoke(tmp_path) -> None:
    rng = np.random.default_rng(9)
    n = 300
    w = rng.normal(size=(n, 2))
    logit = 0.7 * w[:, 0] - 0.4 * w[:, 1]
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    y = 1.4 * t + 0.5 * w[:, 0] + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"x1": w[:, 0], "x2": w[:, 1], "t": t, "y": y})

    session = (
        Session.ingest(frame)
        .set_roles(
            {"x1": "feature", "x2": "feature", "t": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    session.declare_causal_assumptions(
        treatment="t",
        outcome="y",
        confounders=["x1", "x2"],
        acknowledge_unconfoundedness=True,
        acknowledge_positivity=True,
    )
    fit = session.fit_causal(method="aipw", bootstrap_samples=25)
    assert np.isfinite(fit.ate)
    ev = session.evaluate_causal(partition="validation", bootstrap_samples=10)
    assert ev.n_rows > 0
    session.refute_causal(kind="random_confounder")
    path = tmp_path / "bundle"
    session.save_causal_bundle(path)
    assert (path / "meta.json").is_file()
    walk = session.walkthrough()
    assert walk.causal_status.get("has_causal_plan") is True
