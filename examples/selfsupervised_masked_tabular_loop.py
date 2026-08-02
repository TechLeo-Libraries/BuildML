"""Runnable SSL loop: masked tabular pretext → head → eval → bundle.

Requires a GitHub / editable BuildML 2.x install (core sklearn; no extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal([-1.0, -1.0], 0.7, size=(100, 2))
    x1 = rng.normal([1.5, 1.2], 0.7, size=(100, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 100 + [1] * 100

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, stratify=True, random_state=0)
        .scale(method="standard")
    )

    pre = session.fit_ssl_pretext(
        method="masked_tabular",
        latent_dim=8,
        mask_ratio=0.2,
        max_iter=120,
        random_state=0,
    )
    print("pretext:", pre.method, "latent=", pre.latent_dim, "mae=", round(pre.reconstruction_mae or 0.0, 4))

    head = session.finetune_ssl_head(estimator="logistic_regression", random_state=0)
    print("head:", head.estimator_name, "labeled=", head.n_labeled_train)

    ev = session.evaluate_ssl(partition="test")
    print("eval:", {k: round(v, 4) for k, v in ev.metrics.items()})

    out = Path(".buildml-artifacts") / "ssl_bundle"
    path = session.save_ssl_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
