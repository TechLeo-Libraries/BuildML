"""Runnable unsupervised clustering loop (core BuildML — no extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal([0.0, 0.0], 0.35, size=(60, 2))
    b = rng.normal([2.8, 2.8], 0.35, size=(60, 2))
    frame = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    frame["segment"] = [0] * 60 + [1] * 60

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "segment": "ignore"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
        .reduce_dimensions(method="pca", n_components=2, prefix="pc")
    )
    fit = session.fit_clusters(method="kmeans", n_clusters=2, random_state=0)
    metrics = session.evaluate_clusters(
        partition="validation",
        external_label_column="segment",
    )
    print("fit:", fit.to_dict())
    print("eval:", metrics.to_dict())

    out = Path(".buildml-artifacts") / "unsupervised_bundle"
    path = session.save_unsupervised_bundle(out)
    print("bundle:", path)


if __name__ == "__main__":
    main()
