"""Case-based reasoning Session loop (mirrors quickstart-cbr)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(220, 2))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})

    session = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )

    fit = session.cbr.fit(
        task="classification",
        metric="euclidean",
        reuse="distance_weighted",
        k=5,
    )
    print("cbr", fit.n_cases, fit.metric, fit.reuse, fit.train_score)

    neighbors = session.cbr.retrieve(partition="test", k=3)
    t0 = neighbors.traces[0]
    print("retrieve0", t0.neighbor_case_ids, t0.distances)

    pred = session.cbr.predict(partition="test", return_traces=True)
    print(
        "predict0",
        pred.traces[0].prediction,
        pred.traces[0].neighbor_solutions,
    )

    ev = session.cbr.evaluate(partition="validation")
    print("eval", ev.metrics, "mean_d", ev.mean_neighbor_distance)

    out = Path("artifacts") / "cbr_demo_bundle"
    session.cbr.save_bundle(out)
    print("saved", out)

    other = (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    other.cbr.load_bundle(out, trusted=True)
    print("reloaded", other.cbr.evaluate(partition="test").metrics)


if __name__ == "__main__":
    main()
