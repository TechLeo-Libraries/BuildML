"""Online / continual learning example: fit_online → partial_fit → eval → bundle.

Honesty: batch/stream-chunk Session updates via sklearn partial_fit — not a
distributed streaming platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.normal([-1.0, -1.0], 0.55, size=(180, 2))
    x1 = rng.normal([1.2, 1.0], 0.55, size=(180, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 180 + [1] * 180

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )

    fit = session.online.fit(
        estimator="sgd_classifier",
        chunk_size=45,
        n_init=45,
        classes=[0, 1],
    )
    print(
        f"init rows={fit.n_init_rows} remaining={fit.n_remaining_train} "
        f"classes={fit.classes}"
    )

    while True:
        plan = session.online.plan
        assert plan is not None
        remaining = plan.n_train_rows - plan.cursor
        if remaining <= 0:
            break
        update = session.online.partial_fit(n_rows=min(45, remaining))
        print(
            f"update#{update.n_updates} chunk={update.n_chunk_rows} "
            f"seen={update.n_seen_rows} mode={update.update_mode}"
        )

    ev = session.online.evaluate(partition="validation")
    print(f"validation metrics={ev.metrics}")

    out = Path("artifacts") / "online_partial_fit_bundle"
    session.online.save_bundle(out)
    print(f"saved bundle → {out}")


if __name__ == "__main__":
    main()
