"""Synthetic-data loop: fit Gaussian copula → sample → evaluate → bundle.

Run from repo root::

    python examples/synthetic_copula_loop.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from buildml import Session


def main() -> None:
    x, y = make_classification(
        n_samples=360,
        n_features=6,
        n_informative=4,
        weights=[0.65, 0.35],
        random_state=0,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["grp"] = pd.Series(y).map({0: "low", 1: "high"})

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in frame.columns if c.startswith("f")},
                "grp": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, random_state=0)
    )

    fit = session.synthetic.fit(method="gaussian_copula", random_state=0)
    print("fit:", fit.to_dict())

    # Bootstrap path (plain + smoothed)
    session.synthetic.fit(method="bootstrap", smooth_sigma=0.05, random_state=1)
    boot = session.synthetic.sample(n=80, random_state=2)
    print("bootstrap sample n=", boot.n_rows)

    # Restore copula for fidelity / TSTR
    session.synthetic.fit(method="gaussian_copula", random_state=0)
    sample = session.synthetic.sample(n=120, random_state=3)
    assert sample.frame is not None
    print("copula sample head:\n", sample.frame.head())

    fid = session.synthetic.evaluate(mode="fidelity", partition="test")
    print("fidelity:", fid.metrics)

    tstr = session.synthetic.evaluate(mode="tstr", partition="test")
    print("tstr:", tstr.metrics)

    out = Path("artifacts/synthetic_demo_bundle")
    session.synthetic.save_bundle(out)
    print("bundle:", out.resolve())

    # Walkthrough disclosure
    walk = session.walkthrough()
    print("synthetic_status enabled:", walk.synthetic_status.get("enabled"))


if __name__ == "__main__":
    main()
