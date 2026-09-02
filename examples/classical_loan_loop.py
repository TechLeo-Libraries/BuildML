"""Mirror of guides/classical-end-to-end.md — loan approval loop.

The guide snippet uses a 12-row table you can read. This file draws 120
rows so the printed metrics are not three-row noise. It is still a toy,
not a credit model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session


def _toy_loans(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 11, size=n).clip(18, 75)
    age[rng.random(n) < 0.08] = np.nan
    income = rng.normal(62, 16, size=n).clip(25, 140)
    region = rng.choice(["N", "S", "W"], size=n)
    logits = 0.04 * (np.nan_to_num(age, nan=38) - 30) + 0.03 * (income - 50)
    logits += np.where(region == "W", 0.2, 0.0)
    approved = (logits + rng.normal(0, 0.7, size=n) > 0).astype(int)
    return pd.DataFrame(
        {"age": age, "income": income, "region": region, "approved": approved}
    )


def main() -> None:
    frame = _toy_loans()
    print(f"toy table n={len(frame)}; read the loop, not a credit score")

    session = Session.ingest(frame)
    session.set_roles(
        {
            "age": "feature",
            "income": "feature",
            "region": "feature",
            "approved": "target",
        }
    )
    session.split(
        test_size=0.25,
        validation_size=0.25,
        stratify=True,
        random_state=42,
    )
    session.impute(strategy="median")
    session.encode(method="onehot")
    session.scale(method="standard")
    session.fit(LogisticRegression(max_iter=500), task="classification")

    session.calibration(partition="validation")
    session.tune_threshold(partition="validation", fp_cost=1.0, fn_cost=5.0)
    val = session.evaluate(partition="validation")
    test = session.evaluate(partition="test")
    print("validation:", val.metrics)
    print("test:", test.metrics)


if __name__ == "__main__":
    main()
