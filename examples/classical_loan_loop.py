"""Mirror of guides/classical-end-to-end.md — loan approval loop."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session


def main() -> None:
    frame = pd.DataFrame(
        {
            "age": [21, None, 35, 40, 29, 33, 52, 47, 31, None, 44, 38],
            "income": [40, 55, 60, 80, 50, 70, 90, 65, 48, 72, 88, 61],
            "region": ["N", "S", "N", "W", "S", "N", "W", "S", "N", "S", "W", "N"],
            "approved": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        }
    )

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

    val = session.evaluate(partition="validation")
    test = session.evaluate(partition="test")
    print("validation:", val.metrics)
    print("test:", test.metrics)


if __name__ == "__main__":
    main()
