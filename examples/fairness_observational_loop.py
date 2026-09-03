"""Mirror of guides/quickstart-fairness.md — observational group rates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(0)
    n = 400
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    x = rng.normal(size=n)
    logits = x + np.where(group == "B", -0.7, 0.0)
    y = np.where(logits > 0, "approved", "denied")
    frame = pd.DataFrame({"x": x, "group": group, "decision": y})

    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "group": "ignore", "decision": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
        .fit(LogisticRegression(max_iter=500), task="classification")
    )

    print(session.fairness.capability_matrix()["non_goals"][:2])

    report = session.fairness.evaluate(
        sensitive_column="group",
        partition="test",
        positive_label="approved",
        bootstrap_samples=50,
    )
    print(report.demographic_parity_difference)
    print(report.selection_rate_by_group)
    print(report.classical_metrics_by_group["A"]["f1"])
    print(report.to_markdown().splitlines()[0])

    session.evaluate(partition="test")
    attached = session.fairness.attach_to_last_eval(
        sensitive_column="group",
        positive_label="approved",
    )
    print("attached dp", attached.demographic_parity_difference)

    # Same rows, split by household so no household lands in two partitions.
    household = np.repeat(np.arange(n // 8), 8)[:n]
    grouped_frame = frame.copy()
    grouped_frame["household"] = household
    grouped = (
        Session.ingest(grouped_frame)
        .set_roles(
            {
                "x": "feature",
                "group": "ignore",
                "household": "group",
                "decision": "target",
            }
        )
        .group_split(test_size=0.25, validation_size=0.2, random_state=0)
        .fit(LogisticRegression(max_iter=500), task="classification")
    )
    grouped_report = grouped.fairness.evaluate(
        sensitive_column="group",
        partition="test",
        positive_label="approved",
        bootstrap_samples=50,
    )
    print("group_split dp", grouped_report.demographic_parity_difference)


if __name__ == "__main__":
    main()
