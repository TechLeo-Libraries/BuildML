"""Symbolic + neuro-symbolic Session loop (mirrors quickstart-symbolic)."""

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

    fit = session.fit_symbolic(source="decision_tree", task="classification")
    print("symbolic", fit.n_rules, fit.provenance, fit.train_accuracy)

    pred = session.predict_symbolic(partition="test", return_traces=True)
    print("trace0", pred.traces[0].chosen_rule_id, pred.traces[0].fired_rule_ids[:3])

    ev = session.evaluate_symbolic(partition="validation")
    print("symbolic_eval", ev.metrics, "coverage", ev.rule_coverage)

    out = Path("artifacts") / "symbolic_demo_bundle"
    session.save_symbolic_bundle(out)
    print("saved", out)

    constraints = [
        {
            "rule_id": "high_a",
            "if": [{"column": "a", "op": ">", "value": 1.5}],
            "then": 1,
            "hardness": "hard",
            "kind": "constraint",
            "priority": 100,
        }
    ]
    neuro = session.fit_neuro_symbolic(
        mode="constraint_overlay",
        base_estimator="logistic_regression",
        task="classification",
        rules=constraints,
        rule_source="declared",
    )
    print("neuro", neuro.mode, neuro.rule_provenance, neuro.n_rules)
    nev = session.evaluate_neuro_symbolic(partition="test")
    print("neuro_eval", nev.metrics, "repair_rate", nev.repair_rate)


if __name__ == "__main__":
    main()
