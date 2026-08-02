"""End-to-end Session smoke for symbolic + neuro-symbolic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def test_symbolic_and_neuro_symbolic_alpha_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(220, 3))
    y = ((x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 2]) > 0).astype(int)
    frame = pd.DataFrame(
        {"f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "label": y}
    )
    session = (
        Session.ingest(frame)
        .set_roles(
            {"f0": "feature", "f1": "feature", "f2": "feature", "label": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )

    sym = session.fit_symbolic(source="decision_tree", task="classification")
    assert sym.n_rules >= 1
    pred = session.predict_symbolic(partition="test", return_traces=True)
    assert pred.n_rows > 0
    assert len(pred.traces) == pred.n_rows
    ev = session.evaluate_symbolic(partition="validation")
    assert "accuracy" in ev.metrics

    bundle = tmp_path / "sym_bundle"
    session.save_symbolic_bundle(bundle)

    # Neuro-symbolic hybrid on a fresh session path (same data).
    session2 = (
        Session.ingest(frame)
        .set_roles(
            {"f0": "feature", "f1": "feature", "f2": "feature", "label": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    constraints = [
        {
            "rule_id": "high_f0",
            "if": [{"column": "f0", "op": ">", "value": 1.2}],
            "then": 1,
            "hardness": "hard",
            "kind": "constraint",
            "priority": 50,
        }
    ]
    neuro = session2.fit_neuro_symbolic(
        mode="constraint_overlay",
        base_estimator="logistic_regression",
        task="classification",
        rules=constraints,
        rule_source="declared",
    )
    assert neuro.n_rules >= 1
    npred = session2.predict_neuro_symbolic(partition="test")
    assert npred.neural_predictions is not None
    nev = session2.evaluate_neuro_symbolic(partition="validation")
    assert "accuracy" in nev.metrics

    report = session2.walkthrough()
    assert report.to_dict()["symbolic_status"]["has_neuro_symbolic_plan"] is True

    # Reload pure symbolic bundle and re-score.
    session3 = (
        Session.ingest(frame)
        .set_roles(
            {"f0": "feature", "f1": "feature", "f2": "feature", "label": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )
    session3.load_symbolic_bundle(bundle)
    assert session3.symbolic_plan is not None
    assert "accuracy" in session3.evaluate_symbolic(partition="test").metrics
