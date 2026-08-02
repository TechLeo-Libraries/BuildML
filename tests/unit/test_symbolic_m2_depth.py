"""M2 depth coverage for symbolic / neuro-symbolic low-level APIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.symbolic.evaluate import evaluate_neuro_symbolic, evaluate_symbolic
from buildml.symbolic.fit import fit_neuro_symbolic, fit_symbolic
from buildml.symbolic.induce import induce_decision_list, induce_decision_tree_rules
from buildml.symbolic.predict import predict_neuro_symbolic, predict_symbolic
from buildml.symbolic.rules import Predicate, Rule, fire_rules, parse_declared_rules


def _clf_session() -> Session:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(160, 2))
    y = (x[:, 0] > 0).astype(int)
    frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": y})
    return (
        Session.ingest(frame)
        .set_roles({"a": "feature", "b": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
        .scale(method="standard")
    )


def test_fire_rules_decision_list_order() -> None:
    kb = parse_declared_rules(
        [
            {
                "rule_id": "high",
                "if": [{"column": "a", "op": ">", "value": 0}],
                "then": "pos",
                "priority": 10,
            },
            {
                "rule_id": "low",
                "if": [{"column": "a", "op": ">", "value": -10}],
                "then": "neg",
                "priority": 1,
            },
        ],
        default_consequent="default",
    )
    frame = pd.DataFrame({"a": [1.0, -1.0]})
    preds, traces, fire = fire_rules(frame, kb)
    assert preds[0] == "pos"
    assert traces[0].chosen_rule_id == "high"
    assert preds[1] == "neg"
    assert fire.shape == (2, 2)


def test_tree_and_list_induction_train_only() -> None:
    session = _clf_session()
    train = session.dataset._ensure_pandas().loc[list(session.split_plan.train_indices)]
    y = train["y"].to_numpy()
    kb, tree = induce_decision_tree_rules(
        train,
        ["a", "b"],
        y,
        task="classification",
        class_names=("0", "1"),
        max_depth=3,
        max_rules=16,
    )
    assert kb.provenance == "induced_tree"
    assert len(kb.rules) >= 1
    assert tree is not None

    kb2 = induce_decision_list(
        train,
        ["a", "b"],
        y,
        task="classification",
        class_names=("0", "1"),
        max_depth=2,
        max_rules=8,
    )
    assert kb2.provenance == "induced_list"


def test_fit_symbolic_refuses_declared_without_rules() -> None:
    session = _clf_session()
    with pytest.raises(ValidationError, match="declared"):
        fit_symbolic(
            session.dataset,
            session.split_plan,
            source="declared",
            task="classification",
        )


def test_constraint_repair_overrides_predictions() -> None:
    session = _clf_session()
    rules = [
        {
            "rule_id": "always_one",
            "if": [],
            "then": 1,
            "hardness": "hard",
            "kind": "constraint",
            "priority": 100,
        }
    ]
    plan, _ = fit_neuro_symbolic(
        session.dataset,
        session.split_plan,
        mode="constraint_repair",
        base_estimator="logistic_regression",
        task="classification",
        rules=rules,
        rule_source="declared",
        reduce_plan=session._reduce_plan,
    )
    pred = predict_neuro_symbolic(
        session.dataset,
        plan,
        session.split_plan,
        partition="test",
        return_traces=True,
    )
    assert all(str(p) == "1" for p in pred.predictions)
    assert any(t.chosen_rule_id == "always_one" for t in pred.traces)


def test_evaluate_symbolic_holdout_metrics() -> None:
    session = _clf_session()
    plan, fit = fit_symbolic(
        session.dataset,
        session.split_plan,
        source="decision_tree",
        task="classification",
        reduce_plan=session._reduce_plan,
    )
    assert fit.n_train_rows == len(session.split_plan.train_indices)
    train_set = set(session.split_plan.train_indices)
    val_set = set(session.split_plan.validation_indices)
    assert train_set.isdisjoint(val_set)

    ev = evaluate_symbolic(
        session.dataset, plan, session.split_plan, partition="validation"
    )
    assert ev.n_rows == len(session.split_plan.validation_indices)
    assert "accuracy" in ev.metrics

    pred = predict_symbolic(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert pred.n_rows == len(session.split_plan.test_indices)


def test_rules_as_features_mode_scores() -> None:
    session = _clf_session()
    plan, fit = fit_neuro_symbolic(
        session.dataset,
        session.split_plan,
        mode="rules_as_features",
        rule_source="decision_tree",
        task="classification",
        reduce_plan=session._reduce_plan,
    )
    assert fit.n_rules >= 1
    assert plan.rule_feature_names_
    ev = evaluate_neuro_symbolic(
        session.dataset, plan, session.split_plan, partition="test"
    )
    assert "accuracy" in ev.metrics
    assert "neural_final_agreement" in ev.metrics


def test_predicate_roundtrip() -> None:
    rule = Rule(
        rule_id="r",
        antecedents=(Predicate(column="a", op="isna"),),
        consequent=0,
    )
    assert rule.to_dict()["antecedents"][0]["op"] == "isna"
