"""Raw class labels survive real rule/neural consumers and trusted bundles."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.symbolic.features import encode_classification_targets


def _session(labels):
    x = np.linspace(-2, 2, 80)
    frame = pd.DataFrame({"x": x, "y": [labels[int(t > 0)] for t in x]})
    return Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )


@pytest.mark.parametrize("labels", [("001", "002"), ("1.0", "2.0"), (-3, 7), (1.25, 2.75), (False, True)])
@pytest.mark.parametrize("path", ["decision_tree", "decision_list", "constraint_overlay", "rules_as_features", "constraint_repair"])
def test_raw_predictions_metadata_scores_and_bundle(labels, path, tmp_path):
    session = _session(labels)
    original = session.dataset.frame.copy(deep=True)
    train = original.iloc[list(session.split_plan.train_indices)]
    test = original.iloc[list(session.split_plan.test_indices)]
    encoder = LabelEncoder().fit(train.y.to_numpy())
    codes = encoder.transform(train.y.to_numpy())
    if path in {"decision_tree", "decision_list"}:
        session.symbolic.fit(source=path, task="classification", max_depth=2, min_samples_leaf=2)
        pred = session.symbolic.predict(partition="test")
        scored = session.symbolic.evaluate(partition="test")
        plan = session.symbolic_plan
        if path == "decision_tree":
            oracle = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=0).fit(train[["x"]], codes)
            expected = encoder.inverse_transform(oracle.predict(test[["x"]])).tolist()
        else:
            # A single zero boundary perfectly separates this ordered dataset.
            expected = test.y.tolist()
    else:
        session.symbolic.fit_neuro(mode=path, task="classification", rules=[{
            "if": [{"column": "x", "op": ">", "value": 1000}], "then": labels[0]
        }],
            rule_source="declared", base_estimator="logistic_regression")
        pred = session.symbolic.predict_neuro(partition="test")
        scored = session.symbolic.evaluate_neuro(partition="test")
        plan = session.neuro_symbolic_plan
        a, b = train[["x"]].to_numpy(), test[["x"]].to_numpy()
        if path == "rules_as_features":
            a, b = np.column_stack([a, np.zeros(len(a))]), np.column_stack([b, np.zeros(len(b))])
        oracle = LogisticRegression(max_iter=500, random_state=0).fit(a, codes)
        expected = encoder.inverse_transform(oracle.predict(b)).tolist()
        assert list(pred.neural_predictions) == expected
    assert list(pred.predictions) == expected
    assert [type(v) for v in pred.predictions] == [type(v) for v in expected]
    assert scored.metrics["accuracy"] == sum(a == b for a, b in zip(test.y, expected, strict=True)) / len(test)
    assert list(plan.classes_) == encoder.classes_.tolist()
    json.dumps(plan.to_dict())
    out = tmp_path / "bundle"
    session.symbolic.save_bundle(out)
    other = _session(labels)
    other.symbolic.load_bundle(out, trusted=True)
    restored = (other.symbolic.predict if path in {"decision_tree", "decision_list"}
                else other.symbolic.predict_neuro)(partition="test")
    assert restored.predictions == pred.predictions
    assert [type(v) for v in restored.predictions] == [type(v) for v in pred.predictions]
    pd.testing.assert_frame_equal(session.dataset.frame, original)


@pytest.mark.parametrize("labels,classes", [([1, "1", 2], None), (["001", "002"], [1, 2])])
def test_incompatible_labels_are_not_coerced(labels, classes):
    with pytest.raises(ValidationError, match="label|vocabulary"):
        encode_classification_targets(pd.Series(labels), classes=classes)


@pytest.mark.parametrize("labels", [("001", "002"), (False, True), (1.25, 2.75)])
def test_explicit_raw_vocabulary(labels):
    codes, encoder, classes = encode_classification_targets(pd.Series(labels[::-1]), classes=labels)
    assert codes.tolist() == [1, 0]
    assert classes == labels
    assert encoder.inverse_transform(codes).tolist() == list(labels[::-1])


@pytest.mark.parametrize("labels,expected_types", [(("001", "002"), int), (("1.0", "2.0"), float), (("cat", "dog"), str)])
def test_legacy_bundle_keeps_decoding_with_refit_notice(labels, expected_types, tmp_path):
    session = _session(labels)
    session.symbolic.fit_neuro(task="classification", rules=[{
        "if": [{"column": "x", "op": ">", "value": 1000}], "then": labels[0]
    }], rule_source="declared", base_estimator="logistic_regression")
    plan = session.neuro_symbolic_plan
    # Actual old encoder representation; class order and estimator weights agree.
    plan.label_encoder_ = LabelEncoder().fit(np.asarray(labels))
    plan.classes_ = tuple(labels)
    plan.config.pop("raw_class_labels", None)
    before = session.symbolic.predict_neuro(partition="test")
    assert all(type(v) is expected_types for v in before.predictions)
    session.symbolic.save_bundle(tmp_path / "legacy")
    other = _session(labels)
    other.symbolic.load_bundle(tmp_path / "legacy", trusted=True)
    after = other.symbolic.predict_neuro(partition="test")
    assert after.predictions == before.predictions
    assert any("refit" in warning.lower() for warning in after.warnings)
    assert any("refit" in warning.lower() for warning in other.symbolic.evaluate_neuro(partition="test").warnings)


@pytest.mark.parametrize("path", ["symbolic", "constraint_overlay", "constraint_repair"])
def test_string_rules_cannot_impersonate_numeric_labels_in_metrics(path):
    session = _session((1.0, 2.0))
    rules = [{"if": [{"column": "x", "op": op, "value": 0}],
              "then": label, "kind": "constraint", "hardness": "hard"}
             for op, label in [("<=", "1.0"), (">", "2.0")]]
    if path == "symbolic":
        session.symbolic.fit(source="declared", task="classification", rules=rules)
        scored = session.symbolic.evaluate(partition="test")
    else:
        session.symbolic.fit_neuro(mode=path, task="classification", rules=rules,
            rule_source="declared", base_estimator="logistic_regression")
        scored = session.symbolic.evaluate_neuro(partition="test")
        assert scored.metrics["neural_final_agreement"] == 0
        assert scored.repair_rate == 1
    assert scored.metrics["accuracy"] == 0
