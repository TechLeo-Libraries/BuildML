"""Human label rounds retain class identities and reproduce fresh model fits."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from buildml import Session
from buildml.activelearning.features import decode_predictions, encode_labeled_targets
from buildml.core.errors import ValidationError


def _session(labels):
    x = np.linspace(-3, 3, 90)
    truth = [labels[int(value > 0)] for value in x]
    frame = pd.DataFrame({"x": x, "label": pd.Series(truth, dtype=object)})
    session = Session.ingest(frame).set_roles({"x": "feature", "label": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    # Construct the missing train labels before ingest; no private Session mutation.
    pool = list(session.split_plan.train_indices)[::3]
    frame.loc[pool, "label"] = None
    session = Session.ingest(frame).set_roles({"x": "feature", "label": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    session.active_learning.fit(base_estimator="logistic_regression", label_budget=10, random_state=0)
    return session, truth


def _check_model(session):
    frame = session.dataset.frame
    train = frame.iloc[list(session.split_plan.train_indices)].dropna(subset=["label"])
    test = frame.iloc[list(session.split_plan.test_indices)]
    encoder = LabelEncoder().fit(train.label.to_numpy())
    oracle = LogisticRegression(max_iter=500, random_state=0).fit(train[["x"]].to_numpy(), encoder.transform(train.label.to_numpy()))
    plan = session.activelearning_plan
    actual = decode_predictions(plan.estimator_.predict(test[["x"]].to_numpy()), plan.label_encoder_)
    expected = encoder.inverse_transform(oracle.predict(test[["x"]].to_numpy())).tolist()
    assert actual == expected
    assert [type(v) for v in actual] == [type(v) for v in expected]
    assert plan.classes_ == tuple(encoder.classes_.tolist())
    json.dumps(plan.to_dict())
    score = session.active_learning.evaluate(partition="test")
    assert score.metrics["accuracy"] == sum(a == b for a, b in zip(test.label, expected, strict=True)) / len(test)


@pytest.mark.parametrize("labels", [("001", "002"), ("1.0", "2.0"), (-3, 7), (1.25, 2.75), (False, True)])
def test_raw_labels_rounds_reload_and_fresh_fit(labels, tmp_path):
    session, truth = _session(labels)
    _check_model(session)
    holdouts = {p: session.dataset.frame.iloc[list(getattr(session.split_plan, f"{p}_indices"))].copy()
                for p in ("validation", "test")}
    for round_index in range(2):
        query = session.active_learning.suggest_query(batch_size=2)
        session.active_learning.label_rows(indices=query.indices, labels=[truth[i] for i in query.indices])
        assert session.activelearning_plan.n_queries_used == 2 * (round_index + 1)
        _check_model(session)
        if round_index == 0:
            session.active_learning.save_bundle(tmp_path / "bundle")
            other = Session.ingest(session.dataset.frame.copy()).set_roles({"x": "feature", "label": "target"}).split(
                test_size=0.2, validation_size=0.2, random_state=13
            )
            other.active_learning.load_bundle(tmp_path / "bundle", trusted=True)
            session = other
    for partition, expected in holdouts.items():
        pd.testing.assert_frame_equal(session.dataset.frame.iloc[list(getattr(session.split_plan, f"{partition}_indices"))], expected)


def test_mixed_raw_labels_are_not_merged_by_stringification():
    with pytest.raises(ValidationError, match="label"):
        encode_labeled_targets(pd.Series([1, "1", 2], dtype=object))


@pytest.mark.parametrize("labels", [("001", "002"), (False, True), (1.25, 2.75)])
def test_new_class_round_reencodes_all_labeled_rows(labels):
    values = pd.Series([*labels, None], dtype=object)
    _, encoder, _, _, _, _ = encode_labeled_targets(values)
    third = {str: "003", bool: None, float: 3.5}[type(labels[0])]
    if third is None:
        third = labels[0]
    updated = pd.Series([*labels, third], dtype=object)
    codes, expanded, classes, mask, count, pool = encode_labeled_targets(updated, label_encoder=encoder)
    oracle = LabelEncoder().fit(updated.to_numpy())
    assert codes.tolist() == oracle.transform(updated.to_numpy()).tolist()
    assert classes == tuple(oracle.classes_.tolist())
    assert decode_predictions(codes, expanded) == updated.tolist()
    assert mask.tolist() == [True] * 3 and count == 3 and pool == 0


@pytest.mark.parametrize("labels,third", [(("001", "003"), "002"), ((-3, 7), 1), ((1.25, 2.75), 2.0)])
def test_human_new_class_reorders_codes_and_refits_fresh_model(labels, third):
    session, _ = _session(labels)
    query = session.active_learning.suggest_query(batch_size=2)
    session.active_learning.label_rows(indices=query.indices, labels=[third, labels[0]])
    _check_model(session)
    assert third in session.activelearning_plan.classes_
    assert session.activelearning_plan.n_queries_used == 2


@pytest.mark.parametrize("labels", [("001", "002"), ("1.0", "2.0"), ("cat", "dog")])
def test_legacy_string_bundle_preserves_codes_and_refit_mapping(labels, tmp_path):
    session, truth = _session(labels)
    plan = session.activelearning_plan
    plan.label_encoder_ = LabelEncoder().fit(np.asarray(labels))
    old_weights = plan.estimator_.coef_.copy()
    session.active_learning.save_bundle(tmp_path / "legacy")
    other = Session.ingest(session.dataset.frame.copy()).set_roles({"x": "feature", "label": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    other.active_learning.load_bundle(tmp_path / "legacy", trusted=True)
    np.testing.assert_array_equal(other.activelearning_plan.estimator_.coef_, old_weights)
    assert any("refit" in w.lower() for w in other.active_learning.evaluate(partition="test").warnings)
    query = other.active_learning.suggest_query(batch_size=2)
    other.active_learning.label_rows(indices=query.indices, labels=[truth[i] for i in query.indices])
    assert not getattr(other.activelearning_plan.label_encoder_, "_buildml_raw_class_labels_", False)
    train = other.dataset.frame.iloc[list(other.split_plan.train_indices)].dropna(subset=["label"])
    encoder = LabelEncoder().fit(train.label.astype(str))
    oracle = LogisticRegression(max_iter=500, random_state=0).fit(train[["x"]].to_numpy(), encoder.transform(train.label.astype(str)))
    np.testing.assert_allclose(other.activelearning_plan.estimator_.coef_, oracle.coef_)
    expected_type = {"001": int, "1.0": float, "cat": str}[labels[0]]
    actual = decode_predictions(other.activelearning_plan.estimator_.predict(train[["x"]].to_numpy()), other.activelearning_plan.label_encoder_)
    assert all(type(value) is expected_type for value in actual)


def test_numeric_holdout_cannot_impersonate_string_classes(tmp_path):
    session, _ = _session(("1.0", "2.0"))
    session.active_learning.save_bundle(tmp_path / "raw")
    frame = session.dataset.frame.copy()
    positions = list(session.split_plan.test_indices)
    frame.loc[positions, "label"] = [float(value) for value in frame.loc[positions, "label"]]
    other = Session.ingest(frame).set_roles({"x": "feature", "label": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    other.active_learning.load_bundle(tmp_path / "raw", trusted=True)
    assert other.active_learning.evaluate(partition="test").metrics["accuracy"] == 0


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_holdout_labels_counted_without_changing_model(all_missing, tmp_path):
    session, _ = _session(("001", "002"))
    session.active_learning.save_bundle(tmp_path / "raw")
    frame = session.dataset.frame.copy()
    positions = list(session.split_plan.test_indices)
    blank = positions if all_missing else positions[::2]
    frame.loc[blank, "label"] = None
    other = Session.ingest(frame).set_roles({"x": "feature", "label": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=13
    )
    other.active_learning.load_bundle(tmp_path / "raw", trusted=True)
    plan = other.activelearning_plan
    before = plan.estimator_.coef_.copy()
    scored = other.active_learning.evaluate(partition="test")
    assert scored.n_unlabeled_eval == len(blank)
    assert scored.n_labeled_eval == len(positions) - len(blank)
    if all_missing:
        assert scored.metrics == {} and scored.warnings
    else:
        labeled = frame.iloc[positions].dropna(subset=["label"])
        predicted = decode_predictions(plan.estimator_.predict(labeled[["x"]].to_numpy()), plan.label_encoder_)
        assert scored.metrics["accuracy"] == sum(a == b for a, b in zip(labeled.label, predicted, strict=True)) / len(labeled)
    np.testing.assert_array_equal(plan.estimator_.coef_, before)
    assert other.activelearning_plan.n_queries_used == 0
