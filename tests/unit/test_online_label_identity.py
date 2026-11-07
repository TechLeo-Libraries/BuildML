"""Raw labels survive online updates and trusted persistence without type guessing."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from buildml import Session
from buildml.core.errors import ValidationError


def _frame(labels):
    rng = np.random.default_rng(82)
    x = rng.normal(size=(140, 2)) + 3
    codes = (x[:, 0] > x[:, 1]).astype(int)
    return pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": [labels[code] for code in codes]})


def _session(frame):
    return (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=9)
    )


def _fit(session, estimator="sgd_classifier", classes=None):
    return session.online.fit(
        estimator=estimator,
        n_init=18,
        chunk_size=13,
        random_state=11,
        classes=classes,
        drift_disclose=False,
        drift_detector="none",
    )


@pytest.mark.parametrize(
    "labels", [("001", "002"), ("1.0", "2.0"), (-3, 7), (1.25, 2.75), (False, True)]
)
@pytest.mark.parametrize("estimator", ["sgd_classifier", "bernoulli_nb"])
@pytest.mark.parametrize("explicit", [False, True])
def test_incremental_predictions_match_independent_coded_models_and_bundle(
    tmp_path, labels, estimator, explicit
):
    frame = _frame(labels)
    session = _session(frame)
    fit = _fit(session, estimator, labels if explicit else None)
    plan = session.online_plan
    train = session.partition("train")
    encoder = LabelEncoder().fit(train.y.to_numpy())
    assert tuple(plan.classes_) == tuple(encoder.classes_.tolist())
    assert json.loads(json.dumps(fit.to_dict()))["classes"] == encoder.classes_.tolist()
    reference = clone(plan.estimator_)
    reference.partial_fit(
        train[["x", "z"]].iloc[:18].to_numpy(),
        encoder.transform(train.y.iloc[:18]),
        classes=np.arange(2),
    )
    session.online.partial_fit(n_rows=13)
    reference.partial_fit(
        train[["x", "z"]].iloc[18:31].to_numpy(),
        encoder.transform(train.y.iloc[18:31]),
        classes=np.arange(2),
    )
    test = session.partition("test")
    expected = encoder.inverse_transform(reference.predict(test[["x", "z"]].to_numpy())).tolist()
    actual = session.online.predict().predictions
    assert actual == tuple(expected)
    assert all(type(value) is type(labels[0]) for value in actual)
    validation = session.partition("validation")
    expected_codes = reference.predict(validation[["x", "z"]].to_numpy())
    evaluation = session.online.evaluate(drift_check=False)
    assert evaluation.metrics["accuracy"] == pytest.approx(
        np.mean(expected_codes == encoder.transform(validation.y))
    )
    json.dumps(fit.to_dict())
    json.dumps(session.online_plan.to_dict())
    bundle = session.online.save_bundle(tmp_path / "online")
    restored = _session(frame)
    restored.online.load_bundle(bundle, trusted=True)
    assert restored.online.predict().predictions == tuple(expected)
    assert all(type(value) is type(labels[0]) for value in restored.online.predict().predictions)
    restored.online.partial_fit(n_rows=13)
    reference.partial_fit(
        train[["x", "z"]].iloc[31:44].to_numpy(),
        encoder.transform(train.y.iloc[31:44]),
        classes=np.arange(2),
    )
    assert restored.online.predict().predictions == tuple(
        encoder.inverse_transform(reference.predict(test[["x", "z"]].to_numpy())).tolist()
    )
    pd.testing.assert_frame_equal(session.dataset.frame, frame)


def test_mixed_numeric_and_string_labels_do_not_collapse():
    frame = _frame((0, 1))
    frame["y"] = pd.Series(([1, "1", 2] * 47)[: len(frame)], dtype=object)
    with pytest.raises(ValidationError, match="label|class"):
        _fit(_session(frame))


def test_external_update_cannot_impersonate_string_classes_with_numbers():
    session = _session(_frame(("1.0", "2.0")))
    _fit(session)
    before = session.online_plan.estimator_.coef_.copy()
    with pytest.raises(ValidationError, match="class"):
        session.online.partial_fit(
            frame=pd.DataFrame({"x": [4.0, 5.0], "z": [3.0, 3.0], "y": [1.0, 2.0]})
        )
    np.testing.assert_array_equal(session.online_plan.estimator_.coef_, before)
    assert session.online_plan.n_updates == 0


def test_holdout_numbers_cannot_impersonate_trained_string_classes():
    frame = _frame(("1.0", "2.0"))
    frame["y"] = frame.y.astype(object)
    session = _session(frame)
    rows = list(session.split_plan.validation_indices)
    frame.loc[rows, "y"] = 1.0
    session = _session(frame)
    _fit(session)
    with pytest.raises(ValidationError, match="class"):
        session.online.evaluate(drift_check=False)


@pytest.mark.parametrize(
    "labels, convert",
    [(("001", "002"), int), (("1.0", "2.0"), float), (("blue", "red"), str), ((0, 1), int)],
)
def test_real_legacy_bundles_retain_old_decode_and_disclose_refit(tmp_path, labels, convert):
    frame = _frame(labels)
    session = _session(frame)
    _fit(session)
    old_encoder = LabelEncoder().fit([str(label) for label in labels])  # Actual legacy encoder.
    session._online_plan = replace(
        session.online_plan, label_encoder_=old_encoder, classes_=tuple(old_encoder.classes_)
    )
    test = session.partition("test")
    codes = session.online_plan.estimator_.predict(test[["x", "z"]].to_numpy())
    expected = tuple(convert(value) for value in old_encoder.inverse_transform(codes))
    path = session.online.save_bundle(tmp_path / "legacy")
    restored = _session(frame)
    restored.online.load_bundle(path, trusted=True)
    result = restored.online.predict()
    assert result.predictions == expected
    assert any("refit" in text.lower() for text in (*result.disclosures, *result.warnings))
    evaluation = restored.online.evaluate(drift_check=False)
    assert any("refit" in text.lower() for text in (*evaluation.disclosures, *evaluation.warnings))


@pytest.mark.parametrize("all_missing", [False, True])
def test_null_holdout_labels_are_excluded_without_updating_the_model(all_missing):
    frame = _frame(("cold", "hot"))
    original = _session(frame)
    rows = list(original.split_plan.validation_indices)
    frame["y"] = frame.y.astype(object)
    frame.loc[rows if all_missing else rows[:5], "y"] = None
    session = _session(frame)
    _fit(session)
    plan = session.online_plan
    before = plan.estimator_.coef_.copy()
    counters = (plan.n_updates, plan.n_seen_rows)
    labeled = session.partition("validation").dropna(subset=["y"])
    result = session.online.evaluate(drift_check=False)
    assert result.n_rows == len(labeled)
    assert result.warnings
    if all_missing:
        assert result.metrics == {}
    else:
        encoder = LabelEncoder().fit(session.partition("train").y.to_numpy())
        expected = plan.estimator_.predict(labeled[["x", "z"]].to_numpy())
        assert result.metrics["accuracy"] == pytest.approx(
            np.mean(expected == encoder.transform(labeled.y))
        )
    np.testing.assert_array_equal(plan.estimator_.coef_, before)
    assert (plan.n_updates, plan.n_seen_rows) == counters


@pytest.mark.parametrize("partition", ["train", "all"])
def test_evaluation_including_train_rows_is_not_described_as_holdout(partition):
    session = _session(_frame(("001", "002")))
    _fit(session)
    result = session.online.evaluate(partition=partition, drift_check=False)
    assert any("training rows" in text for text in result.warnings)
    assert "never used" not in " ".join(result.disclosures)
