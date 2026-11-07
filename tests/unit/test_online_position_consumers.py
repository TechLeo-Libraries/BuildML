"""Online updates and disclosed refits follow split positions, not index labels."""

import copy
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError

MODELS = ["river_logistic", "river_linear_regression", "sgd_classifier", "sgd_regressor"]


def _frame(layout):
    rng = np.random.default_rng(184)
    x = rng.normal(size=(72, 2))
    x[48:] += 20
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "target": 0.8 * x[:, 0] - 0.3 * x[:, 1]})
    if layout == "permuted":
        frame.index = np.roll(np.arange(72)[::-1], 9)
    elif layout == "duplicate":
        frame.index = np.arange(72) % 9
    return frame


def _session(frame):
    order = np.random.default_rng(83).permutation(48).tolist()
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"})
    s.inject_split(
        train_indices=order,
        validation_indices=list(range(48, 60)),
        test_indices=list(range(60, 72)),
    )
    return s, order


def _start(name, layout):
    river = name.startswith("river_")
    if river:
        pytest.importorskip("river")
    classification = name in {"river_logistic", "sgd_classifier"}
    frame = _frame(layout)
    if classification:
        frame.target = frame.target > 0
    s, order = _session(frame)
    s.online.fit(
        backend="industry" if river else "sklearn",
        estimator=name,
        n_init=12,
        chunk_size=10,
        drift_detector="none",
        drift_disclose=False,
        prefer_reduce_components=False,
        random_state=3,
        allow_refit_fallback=True,
    )
    return s, frame, order


def _reference(session):
    name = session.online_plan.estimator_name
    if name.startswith("river_"):
        from river import linear_model

        return (
            linear_model.LogisticRegression()
            if name == "river_logistic"
            else linear_model.LinearRegression()
        )
    return clone(session.online_plan.estimator_)


def _codes(plan, frame):
    if plan.task == "regression":
        return frame.target.to_numpy(dtype=float)
    lookup = {value: i for i, value in enumerate(plan.classes_)}
    return np.asarray([lookup[value] for value in frame.target])


def _learn(model, plan, rows):
    x, y = rows[["x", "z"]].to_numpy(), _codes(plan, rows)
    if plan.estimator_name.startswith("river_"):
        for row, target in zip(x, y, strict=True):
            model.learn_one(
                dict(f0=float(row[0]), f1=float(row[1])),
                int(target) if plan.task == "classification" else float(target),
            )
    elif plan.task == "classification":
        model.partial_fit(x, y, classes=np.arange(len(plan.classes_)))
    else:
        model.partial_fit(x, y)


def _predictions(model, plan, rows):
    x = rows[["x", "z"]].to_numpy()
    if plan.estimator_name.startswith("river_"):
        raw = [model.predict_one(dict(f0=float(row[0]), f1=float(row[1]))) for row in x]
        raw = [0 if value is None else value for value in raw]
    else:
        raw = model.predict(x)
    return [plan.classes_[int(value)] for value in raw] if plan.task == "classification" else raw


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("layout", ["permuted", "duplicate"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_split_position_stream_matches_independent_training_rows(
    name, layout, explicit, replay, tmp_path
):
    s, frame, order = _start(name, layout)
    reference = _reference(s)
    _learn(reference, s.online_plan, frame.iloc[order[:12]])
    np.testing.assert_allclose(
        s.online_plan.init_feature_means_, frame.iloc[order[:12]][["x", "z"]].mean()
    )
    if replay:
        bundle = s.online.save_bundle(tmp_path / "stream")
        s, _ = _session(frame)
        s.online.load_bundle(bundle, trusted=True)
    requested = order[8:22] if explicit else order[12:22]
    result = (
        s.online.partial_fit(indices=np.asarray(requested, dtype=np.int64))
        if explicit
        else s.online.partial_fit(n_rows=10)
    )
    _learn(reference, s.online_plan, frame.iloc[requested])
    seen = list(dict.fromkeys([*order[:12], *requested]))
    assert result.n_seen_rows == len(seen) and s.online_plan.cursor == 22
    assert list(s.online_plan.seen_train_indices) == seen
    expected = _predictions(reference, s.online_plan, frame.iloc[60:])
    actual = s.online.predict(partition="test").predictions
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert set(seen).isdisjoint(range(48, 72))
    pd.testing.assert_frame_equal(s.to_pandas(), frame)


@pytest.mark.parametrize("name", MODELS)
def test_explicit_rehearsal_cannot_rewind_the_next_unused_cursor(name):
    s, frame, order = _start(name, "memory")
    reference = _reference(s)
    _learn(reference, s.online_plan, frame.iloc[order[:12]])
    s.online.partial_fit(indices=[order[0], order[2]])
    _learn(reference, s.online_plan, frame.iloc[[order[0], order[2]]])
    assert s.online_plan.cursor == 12
    s.online.partial_fit(n_rows=10)
    _learn(reference, s.online_plan, frame.iloc[order[12:22]])
    assert s.online_plan.cursor == 22
    assert set(s.online_plan.seen_train_indices) == set(order[:22])
    np.testing.assert_allclose(
        s.online.predict(partition="test").predictions,
        _predictions(reference, s.online_plan, frame.iloc[60:]),
        atol=1e-12,
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("layout", ["permuted", "duplicate"])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_cumulative_refit_deduplicates_positions_without_dropping_distinct_rows(
    task, layout, overlap, replay, tmp_path
):
    s, frame, order = _start(
        "sgd_classifier" if task == "classification" else "sgd_regressor", layout
    )
    batch = (
        LogisticRegression(C=0.4, solver="liblinear", random_state=3)
        if task == "classification"
        else Ridge(alpha=0.7)
    )
    rows = frame.iloc[order[:12]]
    batch.fit(rows[["x", "z"]].to_numpy(), _codes(s.online_plan, rows))
    s._online_plan = replace(s.online_plan, estimator_=batch, estimator_name="legacy_batch_model")
    if replay:
        bundle = s.online.save_bundle(tmp_path / "legacy")
        s, _ = _session(frame)
        s.online.load_bundle(bundle, trusted=True)
    old_model = s.online_plan.estimator_
    old_prediction = old_model.predict(frame[["x", "z"]].to_numpy())
    selected = order[8:22] if overlap else order[12:26]
    result = s.online.partial_fit(indices=selected)
    positions = list(dict.fromkeys([*order[:12], *selected]))
    rows = frame.iloc[positions]
    expected = clone(batch).fit(rows[["x", "z"]].to_numpy(), _codes(s.online_plan, rows))
    np.testing.assert_allclose(
        s.online_plan.estimator_.predict(frame[["x", "z"]].to_numpy()),
        expected.predict(frame[["x", "z"]].to_numpy()),
        atol=1e-12,
    )
    np.testing.assert_array_equal(old_model.predict(frame[["x", "z"]].to_numpy()), old_prediction)
    assert result.n_seen_rows == len(positions) and result.used_refit_fallback
    assert list(s.online_plan.seen_train_indices) == positions
    assert any(f"n={len(positions)} cumulative rows" in note for note in result.disclosures)
    pd.testing.assert_frame_equal(s.to_pandas(), frame)


@pytest.mark.parametrize("requested", [[True], [np.bool_(False)], [2.0], ["2"], [None], [[2]], []])
@pytest.mark.parametrize("stage", ["fit", "update"])
def test_invalid_position_requests_refuse_before_replacing_stream_or_history(requested, stage):
    s, _, _ = _start("sgd_regressor", "memory")
    plan = s.online_plan
    before = joblib.hash(plan)
    history = copy.deepcopy(s.history)
    with pytest.raises(ValidationError):
        if stage == "fit":
            s.online.fit(estimator="sgd_regressor", indices=requested, n_init=12)
        else:
            s.online.partial_fit(indices=requested)
    assert s.online_plan is plan and joblib.hash(plan) == before
    assert s.history == history
