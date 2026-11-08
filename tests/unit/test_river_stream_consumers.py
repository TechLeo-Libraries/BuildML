"""Real River streams preserve multiclass meaning and frozen holdout detectors."""

import copy
import itertools

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

pytest.importorskip("river")
from river import linear_model, multiclass, tree  # noqa: E402

CLASSIFIERS = ["river_logistic", "river_pa", "river_hoeffding"]
REGRESSORS = ["river_linear_regression", "river_hoeffding_regressor"]
DETECTORS = ["none", "adwin", "page_hinkley"]


def _ready(frame, n_train=240):
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"})
    s.inject_split(
        train_indices=list(range(n_train)),
        validation_indices=list(range(n_train, n_train + 60)),
        test_indices=list(range(n_train + 60, len(frame))),
    )
    return s


def _data(layout):
    rng = np.random.default_rng(634)
    x = rng.normal(size=(360, 2))
    if layout == "regression":
        target = 0.6 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(0, 0.1, len(x))
    else:
        if layout == "binary_bool":
            target = x[:, 0] > x[:, 1]
        else:
            codes = np.argmax(np.column_stack([x[:, 0], x[:, 1], -x.sum(axis=1)]), axis=1)
            if layout == "late_class":
                # Declared class vocabulary contains a class absent from init rows.
                codes[:120] %= 2
            labels = ["003", "09", "21"] if layout != "three_ints" else [7, 21, 90]
            target = [labels[i] for i in codes]
    return pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "target": target})


def _clone_session(source):
    # Copy the actual fitted learner, without copying Session facade caches.
    session = Session(dataset=source.dataset, split_plan=source.split_plan)
    session._online_plan = copy.deepcopy(source.online_plan)
    return session


def _direct(name, n_classes):
    factories = {
        "river_logistic": linear_model.LogisticRegression,
        "river_pa": linear_model.PAClassifier,
        "river_hoeffding": tree.HoeffdingTreeClassifier,
        "river_linear_regression": linear_model.LinearRegression,
        "river_hoeffding_regressor": tree.HoeffdingTreeRegressor,
    }
    model = factories[name]()
    if n_classes > 2 and name in {"river_logistic", "river_pa"}:
        model = multiclass.OneVsRestClassifier(model)
    return model


def _rows(frame):
    return [
        dict(f0=float(x), f1=float(z))
        for x, z in frame[["x", "z"]].itertuples(index=False, name=None)
    ]


def _targets(plan, frame):
    if plan.task == "regression":
        return frame.target.to_numpy()
    codes = {value: i for i, value in enumerate(plan.classes_)}
    return np.asarray([codes[value] for value in frame.target])


def _learn(model, plan, frame):
    for row, target in zip(_rows(frame), _targets(plan, frame), strict=True):
        model.learn_one(row, int(target) if plan.task == "classification" else float(target))


def _expected(model, plan, frame):
    raw = [model.predict_one(row) for row in _rows(frame)]
    if plan.task == "regression":
        return np.asarray([float(value) if value is not None else 0.0 for value in raw])
    return np.asarray([int(value) if value is not None else 0 for value in raw])


@pytest.fixture(
    scope="module",
    params=list(
        itertools.product(
            CLASSIFIERS, ["binary_bool", "three_padded", "three_ints", "late_class"], DETECTORS
        )
    ),
)
def classifier(request, tmp_path_factory):
    name, layout, detector = request.param
    frame = _data(layout)
    s = _ready(frame)
    s.online.fit(
        backend="industry",
        estimator=name,
        n_init=120,
        chunk_size=60,
        drift_detector=detector,
        prefer_reduce_components=False,
    )
    path = s.online.save_bundle(tmp_path_factory.mktemp("river-classifier") / "bundle")
    return s, frame, path


@pytest.fixture(scope="module", params=list(itertools.product(REGRESSORS, DETECTORS)))
def regressor(request, tmp_path_factory):
    name, detector = request.param
    frame = _data("regression")
    s = _ready(frame)
    s.online.fit(
        backend="industry",
        estimator=name,
        n_init=120,
        chunk_size=60,
        drift_detector=detector,
        prefer_reduce_components=False,
    )
    path = s.online.save_bundle(tmp_path_factory.mktemp("river-regressor") / "bundle")
    return s, frame, path


def _metric_oracle(truth, prediction, classification):
    if not classification:
        residual = np.asarray(truth) - np.asarray(prediction)
        return {
            "mae": np.abs(residual).mean(),
            "rmse": np.sqrt(np.mean(residual**2)),
            "r2": 1 - np.sum(residual**2) / np.sum((truth - np.mean(truth)) ** 2),
        }
    classes = sorted(set(truth) | set(prediction))
    scores, supports = [], []
    for label in classes:
        tp = sum(a == label and b == label for a, b in zip(truth, prediction, strict=True))
        fp = sum(a != label and b == label for a, b in zip(truth, prediction, strict=True))
        fn = sum(a == label and b != label for a, b in zip(truth, prediction, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        supports.append(sum(value == label for value in truth))
    return {
        "accuracy": np.mean(np.asarray(truth) == prediction),
        "f1_macro": np.mean(scores),
        "f1_weighted": np.dot(scores, supports) / sum(supports),
    }


def _stream_oracle(fitted, replay):
    source, frame, bundle = fitted
    session = (
        _ready(frame).online.load_bundle(bundle, trusted=True) if replay else _clone_session(source)
    )
    plan = session.online_plan
    n_classes = len(plan.classes_ or ())
    direct = _direct(plan.estimator_name, n_classes)
    _learn(direct, plan, frame.iloc[:120])
    before_data = session.to_pandas()
    for n_seen in (120, 180, 240):
        if n_seen > 120:
            result = session.online.partial_fit(n_rows=60)
            _learn(direct, plan, frame.iloc[n_seen - 60 : n_seen])
            assert result.n_seen_rows == n_seen
            assert session.online_plan.cursor == n_seen
        plan = session.online_plan
        for partition, positions in (("validation", range(240, 300)), ("test", range(300, 360))):
            rows = frame.iloc[list(positions)]
            expected = _expected(direct, plan, rows)
            result = session.online.predict(partition=partition)
            if plan.task == "classification":
                decoded = [plan.classes_[i] for i in expected]
                assert list(result.predictions) == decoded
                assert [type(v) for v in result.predictions] == [type(v) for v in decoded]
            else:
                np.testing.assert_allclose(result.predictions, expected, atol=1e-12)
            evaluated = session.online.evaluate(partition=partition, drift_check=False)
            metrics = _metric_oracle(_targets(plan, rows), expected, plan.task == "classification")
            assert evaluated.metrics == pytest.approx(metrics)
        assert plan.n_seen_rows == n_seen
        assert set(plan.seen_train_indices) == set(range(n_seen))
    pd.testing.assert_frame_equal(session.to_pandas(), before_data)
    assert len(source.online_plan.seen_train_indices) == 120


@pytest.mark.parametrize("replay", [False, True])
def test_classification_stream_and_replay_match_direct_river_and_scalar_metrics(classifier, replay):
    _stream_oracle(classifier, replay)


@pytest.mark.parametrize("replay", [False, True])
def test_regression_stream_and_replay_match_direct_river_and_scalar_metrics(regressor, replay):
    _stream_oracle(regressor, replay)


@pytest.fixture(
    scope="module",
    params=list(itertools.product(CLASSIFIERS + REGRESSORS, ["adwin", "page_hinkley"])),
)
def drift_source(request, tmp_path_factory):
    name, detector = request.param
    classification = name in CLASSIFIERS
    frame = pd.DataFrame(
        {
            "x": np.zeros(1200),
            "z": np.zeros(1200),
            "target": [0] * 400 + ([1] * 800 if classification else [30.0] * 800),
        }
    )
    session = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"})
    session.inject_split(
        train_indices=list(range(400)),
        validation_indices=list(range(400, 800)),
        test_indices=list(range(800, 1200)),
    )
    session.online.fit(
        backend="industry",
        estimator=name,
        n_init=400,
        classes=[0, 1] if classification else None,
        drift_detector=detector,
        prefer_reduce_components=False,
    )
    bundle = session.online.save_bundle(tmp_path_factory.mktemp("river-drift") / "bundle")
    return session, frame, bundle


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_holdout_drift_matches_copied_detector_without_changing_future_updates(
    drift_source, partition, replay
):
    source, frame, bundle = drift_source
    session = _clone_session(source)
    if replay:
        session.online.load_bundle(bundle, trusted=True)
    control = _clone_session(source)
    plan = session.online_plan
    wrapper = plan.estimator_
    detector = copy.deepcopy(
        wrapper.adwin_ if wrapper.drift_detector == "adwin" else wrapper.page_hinkley_
    )
    rows = frame.iloc[400:800] if partition == "validation" else frame.iloc[800:1200]
    preds = _expected(wrapper.model_, plan, rows)
    truth = _targets(plan, rows)
    signals = (
        (preds != truth).astype(float) if plan.task == "classification" else np.abs(preds - truth)
    )
    fired = False
    for signal in signals:
        detector.update(float(signal))
        fired |= bool(detector.drift_detected)
    assert fired, "The real detector must observe the constructed error shift"
    state = joblib.hash(
        (wrapper.adwin_, wrapper.page_hinkley_, wrapper.drift_events_, wrapper.n_seen_)
    )
    ledger = copy.deepcopy(plan.to_dict())
    first = session.online.evaluate(partition=partition)
    assert first.drift_detected == fired
    assert (
        joblib.hash((wrapper.adwin_, wrapper.page_hinkley_, wrapper.drift_events_, wrapper.n_seen_))
        == state
    )
    second = session.online.evaluate(partition=partition)
    assert second.metrics == first.metrics and second.drift_notes == first.drift_notes
    assert session.online_plan.to_dict() == ledger
    # A later real training update must match the stream that never saw holdout.
    update = pd.DataFrame({"x": [0.0] * 40, "z": [0.0] * 40, "target": [0] * 40})
    session.online.partial_fit(frame=update)
    control.online.partial_fit(frame=update)
    a, b = session.online_plan.estimator_, control.online_plan.estimator_
    assert joblib.hash((a.adwin_, a.page_hinkley_, a.drift_events_)) == joblib.hash(
        (b.adwin_, b.page_hinkley_, b.drift_events_)
    )
    np.testing.assert_allclose(a.predict(np.ones((10, 2))), b.predict(np.ones((10, 2))))


@pytest.mark.parametrize("name", ["river_logistic", "river_pa"])
@pytest.mark.parametrize(
    "operation,layout", [("save", None), ("load", "direct"), ("load", "payload")]
)
def test_legacy_binary_models_cannot_claim_multiclass_vocabulary(name, layout, operation, tmp_path):
    frame = _data("three_padded")
    session = _ready(frame)
    session.online.fit(backend="industry", estimator=name, n_init=120, drift_detector="none")
    root = session.online.save_bundle(tmp_path / "valid")
    payload = joblib.load(root / "online_plan.joblib")
    broken = payload["plan"]
    # A genuine real binary learner with the old three-code training behavior.
    binary = _direct(name, 2)
    _learn(binary, broken, frame.iloc[:120])
    broken.estimator_.model_ = binary
    if operation == "save":
        from buildml.online.checkpoint import save_online_bundle

        with pytest.raises(ValidationError, match="refit|multiclass"):
            save_online_bundle(tmp_path / "invalid", broken)
        assert not (tmp_path / "invalid").exists()
    else:
        joblib.dump(broken if layout == "direct" else {"plan": broken}, root / "online_plan.joblib")
        previous = session.online_plan
        history = copy.deepcopy(session.history)
        with pytest.raises(ValidationError, match="refit|multiclass"):
            session.online.load_bundle(root, trusted=True)
        assert session.online_plan is previous and session.history == history
