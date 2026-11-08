"""Synthetic fidelity and utility retain scalar category identities."""

from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from buildml import Session
from buildml.core.errors import ValidationError


def _session(frame, columns):
    session = Session.ingest(frame).set_roles(
        {name: "target" if name == "y" else "feature" for name in frame}
    ).inject_split(
        train_indices=list(range(80)), validation_indices=list(range(80, 100)),
        test_indices=list(range(100, 120)),
    )
    session.synthetic.fit(method="bootstrap", columns=columns, smooth_sigma=0, random_state=3)
    return session


def _replay(session, frame, tmp_path, replay):
    if not replay:
        return session
    session.synthetic.save_bundle(tmp_path / "generator")
    other = Session.ingest(frame).set_roles(dict(session.dataset.roles))
    other.inject_split(
        train_indices=list(session.split_plan.train_indices),
        validation_indices=list(session.split_plan.validation_indices),
        test_indices=list(session.split_plan.test_indices),
    )
    other.synthetic.load_bundle(tmp_path / "generator", trusted=True)
    return other


def _identity(value):
    # Independent specification: nulls form one category, distinct scalar types
    # remain distinct, and numpy scalar wrappers retain their Python meaning.
    if pd.isna(value):
        return ("missing",)
    if isinstance(value, np.generic):
        value = value.item()
    return (type(value).__name__, value)


@pytest.mark.parametrize("values", [
    (1, "1"), (False, "False"), (1.5, "1.5"), (None, "__NA__"),
    (pd.NA, "__NA__"), (True, 1), ("left", "right"), (2, 10),
    (np.int64(1), "1"), (np.float32(1.5), "1.5"), (np.bool_(False), "False"),
])
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_fidelity_matches_counted_categories_without_spelling_collisions(
    values, partition, replay, tmp_path,
):
    category = [values[i % 2] for i in range(80)] + [values[0]] * 40
    frame = pd.DataFrame({"x": np.linspace(-3, 7, 120),
                          "category": pd.Series(category, dtype=object)},
                         index=np.arange(120))
    frame.index = np.arange(900, 1020)  # All split indices are row positions.
    session = _replay(_session(frame, ["x", "category"]), frame, tmp_path, replay)
    before, frozen = session.to_pandas().copy(deep=True), joblib.hash(session.synthesizer_plan)
    sampled = session.synthetic.sample(n=73, random_state=11).frame
    real = session.partition(partition)
    a, b = Counter(map(_identity, real.category)), Counter(map(_identity, sampled.category))
    expected = sum(abs(a[key] / len(real) - b[key] / len(sampled)) for key in a.keys() | b.keys()) / 2
    result = session.synthetic.evaluate(
        mode="fidelity", eval_backend="builtin", partition=partition,
        n_synthetic=73, random_state=11,
    )
    assert result.per_column["category"]["tv"] == pytest.approx(expected)
    assert result.metrics["mean_tv"] == pytest.approx(expected)
    points = sorted(set(real.x) | set(sampled.x))
    ks = max(abs(np.mean(real.x <= point) - np.mean(sampled.x <= point)) for point in points)
    assert result.per_column["x"]["ks"] == pytest.approx(ks)
    assert result.n_real == len(real) and result.n_synthetic == 73
    assert bool(result.warnings) == (partition == "train")
    assert joblib.hash(session.synthesizer_plan) == frozen
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("labels", [(1, "1"), (False, "False"), (1.5, "1.5"),
                                    (2, 10), ("02", "10"), (False, True)])
@pytest.mark.parametrize("holdout", ["known", "unseen_spelling"])
@pytest.mark.parametrize("estimator", ["auto", "ridge"])
@pytest.mark.parametrize("replay", [False, True])
def test_tstr_and_trtr_use_train_label_identity_and_count_unseen_holdout_classes(
    labels, holdout, estimator, replay, tmp_path,
):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 2))
    positive = x[:, 0] - .3 * x[:, 1] > 0
    y = [labels[int(value)] for value in positive]
    if holdout == "unseen_spelling":
        # The same display text with a distinct scalar type is an unseen class.
        y[80:] = [str(value) if not isinstance(value, str) else "unseen:" + value
                  for value in y[80:]]
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": pd.Series(y, dtype=object)})
    session = _replay(_session(frame, ["x", "z", "y"]), frame, tmp_path, replay)
    frozen, before = joblib.hash(session.synthesizer_plan), session.to_pandas().copy(deep=True)
    sample = session.synthetic.sample(n=100, random_state=19).frame
    train, test = session.partition("train"), session.partition("test")
    observed = [_identity(value) for value in train.y]
    classes = list(dict.fromkeys(observed))
    expected = {}
    for name, fitted_rows in [("", sample), ("trtr_", train)]:
        a, b = fitted_rows[["x", "z"]].to_numpy(), test[["x", "z"]].to_numpy()
        mean, scale = a.mean(axis=0), a.std(axis=0)
        model = RidgeClassifier() if estimator == "ridge" else LogisticRegression(max_iter=400)
        codes = np.array([classes.index(_identity(value)) for value in fitted_rows.y])
        model.fit((a - mean) / scale, codes)
        predictions = [classes[code] for code in model.predict((b - mean) / scale)]
        truth = [_identity(value) for value in test.y]
        accuracy = np.mean([actual == predicted for actual, predicted in zip(truth, predictions, strict=True)])
        categories = set(truth) | set(predictions)
        f1 = []
        for category in categories:
            tp = sum(a == category and b == category for a, b in zip(truth, predictions, strict=True))
            denom = sum(a == category for a in truth) + sum(b == category for b in predictions)
            f1.append(2 * tp / denom)
        expected[name + "accuracy"] = accuracy
        expected[name + "macro_f1"] = np.mean(f1)
    result = session.synthetic.evaluate(mode="tstr", estimator=estimator,
                                        n_synthetic=100, random_state=19)
    for key, value in expected.items():
        assert result.metrics[key] == pytest.approx(value)
    assert result.metrics["score"] == pytest.approx(expected["accuracy"])
    assert result.metrics["tstr_gap_vs_trtr"] == pytest.approx(expected["trtr_accuracy"] - expected["accuracy"])
    assert result.warnings == ()
    assert joblib.hash(session.synthesizer_plan) == frozen
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("where", ["train", "test"])
@pytest.mark.parametrize("estimator", ["auto", "ridge"])
def test_missing_utility_targets_refuse_without_storing_a_report(missing, where, estimator):
    frame = pd.DataFrame({"x": np.linspace(-2, 3, 120),
                          "y": pd.Series(["left", "right"] * 60, dtype=object)})
    indices = range(0, 80, 3) if where == "train" else range(100, 120, 3)
    for index in indices:
        frame.loc[index, "y"] = missing
    session = _session(frame, ["x", "y"])
    before, frozen, history = session.to_pandas().copy(), joblib.hash(session.synthesizer_plan), list(session.history)
    with pytest.raises(ValidationError, match="target.*(null|missing)"):
        session.synthetic.evaluate(mode="tstr", estimator=estimator, n_synthetic=100, random_state=19)
    assert session.synthetic_eval_result is None
    assert session.history == history and joblib.hash(session.synthesizer_plan) == frozen
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("estimator", ["auto", "ridge"])
@pytest.mark.parametrize("replay", [False, True])
def test_trtr_missing_target_cannot_hide_when_bootstrap_omits_that_row(
    missing, estimator, replay, tmp_path,
):
    frame = pd.DataFrame({"x": np.arange(120, dtype=float),
                          "y": pd.Series(["left", "right"] * 60, dtype=object)})
    probe = _session(frame, ["x", "y"])
    sampled = probe.synthetic.sample(n=100, random_state=19).frame
    omitted = min(set(range(80)) - set(sampled.x))
    frame.loc[omitted, "y"] = missing
    session = _replay(_session(frame, ["x", "y"]), frame, tmp_path, replay)
    assert session.synthetic.sample(n=100, random_state=19).frame.y.notna().all()
    previous = session.synthetic.evaluate(mode="fidelity", eval_backend="builtin",
                                          n_synthetic=100, random_state=19)
    history, frozen = list(session.history), joblib.hash(session.synthesizer_plan)
    with pytest.raises(ValidationError, match="target.*(null|missing)"):
        session.synthetic.evaluate(mode="tstr", estimator=estimator,
                                   n_synthetic=100, random_state=19)
    assert session.synthetic_eval_result is previous
    assert session.history == history and joblib.hash(session.synthesizer_plan) == frozen


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_one_synthetic_observation_has_conservative_numeric_distance(partition):
    frame = pd.DataFrame({"x": np.linspace(-2, 3, 120), "empty": np.full(120, np.nan)})
    session = _session(frame, ["x", "empty"])
    result = session.synthetic.evaluate(mode="fidelity", eval_backend="builtin",
                                        partition=partition, n_synthetic=1, random_state=5)
    assert result.metrics["mean_ks"] == 1
    assert result.metrics["corr_l1"] == 0
    assert all(column["ks"] == 1 for column in result.per_column.values())


@pytest.mark.parametrize("arguments, message", [
    ({"mode": "unknown"}, "mode"), ({"partition": "all"}, "partition"),
    ({"eval_backend": "unknown"}, "eval_backend"),
    ({"mode": "tstr"}, "target column"),
])
def test_invalid_evaluation_requests_leave_session_and_generator_unchanged(arguments, message):
    frame = pd.DataFrame({"x": np.linspace(-2, 3, 120)})
    session = _session(frame, ["x"])
    frozen, history = joblib.hash(session.synthesizer_plan), list(session.history)
    with pytest.raises(ValidationError, match=message):
        session.synthetic.evaluate(**{"eval_backend": "builtin", **arguments})
    assert session.synthetic_eval_result is None
    assert session.history == history and joblib.hash(session.synthesizer_plan) == frozen
