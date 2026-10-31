"""Typed action identities survive real cloning, bandit metrics and bundles."""

import itertools

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buildml import Session
from buildml.core.errors import ValidationError

VOCABULARIES = [
    ["001", "002"], ["1.00", "1.0"], ["-0", "0"], [False, True],
    [2, 10], [1.25, 2.5], [b"left", b"right"],
    [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
    [np.float32(1.1), np.float32(2.2)],
]
METHODS = ["logistic_regression", "hist_gradient_boosting", "linucb",
           "epsilon_greedy", "softmax"]


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _session(frame):
    return (Session.ingest(frame).set_roles({"x": "feature", "z": "feature",
                                           "action": "target", "reward": "ignore"})
            .split(test_size=.2, validation_size=.2, random_state=19))


def _frame(vocabulary):
    rng = np.random.default_rng(924)
    codes = np.arange(96) % len(vocabulary)
    x = rng.normal(size=(96, 2))
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1],
                          "reward": 2 + .4*x[:, 0] + codes*(.3+.6*x[:, 1])},
                         index=np.arange(96) % 7)
    frame["action"] = pd.array([vocabulary[i] for i in codes], dtype=object)
    return frame


def _fit(session, method):
    if method in METHODS[:2]:
        session.rl.fit_imitation(task="classification", estimator=method, columns=["x", "z"])
    else:
        session.rl.fit(algorithm=method, columns=["x", "z"],
                       action_column="action", reward_column="reward", alpha=.6)


@pytest.fixture(scope="module", params=list(itertools.product(METHODS, VOCABULARIES)))
def fitted(request):
    method, vocabulary = request.param
    frame = _frame(vocabulary)
    session = _session(frame)
    _fit(session, method)
    return session, frame, method, sorted(map(_scalar, vocabulary), key=str)


def _classification_metrics(truth, predictions):
    truth = [str(_scalar(v)) for v in truth]
    predictions = [str(_scalar(v)) for v in predictions]
    labels = sorted(set(truth) | set(predictions))
    f1 = []
    for label in labels:
        tp = sum(a == b == label for a, b in zip(truth, predictions, strict=True))
        denominator = sum(a == label for a in truth) + sum(a == label for a in predictions)
        f1.append(2*tp/denominator if denominator else 0)
    return {"accuracy": sum(a == b for a, b in zip(truth, predictions, strict=True))/len(truth),
            "macro_f1": sum(f1)/len(f1)}


def _bandit_oracle(session, frame, x, vocabulary):
    plan = session.rl_plan
    train = frame.iloc[list(session.split_plan.indices_for("train"))]
    tx = train[["x", "z"]].to_numpy()
    codes = np.array([vocabulary.index(_scalar(v)) for v in train.action])
    rewards = train.reward.to_numpy()
    means, scores = [], []
    for arm in range(len(vocabulary)):
        ax, ay = tx[codes == arm], rewards[codes == arm]
        if plan.algorithm == "linucb":
            covariance = np.eye(2) + ax.T @ ax
            theta = np.linalg.solve(covariance, ax.T @ ay)
            prediction = x @ theta
            uncertainty = np.sqrt(np.einsum("ij,ji->i", x, np.linalg.solve(covariance, x.T)))
            scores.append(prediction + .6*uncertainty)
            np.testing.assert_allclose(plan.policy_.A[arm], covariance, atol=1e-12)
            np.testing.assert_allclose(plan.policy_.b[arm], ax.T @ ay, atol=1e-12)
        else:
            center, target = ax.mean(axis=0), ay.mean()
            theta = np.linalg.solve(np.eye(2)+(ax-center).T@(ax-center),
                                    (ax-center).T@(ay-target))
            prediction = (x-center)@theta+target
            scores.append(prediction)
        means.append(prediction)
    return np.column_stack(scores), np.column_stack(means)


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("replay", [False, True])
def test_real_actions_and_metrics_match_independent_oracles(fitted, partition, replay, tmp_path):
    original, frame, method, vocabulary = fitted
    session = original
    imitation = method in METHODS[:2]
    if replay:
        if imitation:
            original.rl.save_imitation_bundle(tmp_path/"policy")
            session = _session(frame).rl.load_imitation_bundle(tmp_path/"policy", trusted=True)
        else:
            original.rl.save_bundle(tmp_path/"policy")
            session = _session(frame).rl.load_bundle(tmp_path/"policy", trusted=True)
    rows = frame.iloc[list(session.split_plan.indices_for(partition))]
    x = rows[["x", "z"]].to_numpy()
    plan = session.imitation_plan if imitation else session.rl_plan
    fingerprint = joblib.hash(plan)
    if imitation:
        model = plan.estimator_
        if method == "logistic_regression":
            codes = (x@model.coef_[0]+model.intercept_[0] > 0).astype(int)
        else:
            train = frame.iloc[list(session.split_plan.indices_for("train"))]
            reference = clone(model).fit(train[["x", "z"]].to_numpy(),
                                        [vocabulary.index(_scalar(v)) for v in train.action])
            codes = reference.predict(x)
        prediction = session.rl.predict_imitation(partition=partition)
        evaluation = session.rl.evaluate_imitation(partition=partition)
        expected = [vocabulary[i] for i in codes]
        assert evaluation.metrics == pytest.approx(_classification_metrics(rows.action, expected))
    else:
        scores, means = _bandit_oracle(session, frame, x, vocabulary)
        codes = scores.argmax(axis=1)
        expected = [vocabulary[i] for i in codes]
        prediction = session.rl.act(partition=partition)
        np.testing.assert_allclose(prediction.scores, scores, rtol=1e-10, atol=1e-12)
        evaluation = session.rl.evaluate(partition=partition)
        logged = np.array([vocabulary.index(_scalar(v)) for v in rows.action])
        reward = rows.reward.to_numpy()
        match = codes == logged
        propensity = plan.propensity_model_
        logits = x@propensity.coef_[0]+propensity.intercept_[0]
        positive = 1/(1+np.exp(-logits))
        probabilities = np.column_stack([1-positive, positive])
        expected_metrics = {"n_rows": len(rows), "direct_method": means[np.arange(len(x)), codes].mean(),
                            "ips": np.mean(reward*match/np.clip(probabilities[np.arange(len(x)), logged], 1e-6, 1)),
                            "action_match_rate": match.mean(), "mean_logged_reward": reward.mean(),
                            "mean_logged_reward_on_match": reward[match].mean() if match.any() else np.nan}
        for key, value in expected_metrics.items():
            assert evaluation.metrics[key] == pytest.approx(value, nan_ok=True)
        assert evaluation.offline
    assert list(prediction.actions) == expected
    assert all(type(a) is type(b) for a, b in zip(prediction.actions, expected, strict=True))
    assert joblib.hash(plan) == fingerprint


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("vocabulary", [[1, "1"], [True, "True"], [1.5, "1.5"]])
def test_ambiguous_action_fit_refuses_without_replacing_session_state(method, vocabulary):
    session = _session(_frame(["left", "right"]))
    _fit(session, method)
    state = joblib.hash((session.imitation_plan, session.rl_plan, session.history))
    data = session.dataset._ensure_pandas()
    data["action"] = pd.array([vocabulary[i % 2] for i in range(len(data))], dtype=object)
    with pytest.raises(ValidationError, match="[Aa]mbiguous.*action"):
        _fit(session, method)
    assert joblib.hash((session.imitation_plan, session.rl_plan, session.history)) == state


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("partition", ["train", "all"])
def test_diagnostic_partitions_do_not_claim_unseen_demonstrations(method, partition):
    session = _session(_frame(["left", "right"]))
    _fit(session, method)
    result = (session.rl.evaluate_imitation(partition=partition) if method in METHODS[:2]
              else session.rl.evaluate(partition=partition))
    assert any("diagnostic" in note.lower() for note in result.warnings)
    assert not any("held-out" in note.lower() or "holdout rows never" in note.lower()
                   for note in result.disclosures)


@pytest.mark.parametrize("method", METHODS)
def test_holdout_action_type_alias_is_refused_without_mutation(method):
    frame = _frame([1, 2])
    session = _session(frame)
    _fit(session, method)
    data = session.dataset._ensure_pandas()
    position = session.split_plan.indices_for("test")[0]
    data.iloc[position, data.columns.get_loc("action")] = "1"
    state = joblib.hash((session.imitation_plan, session.rl_plan, session.history))
    with pytest.raises(ValidationError, match="action"):
        if method in METHODS[:2]:
            session.rl.evaluate_imitation(partition="test")
        else:
            session.rl.evaluate(partition="test")
    assert joblib.hash((session.imitation_plan, session.rl_plan, session.history)) == state


@pytest.mark.parametrize("method", ["logistic_regression", "linucb"])
@pytest.mark.parametrize("vocabulary", [["001", "002"], [1, 2]])
@pytest.mark.parametrize("replay", [False, True])
def test_legacy_encoders_keep_recorded_keys_without_guessing_types(method, vocabulary, replay, tmp_path):
    frame = _frame(vocabulary)
    session = _session(frame)
    _fit(session, method)
    imitation = method == "logistic_regression"
    plan = session.imitation_plan if imitation else session.rl_plan
    if hasattr(plan.label_encoder_, "buildml_original_classes_"):
        del plan.label_encoder_.buildml_original_classes_
    if replay:
        if imitation:
            session.rl.save_imitation_bundle(tmp_path/"legacy")
            session = _session(frame).rl.load_imitation_bundle(tmp_path/"legacy", trusted=True)
        else:
            session.rl.save_bundle(tmp_path/"legacy")
            session = _session(frame).rl.load_bundle(tmp_path/"legacy", trusted=True)
    result = (session.rl.predict_imitation(partition="test") if imitation
              else session.rl.act(partition="test"))
    assert all(isinstance(value, str) for value in result.actions)
    assert set(result.actions).issubset(set(map(str, vocabulary)))
