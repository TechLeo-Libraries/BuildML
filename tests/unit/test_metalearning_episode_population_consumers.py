"""Public episodic reports must describe the selected task and row populations."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buildml import Session
from buildml.core.errors import ValidationError


def _data(collision=None):
    rng = np.random.default_rng(941)
    parts, bounds, start = [], {}, 0
    populations = {"train": ["seen-a", "seen-b", "seen-c", "seen-d"],
                   "validation": ["seen-a", "new-x", "new-y"],
                   "test": ["seen-b", "new-x", "new-y"]}
    if collision is not None:
        populations["validation"] = list(collision)
        populations["test"] = list(collision)
    for partition, tasks in populations.items():
        for number, task in enumerate(tasks):
            n = 20 if partition == "train" else 18
            labels = np.arange(n) % 2
            strength = [1.4, .05, -.4, .8][number]
            x = rng.normal(size=(n, 2)) + labels[:, None]*np.array([strength, .25])
            parts.append(pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": labels,
                                       "task": pd.Series([task]*n, dtype=object)}))
        stop = sum(len(part) for part in parts)
        bounds[partition] = list(range(start, stop))
        start = stop
    return pd.concat(parts, ignore_index=True), bounds


def _session(method, holdout=0., collision=None, fit=True, **overrides):
    frame, rows = _data(collision)
    session = Session.ingest(frame).set_roles(
        {"x": "feature", "z": "feature", "y": "target", "task": "group"})
    session.inject_split(train_indices=rows["train"], validation_indices=rows["validation"],
                         test_indices=rows["test"])
    if fit:
        options = dict(backend="sklearn", method=method, k_shot=2, n_query=6,
                       n_way=2, n_episodes=2, task_holdout_fraction=holdout, random_state=17)
        options.update(overrides)
        session.metalearning.fit(**options)
    return session


@pytest.fixture(scope="module", params=["prototypical", "warm_start"])
def method(request):
    return request.param


@pytest.fixture(scope="module", params=[0., .25])
def fitted(method, request):
    return _session(method, request.param)


@pytest.fixture(scope="module")
def simple(method):
    return _session(method)


def _selected(session, partition, prefer):
    frame = session.to_pandas() if partition == "all" else session.partition(partition)
    present = list(dict.fromkeys(frame.task.tolist()))
    plan = session.metalearning_plan
    novel = [task for task in present if task not in plan.train_task_ids]
    if not prefer:
        selected = present
    elif partition == "train" and plan.held_out_task_ids:
        selected = list(plan.held_out_task_ids)
    else:
        selected = novel or present
    return frame, selected, novel, [task for task in present if task in plan.train_task_ids]


def _oracle(session, partition, prefer, seed):
    frame, tasks, novel, overlapping = _selected(session, partition, prefer)
    rng = np.random.default_rng(seed)
    per_task = {}
    for task in tasks:
        rows = frame.loc[frame.task == task]
        support, query = [], []
        # Independently draw the documented balanced two-class episode, retaining
        # source row positions. No BuildML sampler or metric helper is called.
        for label in rng.choice([0, 1], 2, replace=False):
            members = np.flatnonzero(rows.y.to_numpy() == label)
            picked = rng.choice(members, 5, replace=False)
            support.extend(picked[:2])
            query.extend(picked[2:])
        train, test = rows.iloc[support], rows.iloc[query]
        if session.metalearning_plan.method == "prototypical":
            means = [train.loc[train.y == label, ["x", "z"]].mean().to_numpy() for label in [0, 1]]
            predicted = np.array([min([0, 1], key=lambda label: float(np.sum((row-means[label])**2)))
                                  for row in test[["x", "z"]].to_numpy()])
        else:
            model = clone(session.metalearning_plan.init_estimator_).fit(
                train[["x", "z"]].to_numpy(), train.y.to_numpy())
            logits = (test[["x", "z"]].to_numpy() @ model.coef_.T + model.intercept_)[:, 0]
            predicted = (logits > 0).astype(int)
        truth = test.y.to_numpy()
        f1 = []
        for label in [0, 1]:
            tp = int(np.sum((truth == label) & (predicted == label)))
            denominator = int(np.sum(truth == label) + np.sum(predicted == label))
            f1.append(2*tp/denominator if denominator else 0.)
        per_task[str(task)] = dict(accuracy=float(np.mean(predicted == truth)),
                                   f1_macro=float(np.mean(f1)), n_support=4., n_query=6.)
    return per_task, novel, overlapping


@pytest.mark.parametrize("partition", ["train", "validation", "test", "all"])
@pytest.mark.parametrize("prefer", [True, False])
def test_report_population_and_scalar_metrics_follow_the_requested_task_policy(fitted, partition, prefer):
    session = fitted
    expected, novel, overlapping = _oracle(session, partition, prefer, 23)
    before = session.to_pandas().copy(deep=True)
    fingerprint = joblib.hash(session.metalearning_plan)
    result = session.metalearning.evaluate(partition=partition, prefer_novel_tasks=prefer,
                                           random_state=23)
    assert result.novel_task_ids == tuple(novel)
    assert result.overlapping_task_ids == tuple(overlapping)
    assert result.n_tasks_evaluated == len(expected)
    assert result.n_query_rows == 6*len(expected)
    assert set(result.per_task_metrics) == set(expected)
    for task, metrics in expected.items():
        assert result.per_task_metrics[task] == pytest.approx(metrics)
    assert result.metrics == pytest.approx({
        "mean_accuracy": np.mean([item["accuracy"] for item in expected.values()]),
        "mean_f1_macro": np.mean([item["f1_macro"] for item in expected.values()]),
        "n_tasks_scored": len(expected)})
    serialized = json.loads(json.dumps(result.to_dict()))
    assert len(serialized["per_task_metrics"]) == serialized["n_tasks_evaluated"]
    selected_overlap = set(expected).intersection(str(task) for task in overlapping)
    if selected_overlap:
        assert any("overlap" in message.lower() for message in result.warnings)
        if partition in {"train", "all"}:
            assert not any("rows were never used for meta-training" in message for message in result.disclosures)
    assert joblib.hash(session.metalearning_plan) == fingerprint
    pd.testing.assert_frame_equal(session.to_pandas(), before)


@pytest.mark.parametrize("collision", [(1, "1"), (False, "False")])
@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("prefer", [True, False])
def test_ambiguous_task_report_keys_refuse_before_replacing_a_prior_report(method, collision, partition, prefer):
    session = _session(method, collision=collision)
    previous = session.metalearning.evaluate(partition="train")
    before, fingerprint = list(session._history), joblib.hash(session.metalearning_plan)
    with pytest.raises(ValidationError, match="ambiguous|collision"):
        session.metalearning.evaluate(partition=partition, prefer_novel_tasks=prefer)
    assert session._history == before and session._metalearning_eval_result is previous
    assert joblib.hash(session.metalearning_plan) == fingerprint


BAD_COUNTS = [True, 2.5, "2", np.nan, 0]


@pytest.mark.parametrize("field", ["k_shot", "n_query", "n_way", "n_episodes"])
@pytest.mark.parametrize("value", BAD_COUNTS)
def test_refit_refuses_lossy_episode_counts_without_replacing_a_working_plan(method, field, value):
    session = _session(method)
    previous = session.metalearning_plan
    before, fingerprint = list(session._history), joblib.hash(previous)
    options = dict(backend="sklearn", method=method, k_shot=2, n_query=6, n_way=2,
                   n_episodes=2, task_holdout_fraction=0., random_state=17)
    options[field] = value
    with pytest.raises(ValidationError, match=field):
        session.metalearning.fit(**options)
    assert session._history == before and session.metalearning_plan is previous
    assert joblib.hash(previous) == fingerprint


@pytest.mark.parametrize("field", ["k_shot", "n_query", "n_way"])
@pytest.mark.parametrize("value", BAD_COUNTS)
def test_evaluation_refuses_invalid_episode_counts_before_replacing_a_report(simple, field, value):
    session = simple
    previous = session.metalearning.evaluate()
    before, fingerprint = list(session._history), joblib.hash(session.metalearning_plan)
    with pytest.raises(ValidationError, match=field):
        session.metalearning.evaluate(**{field: value})
    assert session._history == before and session._metalearning_eval_result is previous
    assert joblib.hash(session.metalearning_plan) == fingerprint


@pytest.mark.parametrize("value", BAD_COUNTS)
def test_support_cap_refuses_lossy_counts_without_replacing_adaptation(simple, value):
    session = simple
    previous = session.metalearning.adapt(task_id="new-x", partition="validation",
                                                max_support_per_class=2)
    before, fingerprint = list(session._history), joblib.hash(session.metalearning_plan)
    with pytest.raises(ValidationError, match="max_support_per_class"):
        session.metalearning.adapt(task_id="new-x", partition="validation",
                                         max_support_per_class=value)
    assert session._history == before and session._metalearning_adapt_result is previous
    assert joblib.hash(session.metalearning_plan) == fingerprint


@pytest.mark.parametrize("prefer", [True, False])
def test_public_bundle_roundtrip_preserves_population_policy_and_numerical_report(simple, prefer, tmp_path):
    source = simple
    root = source.metalearning.save_bundle(tmp_path / "owned-meta")
    restored = _session(source.metalearning_plan.method, fit=False)
    restored.metalearning.load_bundle(root, trusted=True)
    for session in [source, restored]:
        expected, _, _ = _oracle(session, "test", prefer, 23)
        result = session.metalearning.evaluate(partition="test", prefer_novel_tasks=prefer,
                                               random_state=23)
        assert result.n_tasks_evaluated == len(expected)
        for task in expected:
            assert result.per_task_metrics[task] == pytest.approx(expected[task])


def test_numpy_integral_protocols_and_support_caps_remain_supported(method):
    session = _session(method, k_shot=np.int64(2), n_query=np.int64(6),
                       n_way=np.int64(2), n_episodes=np.int64(2))
    result = session.metalearning.evaluate(k_shot=np.int64(2), n_query=np.int64(6), n_way=np.int64(2))
    assert result.n_tasks_evaluated == 2 and result.n_query_rows == 12
    adaptation = session.metalearning.adapt(task_id="new-x", partition="validation",
                                                   max_support_per_class=np.int64(2))
    assert adaptation.n_support == 4


@pytest.mark.parametrize("problem", ["empty", "missing_feature", "missing_target", "missing_label",
                                    "missing_task", "mixed_tasks", "conflicting_task"])
def test_explicit_support_refuses_invalid_row_contracts_atomically(simple, problem):
    session = simple
    frame = session.partition("validation")
    support = frame.loc[frame.task == "new-x"].copy()
    options = {}
    if problem == "empty":
        support = support.iloc[:0]
    elif problem == "missing_feature":
        support = support.drop(columns="x")
    elif problem == "missing_target":
        support = support.drop(columns="y")
    elif problem == "missing_label":
        support["y"] = support.y.astype(object)
        support.iloc[0, support.columns.get_loc("y")] = None
    elif problem == "missing_task":
        support.iloc[0, support.columns.get_loc("task")] = None
    elif problem == "mixed_tasks":
        support.iloc[0, support.columns.get_loc("task")] = "new-y"
    else:
        options["task_id"] = "conflicting-external-task"
    prior = session.metalearning.adapt(task_id="new-x", partition="validation")
    before, fingerprint = list(session._history), joblib.hash(session.metalearning_plan)
    untouched = support.copy(deep=True)
    with pytest.raises(ValidationError, match="Empty|missing|null|conflicts|only one"):
        session.metalearning.adapt(support_frame=support, max_support_per_class=2, **options)
    assert session._history == before and session._metalearning_adapt_result is prior
    assert joblib.hash(session.metalearning_plan) == fingerprint
    pd.testing.assert_frame_equal(support, untouched)


def test_anonymous_explicit_support_retains_its_observed_rows_and_classes(simple):
    session = simple
    frame = session.partition("validation")
    support = frame.loc[frame.task == "new-x"].drop(columns="task")
    untouched, fingerprint = support.copy(deep=True), joblib.hash(session.metalearning_plan)
    result = session.metalearning.adapt(support_frame=support, max_support_per_class=2,
                                        random_state=31)
    assert result.task_id is None and result.n_support == 4 and result.n_classes_adapted == 2
    assert result.classes_ == (0, 1)
    if session.metalearning_plan.method == "prototypical":
        rng = np.random.default_rng(31)
        for label in [0, 1]:
            rows = support.loc[support.y == label]
            expected = rows.iloc[rng.choice(len(rows), 2, replace=False)][["x", "z"]].mean().to_numpy()
            np.testing.assert_allclose(result.prototypes_[label], expected)
    assert joblib.hash(session.metalearning_plan) == fingerprint
    pd.testing.assert_frame_equal(support, untouched)
