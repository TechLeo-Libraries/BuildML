"""Public request controls preserve case identity and weighted predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def data(kind="integer", shift=0, task="regression"):
    rng = np.random.default_rng(723)
    x = rng.normal(size=(12, 2))
    labels = list(range(shift, shift + 12))
    if kind == "string":
        labels = [f"review-{i}" for i in labels]
    elif kind == "tuple":
        labels = [("review", i) for i in labels]
    y = 2 * x[:, 0] - 0.4 * x[:, 1] + np.sin(np.arange(12))
    if task == "classification":
        y = np.asarray(["allow", "deny"] * 6)
    return pd.DataFrame(
        {"x": x[:, 0], "z": x[:, 1], "y": y},
        index=pd.Index(labels, tupleize_cols=False),
    )


def session(raw):
    return (
        Session.ingest(raw)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .inject_split(
            train_indices=[7, 2, 5, 0, 6, 1, 4, 3], validation_indices=[8, 9], test_indices=[10, 11]
        )
    )


def fit(s, **options):
    return s.cbr.fit(backend="sklearn", prefer_reduce_components=False, **options)


def check(s, memory, k, eps=1e-8):
    """Use raw distances and explicit votes, independent of CBR helpers."""
    q = s.partition("test")
    # Retention must keep the original eight-case transform.
    fitted = memory.iloc[:8][["x", "z"]].to_numpy()
    mean, scale = fitted.mean(axis=0), fitted.std(axis=0)
    x = (memory[["x", "z"]].to_numpy() - mean) / scale
    query = (q[["x", "z"]].to_numpy() - mean) / scale
    d = np.sqrt(((query[:, None] - x[None]) ** 2).sum(axis=2))
    prediction = s.cbr.predict(partition="test", k=k, return_traces=True)
    retrieval = s.cbr.retrieve(partition="test", k=k)
    expected_predictions, all_distances = [], []
    for i, row in enumerate(d):
        nearest = np.argsort(row, kind="stable")[:k]
        distances = row[nearest]
        weights = 1 / (distances + eps)
        outcomes = memory.y.to_numpy()[nearest]
        if s.cbr_plan.task == "regression":
            answer = np.dot(weights, outcomes) / weights.sum()
        else:
            totals = {label: weights[outcomes == label].sum() for label in np.unique(outcomes)}
            answer = max(totals, key=totals.get)
        expected_predictions.append(answer)
        all_distances.extend(distances)
        for trace in (prediction.traces[i], retrieval.traces[i]):
            assert list(trace.neighbor_row_indices) == memory.index[nearest].tolist()
            np.testing.assert_allclose(trace.distances, distances, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(prediction.traces[i].weights, weights, rtol=1e-11)
    if s.cbr_plan.task == "regression":
        np.testing.assert_allclose(prediction.predictions, expected_predictions, rtol=1e-11)
        expected_mae = np.abs(np.asarray(expected_predictions) - q.y.to_numpy()).mean()
        evaluation = s.cbr.evaluate(partition="test", k=k)
        assert evaluation.metrics["mae"] == pytest.approx(expected_mae)
    else:
        assert list(prediction.predictions) == expected_predictions
        evaluation = s.cbr.evaluate(partition="test", k=k)
        assert evaluation.metrics["accuracy"] == pytest.approx(
            np.mean(np.asarray(expected_predictions) == q.y.to_numpy())
        )
    assert evaluation.mean_neighbor_distance == pytest.approx(np.mean(all_distances))
    json.dumps(prediction.to_dict(), allow_nan=False)
    json.dumps(retrieval.to_dict(), allow_nan=False)
    json.dumps(evaluation.to_dict(), allow_nan=False)


def evidence(s):
    s.cbr.retrieve()
    s.cbr.predict()
    s.cbr.evaluate()
    return (
        s.cbr_plan,
        s._cbr_fit_result,
        s._cbr_retrieve_result,
        s._cbr_predict_result,
        s._cbr_eval_result,
        s._cbr_retain_result,
        s.history,
        s.dataset.to_pandas().copy(deep=True),
        s.cbr_plan.case_base.numeric_matrix.copy(),
    )


def unchanged(s, before):
    plan, fitted, retrieved, predicted, evaluated, retained, history, frame, matrix = before
    assert s.cbr_plan is plan
    assert s._cbr_fit_result is fitted
    assert s._cbr_retrieve_result is retrieved
    assert s._cbr_predict_result is predicted
    assert s._cbr_eval_result is evaluated
    assert s._cbr_retain_result is retained
    assert s.history == history
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), frame)
    np.testing.assert_array_equal(plan.case_base.numeric_matrix, matrix)


BAD_COUNTS = [True, np.bool_(False), 1.8, "3", np.nan, np.inf, -1, 0]


@pytest.mark.parametrize("route", ["fit", "predict", "retrieve", "evaluate"])
@pytest.mark.parametrize(
    "bad",
    BAD_COUNTS,
    ids=["true", "numpy-bool", "fraction", "text", "nan", "inf", "negative", "zero"],
)
def test_invalid_neighbor_counts_preserve_prior_reports_and_case_memory(route, bad):
    s = session(data())
    fit(s, k=3)
    before = evidence(s)
    with pytest.raises(ValidationError, match="k"):
        if route == "fit":
            fit(s, k=bad)
        else:
            getattr(s.cbr, route)(k=bad)
    unchanged(s, before)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("k", [1, np.int64(3), 20], ids=["one", "numpy-integer", "beyond-memory"])
def test_integer_counts_and_omitted_overrides_survive_bundles_and_match_raw_arithmetic(
    task, k, tmp_path
):
    s = session(data(task=task))
    result = fit(s, task=task, k=k)
    assert result.k == int(k) and type(s.cbr_plan.k) is int
    path = s.cbr.save_bundle(tmp_path / "reasoner")
    restored = session(data(task=task))
    restored.cbr.load_bundle(path, trusted=True)
    check(restored, s.partition("train"), int(k))
    assert restored.cbr.predict(k=None).to_dict() == restored.cbr.predict(k=int(k)).to_dict()
    assert restored.cbr.retrieve(k=None).to_dict() == restored.cbr.retrieve(k=int(k)).to_dict()
    assert restored.cbr.evaluate(k=None).to_dict() == restored.cbr.evaluate(k=int(k)).to_dict()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -0.2, True, "0.1", None, 1j])
def test_weight_floor_must_be_finite_positive_numeric_before_refitting_a_saved_plan(bad, tmp_path):
    owner = session(data())
    fit(owner, k=3)
    path = owner.cbr.save_bundle(tmp_path / "valid")
    s = session(data())
    s.cbr.load_bundle(path, trusted=True)
    before = evidence(s)
    with pytest.raises(ValidationError, match="distance_eps"):
        fit(s, k=3, distance_eps=bad)
    unchanged(s, before)


@pytest.mark.parametrize("eps", [np.float32(0.125), 2])
def test_positive_numeric_weight_floors_reach_predictions_and_saved_bundles(eps, tmp_path):
    s = session(data())
    fit(s, k=3, distance_eps=eps)
    path = s.cbr.save_bundle(tmp_path / "reasoner")
    restored = session(data())
    restored.cbr.load_bundle(path, trusted=True)
    assert restored.cbr_plan.distance_eps == float(eps)
    check(restored, s.partition("train"), 3, float(eps))


def selection(labels, container):
    if container == "numpy":
        result = np.empty(len(labels), dtype=object)
        result[:] = labels
        return result
    if container == "index":
        return pd.Index(labels, tupleize_cols=False)
    if container == "series":
        return pd.Series(labels)
    return list(labels)


@pytest.mark.parametrize("kind", ["integer", "string", "tuple"])
@pytest.mark.parametrize("container", ["list", "numpy", "index", "series"])
@pytest.mark.parametrize("allow", [True, False])
def test_indexed_retention_honors_training_overlap_and_exact_label_order_after_loading(
    kind, container, allow, tmp_path
):
    owner = session(data(kind))
    fit(owner, k=3)
    path = owner.cbr.save_bundle(tmp_path / "original")
    s = session(data(kind, shift=1000))
    s.cbr.load_bundle(path, trusted=True)
    mean, scale = (
        s.cbr_plan.case_base.numeric_mean_.copy(),
        s.cbr_plan.case_base.numeric_scale_.copy(),
    )
    selected = s.partition("train").iloc[[1, 0]]
    result = s.cbr.retain(
        row_indices=selection(selected.index.tolist(), container),
        source_disclosure="Reviewed new training outcomes",
        allow_overlap_with_train=allow,
    )
    assert (result.n_added, result.n_skipped) == ((2, 0) if allow else (0, 2))
    memory = pd.concat([owner.partition("train"), selected]) if allow else owner.partition("train")
    assert [c.row_index for c in s.cbr_plan.case_base.cases] == memory.index.tolist()
    np.testing.assert_array_equal(s.cbr_plan.case_base.numeric_mean_, mean)
    np.testing.assert_array_equal(s.cbr_plan.case_base.numeric_scale_, scale)
    check(s, memory, 3)
    saved = s.cbr.save_bundle(tmp_path / "retained")
    restored = session(data(kind, shift=1000))
    restored.cbr.load_bundle(saved, trusted=True)
    check(restored, memory, 3)
    repeated = restored.cbr.retain(
        row_indices=selection(selected.index.tolist(), container),
        source_disclosure="Same reviewed cases",
        allow_overlap_with_train=allow,
    )
    assert (repeated.n_added, repeated.n_skipped) == (0, 2)
    check(restored, memory, 3)


@pytest.mark.parametrize("container", ["numpy", "index", "series"])
@pytest.mark.parametrize("fault", ["empty", "holdout", "unknown", "duplicate", "overlap-flag"])
def test_indexed_retention_refusals_preserve_loaded_memory_and_reports(container, fault, tmp_path):
    owner = session(data())
    fit(owner, k=3)
    path = owner.cbr.save_bundle(tmp_path / "original")
    s = session(data(shift=1000))
    s.cbr.load_bundle(path, trusted=True)
    before = evidence(s)
    labels = {
        "empty": [],
        "holdout": [1000, 1010],
        "unknown": [1000, 9999],
        "duplicate": [1000, 1000],
        "overlap-flag": [1000, 1001],
    }[fault]
    message = {
        "empty": "non-empty",
        "holdout": "validation/test",
        "unknown": "not in dataset",
        "duplicate": "unique|duplicate",
        "overlap-flag": "boolean",
    }[fault]
    with pytest.raises(ValidationError, match=message):
        s.cbr.retain(
            row_indices=selection(labels, container),
            source_disclosure="Reviewed outcomes",
            allow_overlap_with_train="False" if fault == "overlap-flag" else True,
        )
    unchanged(s, before)


def test_zero_label_numpy_selection_is_a_case_identity_and_existing_memory_is_skipped():
    s = session(data())
    fit(s, k=3)
    result = s.cbr.retain(row_indices=np.array([0]), source_disclosure="Same reviewed case")
    assert (result.n_added, result.n_skipped, result.n_cases_after) == (0, 1, 8)
    check(s, s.partition("train"), 3)
