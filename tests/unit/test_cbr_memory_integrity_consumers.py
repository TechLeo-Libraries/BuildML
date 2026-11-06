"""Retained memory and ANN distance weighting agree with raw case arithmetic."""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import buildml
from buildml import Session
from buildml.cbr.extras import cbr_industry_available
from buildml.cbr.predict import predict_cbr
from buildml.core.errors import ValidationError


def frame(task="regression", shift=0):
    rng = np.random.default_rng(801)
    x = rng.uniform(-2, 4, size=(12, 2))
    y = 1.3 + 2 * x[:, 0] - 0.7 * x[:, 1] + np.sin(np.arange(12))
    if task == "classification_text":
        y = np.asarray(["deny", "grant"] * 6)
    elif task == "classification_numeric":
        y = np.asarray([-3, 20] * 6)
    return pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y}, index=np.arange(12) + 100 + shift)


def session(raw):
    return (
        Session.ingest(raw)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .inject_split(
            train_indices=[7, 2, 5, 0, 6, 1, 4, 3], validation_indices=[8, 9], test_indices=[10, 11]
        )
    )


def close(s):
    plan = s.cbr_plan
    index = None if plan is None else plan.case_base.ann_index_
    if index is not None and callable(getattr(index, "close", None)):
        index.close()


@contextmanager
def fitted(backend="sklearn", task="regression", metric="euclidean", standardize=False, raw=None):
    if backend == "industry" and not cbr_industry_available():
        pytest.skip("Real ANN library absent in this environment")
    s = session(frame(task) if raw is None else raw)
    try:
        s.cbr.fit(
            backend=backend,
            task="regression" if task == "regression" else "classification",
            metric=metric,
            standardize=standardize,
            prefer_reduce_components=False,
            k=3,
        )
        yield s
    finally:
        close(s)


def raw_predictions(memory, queries, task, metric, mean=None, scale=None):
    x = memory[["x", "z"]].to_numpy(dtype=float)
    q = queries[["x", "z"]].to_numpy(dtype=float)
    if mean is not None:
        x, q = (x - mean) / scale, (q - mean) / scale
    if metric == "euclidean":
        distances = np.sqrt(((q[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    else:
        distances = 1 - np.clip(
            q @ x.T / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(x, axis=1)), -1, 1
        )
    answers = []
    for row in distances:
        nearest = np.argsort(row, kind="stable")[:3]
        weights = 1 / (row[nearest] + 1e-8)
        solutions = memory.y.to_numpy()[nearest]
        if task == "regression":
            answer = np.average(solutions, weights=weights)
        else:
            totals = {label: weights[solutions == label].sum() for label in np.unique(solutions)}
            answer = max(totals, key=totals.get)
        answers.append((answer, memory.index.to_numpy()[nearest], row[nearest], weights))
    return answers


def check_predictions(s, memory, queries, task, metric, mean, scale):
    expected = raw_predictions(memory, queries, task, metric, mean, scale)
    result = s.cbr.predict(partition="test", return_traces=True)
    retrieved = s.cbr.retrieve(partition="test", k=3)
    for trace, retrieval, (answer, indices, distances, weights) in zip(
        result.traces, retrieved.traces, expected, strict=True
    ):
        np.testing.assert_array_equal(trace.neighbor_row_indices, indices)
        np.testing.assert_array_equal(retrieval.neighbor_row_indices, indices)
        np.testing.assert_allclose(trace.distances, distances, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(retrieval.distances, distances, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(trace.weights, weights, rtol=3e-4, atol=2e-6)
        if task == "regression":
            assert trace.prediction == pytest.approx(answer, rel=3e-4, abs=2e-5)
        else:
            assert trace.prediction == answer
    json.dumps(result.to_dict(), allow_nan=False)
    return result


@pytest.mark.parametrize("backend", ["sklearn", "industry"])
@pytest.mark.parametrize("task", ["regression", "classification_text", "classification_numeric"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("standardize", [False, True])
def test_native_and_ann_memory_predictions_retention_and_saved_bundles_match_raw_arithmetic(
    backend, task, metric, standardize, tmp_path
):
    with fitted(backend, task, metric, standardize) as s:
        original = s.dataset.to_pandas().copy(deep=True)
        train, queries = s.partition("train"), s.partition("test")
        values = train[["x", "z"]].to_numpy()
        mean, scale = (values.mean(axis=0), values.std(axis=0)) if standardize else (None, None)
        before = check_predictions(s, train, queries, task, metric, mean, scale)
        if backend == "industry" and s.cbr_plan.case_base.ann_library_ == "isolated":
            import psutil

            worker = s.cbr_plan.case_base.ann_index_._process
            assert (
                Path(psutil.Process(worker.pid).cwd()).resolve()
                == Path(buildml.__file__).resolve().parents[1]
            )
        added = queries.iloc[[0]].copy()
        added.index = [1000]
        origin = mean if standardize else np.zeros(2)
        if metric == "cosine":
            # A nonzero angular gap avoids inverse-weight sensitivity to float32 rounding.
            units = scale if standardize else np.ones(2)
            direction = (queries[["x", "z"]].iloc[0].to_numpy() - origin) / units
            angle = 0.05
            rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            added.loc[:, ["x", "z"]] = origin + units * (rotation @ direction)
        else:
            added.loc[:, ["x", "z"]] = origin + 0.99 * (
                queries[["x", "z"]].iloc[0].to_numpy() - origin
            )
        added["y"] = (
            17.0 if task == "regression" else "grant" if task == "classification_text" else 20
        )
        retained = s.cbr.retain(
            labeled_frame=added,
            source_disclosure="Reviewed external outcomes — not Session holdout",
        )
        assert retained.n_added == 1 and retained.n_cases_after == len(train) + 1
        assert s.cbr_plan.case_base.n_retained == 1
        if standardize:
            np.testing.assert_array_equal(s.cbr_plan.case_base.numeric_mean_, mean)
            np.testing.assert_array_equal(s.cbr_plan.case_base.numeric_scale_, scale)
        memory = pd.concat([train, added])
        after = check_predictions(s, memory, queries, task, metric, mean, scale)
        assert after.traces[0].neighbor_row_indices[0] == 1000
        assert before.traces[0].neighbor_row_indices != after.traces[0].neighbor_row_indices
        path = s.cbr.save_bundle(tmp_path / "memory")
        restored = session(original)
        try:
            restored.cbr.load_bundle(path, trusted=True)
            repeated = check_predictions(restored, memory, queries, task, metric, mean, scale)
            assert repeated.predictions == after.predictions
            assert restored.cbr_plan.case_base.n_cases == len(memory)
            assert restored.cbr_plan.case_base.n_retained == 1
        finally:
            close(restored)
        pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


def state(s):
    s.cbr.predict(partition="test", return_traces=True)
    s.cbr.evaluate(partition="test")
    plan = s.cbr_plan
    return (
        plan,
        plan.case_base.numeric_matrix.copy(),
        plan.case_base.search_matrix_.copy(),
        s._cbr_predict_result,
        s._cbr_eval_result,
        s.history,
        s.dataset.to_pandas().copy(deep=True),
    )


def unchanged(s, before):
    plan, numeric, search, prediction, evaluation, history, data = before
    assert (
        s.cbr_plan is plan
        and s._cbr_predict_result is prediction
        and s._cbr_eval_result is evaluation
    )
    assert s.history == history
    np.testing.assert_array_equal(plan.case_base.numeric_matrix, numeric)
    np.testing.assert_array_equal(plan.case_base.search_matrix_, search)
    index = plan.case_base.ann_index_
    if index is not None:
        count = (
            index.ntotal if plan.case_base.ann_library_ != "hnswlib" else index.get_current_count()
        )
        assert count == plan.case_base.n_cases
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), data)
    repeat = predict_cbr(s.dataset, plan, s.split_plan, partition="test", return_traces=True)
    assert repeat.predictions == prediction.predictions


@pytest.mark.parametrize("route", ["fit", "retain"])
@pytest.mark.parametrize(
    "bad",
    [np.inf, -np.inf, 1 + 2j, "1.5", None],
    ids=["infinite", "negative-infinite", "complex", "numeric-string", "null"],
)
def test_regression_outcomes_are_finite_real_numeric_before_case_memory_changes(route, bad):
    if route == "fit":
        raw = frame()
        if isinstance(bad, (complex, str)):
            raw["y"] = raw.y.astype(complex if isinstance(bad, complex) else object)
        raw.loc[100, "y"] = bad
        s = session(raw)
        history = s.history
        with pytest.raises(ValidationError, match="numeric|finite|real|null"):
            s.cbr.fit(task="regression", backend="sklearn", prefer_reduce_components=False)
        assert s.cbr_plan is None and s.history == history
    else:
        with fitted() as s:
            before = state(s)
            added = pd.DataFrame(
                {"x": [1.0, 2.0], "z": [3.0, 4.0], "y": [4.0, bad]}, index=[1000, 1001]
            )
            with pytest.raises(ValidationError, match="numeric|finite|real|null"):
                s.cbr.retain(labeled_frame=added, source_disclosure="Reviewed external cases")
            unchanged(s, before)


BAD_DISCLOSURES = [None, True, 0, [], {}, "", " \t"]


@pytest.mark.parametrize("disclosure", BAD_DISCLOSURES)
def test_provenance_requires_a_nonempty_string_and_preserves_memory(disclosure):
    with fitted() as s:
        before = state(s)
        with pytest.raises(ValidationError, match="source_disclosure"):
            s.cbr.retain(
                labeled_frame=frame().iloc[[0]].set_axis([1000]), source_disclosure=disclosure
            )
        unchanged(s, before)


@pytest.mark.parametrize("identity", [1000, "external", ("batch", 2)])
def test_duplicate_retained_identities_are_ambiguous_and_refused_atomically(identity):
    with fitted() as s:
        before = state(s)
        added = frame().iloc[:2].copy()
        added.index = pd.Index([identity, identity], tupleize_cols=False)
        with pytest.raises(ValidationError, match="unique|duplicate"):
            s.cbr.retain(labeled_frame=added, source_disclosure="Reviewed external cases")
        unchanged(s, before)


@pytest.mark.parametrize("value", ["False", 0, 1, None])
def test_retention_overlap_control_refuses_ambiguous_truth_values(value):
    with fitted() as s:
        before = state(s)
        with pytest.raises(ValidationError, match="boolean"):
            s.cbr.retain(
                labeled_frame=frame().iloc[[0]].set_axis([1000]),
                source_disclosure="Reviewed external cases",
                allow_overlap_with_train=value,
            )
        unchanged(s, before)


@pytest.mark.parametrize(
    "fault",
    ["no_provenance", "duplicate_ids", "overlap_flag", "infinite_target", "missing_feature"],
)
def test_ann_refusals_leave_live_index_predictions_and_reports_intact(fault):
    with fitted("industry") as s:
        before = state(s)
        added = frame().iloc[:2].copy().set_axis([1000, 1001])
        options = {"source_disclosure": "Reviewed external cases"}
        if fault == "no_provenance":
            options["source_disclosure"] = None
        elif fault == "duplicate_ids":
            added.index = [1000, 1000]
        elif fault == "overlap_flag":
            options["allow_overlap_with_train"] = "False"
        elif fault == "infinite_target":
            added.loc[1001, "y"] = np.inf
        else:
            added = added.drop(columns="x")
        with pytest.raises(ValidationError):
            s.cbr.retain(labeled_frame=added, **options)
        unchanged(s, before)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "missing_solution",
        "missing_feature",
        "null_solution",
        "non_numeric_feature",
        "infinite_feature",
        "complex_feature",
        "null_feature",
        "holdout",
        "unknown_index",
    ],
)
def test_retention_schema_and_holdout_refusals_preserve_all_prior_evidence(fault):
    with fitted() as s:
        before = state(s)
        added = frame().iloc[:2].copy().set_axis([1000, 1001])
        if fault == "unknown_index":
            with pytest.raises(ValidationError, match="not in dataset"):
                s.cbr.retain(row_indices=["absent"], source_disclosure="Reviewed")
        else:
            if fault == "empty":
                added = added.iloc[:0]
            elif fault == "missing_solution":
                added = added.drop(columns="y")
            elif fault == "missing_feature":
                added = added.drop(columns="x")
            elif fault == "null_solution":
                added.loc[1001, "y"] = np.nan
            elif fault == "non_numeric_feature":
                added["x"] = ["bad", "values"]
            elif fault == "infinite_feature":
                added.loc[1001, "x"] = np.inf
            elif fault == "complex_feature":
                added["x"] = added.x.astype(complex)
                added.loc[1001, "x"] += 2j
            elif fault == "null_feature":
                added.loc[1001, "x"] = np.nan
            else:
                added.index = [s.partition("test").index[0], 1001]
                before = state(s)
            with pytest.raises(ValidationError):
                s.cbr.retain(labeled_frame=added, source_disclosure="Reviewed external cases")
        unchanged(s, before)


@pytest.mark.parametrize("backend", ["sklearn", "industry"])
@pytest.mark.parametrize(
    "allow",
    [True, False, np.bool_(True), np.bool_(False)],
    ids=["true", "false", "numpy-true", "numpy-false"],
)
def test_loaded_bundle_can_admit_or_skip_new_dataset_training_rows_without_refitting_transforms(
    backend, allow, tmp_path
):
    with fitted(backend, standardize=True) as owner:
        path = owner.cbr.save_bundle(tmp_path / "original")
        raw = frame(shift=1000)
        raw["x"] += 0.26
        raw["z"] -= 0.33
        raw["y"] += 1.7
        newcomer = session(raw)
        try:
            newcomer.cbr.load_bundle(path, trusted=True)
            old_mean = newcomer.cbr_plan.case_base.numeric_mean_.copy()
            old_scale = newcomer.cbr_plan.case_base.numeric_scale_.copy()
            selected = newcomer.partition("train").iloc[:2]
            result = newcomer.cbr.retain(
                labeled_frame=selected,
                source_disclosure="Reviewed new training outcomes",
                allow_overlap_with_train=allow,
            )
            assert (result.n_added, result.n_skipped) == ((2, 0) if allow else (0, 2))
            assert newcomer.cbr_plan.case_base.n_cases == 8 + (2 if allow else 0)
            np.testing.assert_array_equal(newcomer.cbr_plan.case_base.numeric_mean_, old_mean)
            np.testing.assert_array_equal(newcomer.cbr_plan.case_base.numeric_scale_, old_scale)
            memory = (
                pd.concat([owner.partition("train"), selected])
                if allow
                else owner.partition("train")
            )
            check_predictions(
                newcomer,
                memory,
                newcomer.partition("test"),
                "regression",
                "euclidean",
                old_mean,
                old_scale,
            )
        finally:
            close(newcomer)


@pytest.mark.parametrize("backend", ["sklearn", "industry"])
@pytest.mark.parametrize("task", ["regression", "classification_text"])
def test_repeated_retention_skips_existing_case_identities_without_changing_predictions(
    backend, task
):
    with fitted(backend, task) as s:
        added = frame(task).iloc[:2].copy().set_axis([1000, 1001])
        first = s.cbr.retain(labeled_frame=added, source_disclosure="Reviewed production outcomes")
        assert first.n_added == 2
        before = predict_cbr(
            s.dataset, s.cbr_plan, s.split_plan, partition="test", return_traces=True
        )
        again = s.cbr.retain(labeled_frame=added, source_disclosure="Same reviewed outcomes")
        assert (again.n_added, again.n_skipped, again.n_cases_after) == (0, 2, 10)
        assert s.cbr_plan.case_base.n_retained == 2
        after = predict_cbr(
            s.dataset, s.cbr_plan, s.split_plan, partition="test", return_traces=True
        )
        assert before.to_dict() == after.to_dict()
