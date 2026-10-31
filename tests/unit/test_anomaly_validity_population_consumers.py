"""Anomaly decisions retain valid bounds and refuse undefined populations."""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import OneClassSVM

from buildml import Session
from buildml.core.errors import ValidationError

BACKINGS = [
    "pandas",
    "pandas-parquet",
    "polars-memory",
    "polars-lazy",
    "duckdb-memory",
    "duckdb-lazy",
]
FAMILIES = ["float", "object", "string", "category", "boolean", "integer"]
TRAIN = [*range(1, 64, 2), *range(0, 64, 2)]
POLICIES = [
    "contamination",
    "quantile",
    "decision_zero",
    "absolute",
    "positive_inf",
    "negative_inf",
    "numeric_string",
]
BAD_THRESHOLDS = [
    pytest.param(np.nan, id="numeric_nan"),
    pytest.param("NaN", id="text_nan"),
    pytest.param(1j, id="complex"),
    pytest.param(10**1000, id="overflow"),
]


def _frame(family="string"):
    rng = np.random.default_rng(519)
    values = rng.normal(size=(96, 2))
    unusual = np.arange(96) % 4 == 0
    values[unusual] += [2.0, -0.7]
    normal = "normal" if family in {"object", "string", "category"} else 0
    if family in {"object", "string", "category"}:
        label = pd.Series(np.where(unusual, "odd", "normal"), dtype=family)
    else:
        label = pd.Series(
            unusual, dtype={"float": "float64", "boolean": "boolean", "integer": "Int64"}[family]
        )
    frame = pd.DataFrame(
        dict(x=values[:, 0], z=values[:, 1], label=label, row_id=[f"{i:04}" for i in range(96)])
    )
    frame.index = [f"row-{i // 2}" for i in range(96)]
    return frame, normal


def _session(frame, tmp_path, backing="pandas", target=True):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("parquet", "lazy")):
        path = tmp_path / "rows.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles(
        {"x": "feature", "z": "feature", "label": "target" if target else "ignore", "row_id": "id"}
    )
    return session.inject_split(
        train_indices=TRAIN,
        validation_indices=list(range(64, 80)),
        test_indices=list(range(80, 96)),
    )


def _fit_options(mode="unsupervised", normal="normal"):
    return dict(
        method="one_class_svm",
        mode=mode,
        nu=0.1,
        kernel="rbf",
        gamma="scale",
        contamination=0.2,
        normal_label_column="label",
        normal_label_value=normal,
        prefer_reduce_components=False,
        random_state=7,
    )


def _rbf(model, frame):
    rows = frame[["x", "z"]].to_numpy()
    distance = ((rows[:, None, :] - model.support_vectors_[None, :, :]) ** 2).sum(axis=2)
    return -(np.exp(-model._gamma * distance) @ model.dual_coef_[0] + model.intercept_[0])


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    return ordered[lower] + (position - lower) * (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    )


def _reference(session, mode, normal):
    train = session.partition("train")
    fit = train if mode == "unsupervised" else train.loc[train.label.eq(normal)]
    return OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(fit[["x", "z"]].to_numpy()), len(
        fit
    )


def _snapshot(session):
    return (
        session.dataset.to_pandas().copy(deep=True),
        session.split_plan,
        session.anomaly_plan,
        joblib.hash(session.anomaly_plan),
        session.anomaly_score_result,
        session.anomaly_eval_result,
        copy.deepcopy(session.history),
    )


def _unchanged(session, before):
    frame, split, plan, digest, score, evaluation, history = before
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), frame)
    assert session.split_plan is split and session.anomaly_plan is plan
    assert joblib.hash(plan) == digest and session.anomaly_score_result is score
    assert session.anomaly_eval_result is evaluation and session.history == history


@pytest.fixture(scope="module")
def control_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("anomaly-validity-control")
    frame, normal = _frame()
    session = _session(frame, root)
    session.anomaly.fit(**_fit_options(normal=normal))
    return session.anomaly.save_bundle(root / "detector")


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("mode", ["unsupervised", "novelty"])
@pytest.mark.parametrize("policy", POLICIES)
def test_valid_decisions_match_rbf_quantiles_saved_plans_and_aligned_attachments(
    backing, mode, policy, tmp_path
):
    frame, normal = _frame()
    session = _session(frame, tmp_path, backing)
    original_dataset = session.dataset
    try:
        before, split = session.to_pandas().copy(), session.split_plan
        extra = dict(threshold_policy=policy, quantile=0.3)
        if policy in {"absolute", "positive_inf", "negative_inf", "numeric_string"}:
            extra = dict(
                threshold_policy="score_threshold",
                score_threshold={
                    "absolute": 0.1,
                    "positive_inf": np.inf,
                    "negative_inf": -np.inf,
                    "numeric_string": "0.1",
                }[policy],
            )
        result = session.anomaly.fit(**_fit_options(mode, normal), **extra)
        reference, n_fit = _reference(session, mode, normal)
        plan = session.anomaly_plan
        assert result.n_fit_rows == n_fit and plan.n_train_rows == 64
        np.testing.assert_allclose(plan.estimator_.support_vectors_, reference.support_vectors_)
        train_scores = _rbf(reference, session.partition("train"))
        if policy == "contamination":
            expected = _quantile(train_scores, 0.8)
        elif policy == "quantile":
            expected = _quantile(train_scores, 0.7)
        else:
            expected = {
                "decision_zero": 0.0,
                "absolute": 0.1,
                "positive_inf": np.inf,
                "negative_inf": -np.inf,
                "numeric_string": 0.1,
            }[policy]
        assert plan.threshold_ == pytest.approx(expected)
        path = session.anomaly.save_bundle(tmp_path / "detector")
        session.anomaly.load_bundle(path, trusted=True)
        plan = session.anomaly_plan
        frozen = joblib.hash(plan)
        test = session.partition("test")
        scores = _rbf(reference, test)
        actual = session.anomaly.score(partition="test")
        np.testing.assert_allclose(actual.scores, scores, atol=2e-12)
        np.testing.assert_array_equal(actual.flags, scores >= expected)
        for override in [-np.inf, np.inf, "0.1"]:
            selected = session.anomaly.score(partition="test", override_threshold=override)
            np.testing.assert_array_equal(selected.flags, scores >= float(override))
        attached = session.anomaly.score(partition="all", attach=True)
        all_scores = _rbf(reference, before)
        np.testing.assert_allclose(attached.scores, all_scores, atol=2e-12)
        after = session.to_pandas()
        pd.testing.assert_frame_equal(after[list(before)], before)
        np.testing.assert_array_equal(after[plan.flag_column], all_scores >= expected)
        np.testing.assert_allclose(after[plan.score_column], all_scores, atol=2e-12)
        assert session.split_plan is split and joblib.hash(plan) == frozen
    finally:
        if original_dataset is not session.dataset:
            original_dataset.close_native()
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("family", FAMILIES)
def test_novelty_training_null_labels_refuse_without_replacing_detector_state(
    backing, family, control_bundle, tmp_path
):
    frame, normal = _frame(family)
    frame.iloc[7, frame.columns.get_loc("label")] = (
        None if family in {"object", "category"} else pd.NA
    )
    session = _session(frame, tmp_path, backing)
    try:
        session.anomaly.load_bundle(control_bundle, trusted=True)
        session.anomaly.score(partition="test")
        before = _snapshot(session)
        with pytest.raises(ValidationError, match="null"):
            session.anomaly.fit(**_fit_options("novelty", normal))
        _unchanged(session, before)
    finally:
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("family", FAMILIES)
def test_unsupervised_fit_ignores_unused_missing_labels(backing, family, tmp_path):
    frame, normal = _frame(family)
    frame.iloc[7, frame.columns.get_loc("label")] = (
        None if family in {"object", "category"} else pd.NA
    )
    session = _session(frame, tmp_path, backing)
    try:
        session.anomaly.fit(**_fit_options(normal=normal))
        reference, n_fit = _reference(session, "unsupervised", normal)
        assert session.anomaly_plan.n_fit_rows == n_fit == 64
        actual = session.anomaly.score(partition="test")
        np.testing.assert_allclose(
            actual.scores, _rbf(reference, session.partition("test")), atol=2e-12
        )
    finally:
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("threshold", BAD_THRESHOLDS)
@pytest.mark.parametrize("route", ["fit", "score", "evaluate"])
def test_undefined_thresholds_refuse_before_attachment_or_state_changes(
    backing, threshold, route, control_bundle, tmp_path
):
    frame, normal = _frame()
    session = _session(frame, tmp_path, backing)
    try:
        session.anomaly.load_bundle(control_bundle, trusted=True)
        session.anomaly.score(partition="test")
        before = _snapshot(session)
        with pytest.raises(ValidationError, match="threshold"):
            if route == "fit":
                session.anomaly.fit(
                    **_fit_options(normal=normal),
                    threshold_policy="score_threshold",
                    score_threshold=threshold,
                )
            elif route == "score":
                session.anomaly.score(partition="all", attach=True, override_threshold=threshold)
            else:
                session.anomaly.evaluate(partition="test", override_threshold=threshold)
        _unchanged(session, before)
    finally:
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize(
    "problem", ["no_target", "missing_column", "few_normals", "bad_decision_method"]
)
def test_invalid_fit_populations_and_policies_preserve_existing_results(
    backing, problem, control_bundle, tmp_path
):
    frame, normal = _frame()
    if problem == "few_normals":
        frame.iloc[:3, frame.columns.get_loc("label")] = "rare"
    session = _session(frame, tmp_path, backing, target=problem != "no_target")
    try:
        session.anomaly.load_bundle(control_bundle, trusted=True)
        session.anomaly.score(partition="test")
        options = _fit_options("novelty", normal)
        if problem == "no_target":
            options["normal_label_column"] = None
        elif problem == "missing_column":
            options["normal_label_column"] = "absent"
        elif problem == "few_normals":
            options["normal_label_value"] = "rare"
        else:
            options.update(
                method="isolation_forest",
                mode="unsupervised",
                threshold_policy="decision_zero",
                n_estimators=12,
            )
        before = _snapshot(session)
        with pytest.raises(ValidationError):
            session.anomaly.fit(**options)
        _unchanged(session, before)
    finally:
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("family", FAMILIES)
def test_novelty_uses_only_training_labels_with_missing_holdout_outcomes(backing, family, tmp_path):
    frame, normal = _frame(family)
    frame.iloc[64:, frame.columns.get_loc("label")] = (
        None if family in {"object", "category"} else pd.NA
    )
    session = _session(frame, tmp_path, backing)
    try:
        reference, n_fit = _reference(session, "novelty", normal)
        session.anomaly.fit(**_fit_options("novelty", normal))
        assert session.anomaly_plan.n_fit_rows == n_fit
        np.testing.assert_allclose(
            session.anomaly.score(partition="test").scores,
            _rbf(reference, session.partition("test")),
            atol=2e-12,
        )
        assert session.to_pandas().label.iloc[64:].isna().all()
    finally:
        session.close_native()
