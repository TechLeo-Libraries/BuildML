"""Custom feature callbacks preserve retained columns and fitted output identity."""
from __future__ import annotations

import copy
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.pipeline import load_pipeline_bundle, predict_from_pipeline
from buildml.preprocess import apply_preprocess_plans
from buildml.preprocess.custom import fit_custom_transform, register_transform, unregister_transform

BACKINGS = ["pandas", "polars-memory", "polars-lazy", "duckdb-memory", "duckdb-lazy"]
ROLES = {"x": "feature", "z": "feature", "spare": "feature", "target": "target",
         "row_id": "id", "cohort": "group", "stamp": "time", "weight": "weight"}


def _session(tmp_path, backing):
    i = np.arange(42)
    x = np.linspace(-4, 5, len(i))
    frame = pd.DataFrame({"x": x, "z": np.cos(i / 3), "spare": np.sin(i / 5),
                          "target": 2 + .35*x + .02*x*x + .1*np.cos(i),
                          "row_id": [f"{n:04}" for n in i], "cohort": ["a", "b"]*21,
                          "stamp": pd.date_range("2025-01-01", periods=len(i)),
                          "weight": 1 + (i % 5)*.1}, index=[f"r-{n//2}" for n in i])
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith("lazy"):
        file = tmp_path / "source.parquet"
        frame.to_parquet(file, index=False)
        session = Session.ingest(file, engine=engine, mode="lazy")
        frame = frame.reset_index(drop=True)
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles(ROLES).split(test_size=.25, validation_size=.25, random_state=23)
    return session, frame


@contextmanager
def _registered(operation="add", *, drop=False, declared=False, output=None, broken=None, stage="fit"):
    name = "test_custom_output_contract"
    fits = []
    names = [output] if output else {"add": ["centered"], "replace": ["x"],
                                   "two": ["centered", "square"]}[operation]

    def fit(frame, params):
        fits.append(frame.copy())
        return {"mean": float(frame.x.mean()), "rows": len(frame), "factor": params.get("factor", 1.25)}

    def transform(frame, artifact):
        values = (frame.x - artifact["mean"]) * artifact["factor"]
        result = pd.DataFrame({name: values**(2 if name == "square" else 1) for name in names}, index=frame.index)
        if broken and (stage == "fit" or len(frame) != artifact["rows"]):
            if broken == "array":
                return result.to_numpy()
            if broken == "index":
                return result.reset_index(drop=True)
            if broken == "empty":
                return pd.DataFrame(index=frame.index)
            if broken == "duplicate":
                return pd.concat([result, result], axis=1)
        return result

    declared_names = ["centered", "missing"] if broken == "missing" else (
        ["centered", "centered"] if broken == "duplicate_names" else names)
    register_transform(name, fit=fit, transform=transform, drop_input_columns=drop,
                       output_columns=(lambda artifact, columns: declared_names) if declared else None)
    try:
        yield name, fits, names
    finally:
        unregister_transform(name)


def _manual(frame, mean, names, drop):
    out = frame.copy()
    values = (frame.x - mean) * 1.25
    if drop and "x" not in names:
        out = out.drop(columns="x")
    for name in names:
        out[name] = values**(2 if name == "square" else 1)
    return out


def _ridge_reference(frame, train, features):
    design = np.column_stack([np.ones(len(frame)), frame[features].to_numpy()])
    weights = frame.iloc[train].weight.to_numpy()
    x = design[train]
    beta = np.linalg.solve(x.T @ (weights[:, None]*x) + np.diag([0.] + [.75]*len(features)),
                           x.T @ (weights*frame.iloc[train].target.to_numpy()))
    return beta


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("operation", ["add", "replace", "two"])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_fitted_custom_features_and_saved_predictions_match_weighted_oracle(backing, operation, drop, declared, tmp_path):
    session, frame = _session(tmp_path, backing)
    train = list(session.split_plan.train_indices)
    mean = float(frame.iloc[train].x.mean())
    with _registered(operation, drop=drop, declared=declared) as (name, fits, names):
        try:
            session.apply_custom_transform(name, columns=["x"], params={"factor": 1.25})
            assert len(fits) == 1 and list(fits[0]) == ["x"]
            np.testing.assert_allclose(fits[0].x, frame.iloc[train].x)
            expected = _manual(frame, mean, names, drop)
            pd.testing.assert_frame_equal(session.to_pandas(), expected, check_dtype=False)
            assert session.custom_plan.artifact_["mean"] == pytest.approx(mean)
            for col in ["target", "row_id", "cohort", "stamp", "weight", "spare"]:
                assert session.dataset.roles[col].value == ROLES[col]
            plan_hash = joblib.hash(session.custom_plan)
            session.fit(Ridge(alpha=.75), task="regression")
            features = list(session.fit_result.feature_columns)
            expected_features = {"x", "z", "spare", *names}
            if drop and "x" not in names:
                expected_features.remove("x")
            assert set(features) == expected_features
            beta = _ridge_reference(expected, train, features)
            for partition in ["validation", "test"]:
                positions = list(getattr(session.split_plan, partition+"_indices"))
                design = np.column_stack([np.ones(len(positions)), expected.iloc[positions][features]])
                np.testing.assert_allclose(session.fit_result.estimator.predict(session.partition(partition)[features]),
                                           design @ beta, atol=3e-10)
            path = session.save_pipeline(tmp_path / "model", evaluate_partition=None)
            bundle = load_pipeline_bundle(path, trusted=True)
            bundle_hash = joblib.hash(bundle)
            score = frame.drop(columns="target").iloc[[3, 1, 7, 2]].copy()
            score["x"] = [-15., 10., .2, 3.]
            transformed = _manual(score, mean, names, drop)
            reference = np.column_stack([np.ones(len(score)), transformed[features]]) @ beta
            for source in [bundle, path]:
                result = predict_from_pipeline(source, score, trusted=source == path)
                np.testing.assert_allclose(result.predictions, reference, atol=3e-10)
                assert result.predictions.index.equals(score.index)
            prepared = predict_from_pipeline(bundle, transformed, apply_plans=False)
            np.testing.assert_allclose(prepared.predictions, reference, atol=3e-10)
            assert len(fits) == 1 and joblib.hash(bundle) == bundle_hash
            assert joblib.hash(session.custom_plan) == plan_hash
        finally:
            session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("column", ["target", "row_id", "cohort", "stamp", "weight", "spare"])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_custom_outputs_cannot_overwrite_retained_fields(backing, column, drop, declared, tmp_path):
    session, frame = _session(tmp_path, backing)
    session.fit(Ridge(alpha=.75), task="regression")
    fitted, history = session.fit_result, copy.deepcopy(session.history)
    roles, split = dict(session.dataset.roles), session.split_plan.to_dict()
    prediction = fitted.estimator.predict(frame[list(fitted.feature_columns)])
    with _registered(drop=drop, declared=declared, output=column) as (name, _, _):
        try:
            with pytest.raises(ValidationError, match="Custom.*(collid|overwrite|retained)"):
                session.apply_custom_transform(name, columns=["x"])
            assert session.history == history and session.fit_result is fitted and session.custom_plan is None
            assert session.dataset.roles == roles and session.split_plan.to_dict() == split
            pd.testing.assert_frame_equal(session.to_pandas(), frame, check_dtype=False)
            np.testing.assert_array_equal(fitted.estimator.predict(frame[list(fitted.feature_columns)]), prediction)
        finally:
            session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("route", ["apply", "pipeline"])
@pytest.mark.parametrize("output_role", ["feature", "id"])
def test_replay_refuses_to_replace_a_new_caller_column(backing, route, output_role, tmp_path):
    fitted, frame = _session(tmp_path, "pandas")
    with _registered() as (name, _, _):
        fitted.apply_custom_transform(name, columns=["x"]).fit(Ridge(alpha=.75), task="regression")
        path = fitted.save_pipeline(tmp_path / "pipeline", evaluate_partition=None)
        bundle = load_pipeline_bundle(path, trusted=True)
        receiver, incoming = _session(tmp_path, backing)
        try:
            incoming = incoming.assign(centered=np.arange(len(incoming)) + 10_000.)
            # Add the caller's field before native ingest, without changing the fitted artifact.
            receiver.close_native()
            engine = backing.split("-")[0]
            if backing.endswith("lazy"):
                file = tmp_path / "incoming.parquet"
                incoming.to_parquet(file, index=False)
                receiver = Session.ingest(file, engine=engine, mode="lazy")
            else:
                receiver = Session.ingest(incoming, engine=engine)
            receiver.set_roles({**ROLES, "centered": output_role})
            before_plan, before_bundle = joblib.hash(fitted.custom_plan), joblib.hash(bundle)
            roles = dict(receiver.dataset.roles)
            with pytest.raises(ValidationError, match="Custom.*retained"):
                if route == "apply":
                    apply_preprocess_plans(receiver.dataset, custom_plan=fitted.custom_plan,
                                           roles={"centered": output_role})
                else:
                    predict_from_pipeline(bundle, receiver.dataset, roles={"centered": output_role})
            assert receiver.dataset.roles == roles
            pd.testing.assert_frame_equal(receiver.to_pandas(), incoming, check_dtype=False)
            assert joblib.hash(fitted.custom_plan) == before_plan and joblib.hash(bundle) == before_bundle
        finally:
            receiver.close_native()


@pytest.mark.parametrize("stage,broken", [("fit", kind) for kind in ["array", "index", "empty", "duplicate", "missing", "duplicate_names"]]
                         + [("replay", kind) for kind in ["array", "index", "empty", "duplicate"]])
def test_custom_callback_output_contract_errors_are_explicit(stage, broken, tmp_path):
    session, frame = _session(tmp_path, "pandas")
    with _registered(declared=broken in {"missing", "duplicate_names"}, broken=broken, stage=stage) as (name, _, _):
        if stage == "fit":
            with pytest.raises(ValidationError, match="Custom transform"):
                fit_custom_transform(session.dataset, session.split_plan, name=name, columns=["x"])
        else:
            plan = fit_custom_transform(session.dataset, session.split_plan, name=name, columns=["x"])
            score = frame.iloc[:3].copy()
            before, artifact = score.copy(), joblib.hash(plan)
            with pytest.raises(ValidationError, match="Custom transform"):
                apply_preprocess_plans(score, custom_plan=plan)
            pd.testing.assert_frame_equal(score, before)
            assert joblib.hash(plan) == artifact
