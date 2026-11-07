"""Audit readiness follows live checkpoint state and canonical operation names."""

import copy
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.session.facade_registry import DOMAIN_FACADES


def _session(*, target_slope=2.5, classification=False):
    x = np.linspace(-2, 3, 48)
    y = np.where(x > .4, "warm", "cold") if classification else target_slope*x - 3 + np.cos(x)
    frame = pd.DataFrame({"node": np.arange(48), "x": x, "z": np.sin(x), "y": y})
    session = Session.ingest(frame).set_roles({"node": "id", "x": "feature", "z": "feature", "y": "target"})
    return session.inject_split(train_indices=list(range(32)), validation_indices=list(range(32, 40)),
                                test_indices=list(range(40, 48)))


def _fitted_classical():
    session = _session()
    session.classical.fit(Ridge(alpha=1.3), task="regression")
    train = session.partition("train")
    x = np.column_stack([np.ones(len(train)), train[["x", "z"]].to_numpy()])
    beta = np.linalg.solve(x.T @ x + np.diag([0, 1.3, 1.3]), x.T @ train.y.to_numpy())
    test = session.partition("test")
    expected = np.column_stack([np.ones(len(test)), test[["x", "z"]].to_numpy()]) @ beta
    np.testing.assert_allclose(session.classical.predict(partition="test"), expected, atol=2e-13)
    session.classical.evaluate(partition="test")
    return session


ALIASES = [(domain, method, DOMAIN_FACADES[domain]["bindings"][method]) for domain, method in [
    ("classical", "fit"), ("classical", "evaluate"), ("graph", "fit"), ("graph", "evaluate"),
    ("preprocess", "scale"), ("preprocess", "impute"), ("forecast", "fit"), ("forecast", "evaluate"),
    ("audit", "summarize"), ("audit", "dry_run"), ("online", "fit"), ("synthetic", "fit"),
]]


@pytest.mark.parametrize("domain,method,canonical", ALIASES)
@pytest.mark.parametrize("prefix", ["", "session."])
@pytest.mark.parametrize("stage", ["empty", "split", "evaluated"])
def test_facade_preview_matches_flat_readiness_parameters_and_prerequisite_graph(domain, method, canonical, prefix, stage):
    session = Session() if stage == "empty" else _fitted_classical() if stage == "evaluated" else _session()
    before = joblib.hash((session.history, session.split_plan, session.fit_result))
    parameters = {"method": "minmax"} if canonical == "scale" else {}
    flat = session.audit.dry_run(canonical, parameters=parameters)
    qualified = session.audit.dry_run(f"{prefix}{domain}.{method}", parameters=parameters)
    assert qualified.steps[0].to_dict() == flat.steps[0].to_dict()
    assert qualified.prerequisite_graph.to_dict() == flat.prerequisite_graph.to_dict()
    assert joblib.hash((session.history, session.split_plan, session.fit_result)) == before


@pytest.mark.parametrize("domain", ["classical", "graph"])
@pytest.mark.parametrize("route", ["load", "reattach"])
def test_historical_evaluation_is_blocked_until_actual_model_bundle_is_restored(domain, route, tmp_path, capsys):
    if domain == "classical":
        original = _fitted_classical()
        saved = original.classical.save_pipeline(tmp_path / "model")
        expected = np.asarray(original.classical.predict(partition="test"))
        operations = ["predict", "evaluate", "save_pipeline"]
    else:
        original = _session(classification=True)
        original.graph.set_spec([(i, i+1) for i in range(47)], node_id_col="node")
        original.graph.fit(method="classical", include_graph_metrics=False)
        original.graph.evaluate(partition="test")
        saved = original.graph.save_bundle(tmp_path / "model")
        expected = np.asarray(original.graph.predict(partition="test").probabilities)
        operations = ["fit_graph", "predict_graph", "evaluate_graph", "save_graph_bundle"]
    checkpoint = original.checkpoint_save(tmp_path / "checkpoint")
    restored = (Session.checkpoint_load(checkpoint, trusted=True) if route == "load"
                else Session().reattach(checkpoint, trusted=True))
    with pytest.raises(ValidationError, match="No fitted estimator|No graph plan"):
        getattr(restored, domain).evaluate(partition="test")
    history = copy.deepcopy(restored.history)
    preview = restored.audit.dry_run(operations)
    for step in preview.steps:
        assert not step.available, step
        assert step.status == "blocked" and step.blocked_reasons
    workflow = {step.operation: step for step in restored.audit.workflow()}
    roots = {node["operation"]: node for node in preview.prerequisite_graph.nodes}
    for operation in operations:
        assert workflow[operation].status.value == "blocked"
        assert restored.audit.explain(operation, moment="before").status.value == "blocked"
        assert not roots[operation]["available"]
    preview.show()
    text = capsys.readouterr().out
    assert all(f"[blocked] {op}" in text for op in operations)
    assert restored.history == history
    if domain == "classical":
        restored.classical.load_pipeline(saved, trusted=True)
        actual = np.asarray(restored.classical.predict(partition="test"))
    else:
        restored.graph.load_bundle(saved, trusted=True)
        actual = np.asarray(restored.graph.predict(partition="test").probabilities)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    getattr(restored, domain).evaluate(partition="test")
    assert all(step.available for step in restored.audit.dry_run(operations).steps)


@pytest.mark.parametrize("data_only", [False, True])
def test_reattach_discards_previous_model_and_domain_state_without_changing_session_identity(data_only, tmp_path):
    source = _session()
    checkpoint = source.checkpoint_save(tmp_path / "checkpoint")
    warm = _session(classification=True)
    warm.classical.fit(LogisticRegression(), task="classification")
    warm.graph.set_spec([(i, i+1) for i in range(47)], node_id_col="node")
    warm.graph.fit(method="classical", include_graph_metrics=False)
    warm.rag.ingest_corpus([{"doc_id": "old", "text": "Previous project private reference."}])
    warm.rag.embed_and_index(embedder="hashing", chunk_size=32, chunk_overlap=0)
    warm.ai.configure(provider="mock")
    facade = warm.classical
    identity = id(warm)
    assert warm.fit_result is not None and warm.graph_plan is not None and warm.rag.index_result is not None
    assert warm.rag.retrieve("private", k=1).hits[0].doc_id == "old"
    returned = warm.reattach(checkpoint, data_only=data_only, trusted=not data_only)
    assert returned is warm and id(warm) == identity
    assert warm.fit_result is None and warm.graph_plan is None and warm.graph_spec is None
    assert warm.rag.index_result is None
    assert not warm.ai.status()["enabled"]
    with pytest.raises(ValidationError, match="No RAG index"):
        warm.rag.retrieve("private", k=1)
    with pytest.raises(ValidationError, match="No fitted estimator"):
        facade.predict(partition="test")
    pd.testing.assert_frame_equal(warm.to_pandas(), source.to_pandas())


@pytest.mark.parametrize("stage", ["empty", "split", "fitted", "evaluated"])
def test_human_audit_summaries_match_real_history_counts_and_preserve_training_state(stage, capsys):
    session = Session() if stage == "empty" else _session()
    if stage in {"fitted", "evaluated"}:
        session.classical.fit(Ridge(alpha=1.3), task="regression")
    if stage == "evaluated":
        session.classical.evaluate(partition="test")
    before = joblib.hash((session.history, session.split_plan, session.fit_result))
    summary = session.audit.summarize()
    counts = Counter(row["operation_id"] for row in session.history)
    assert summary.n_operations == len(session.history)
    assert summary.operation_counts == dict(counts)
    assert summary.warning_count == sum(len(row["warnings"]) for row in session.history)
    assert summary.has_split == (stage != "empty")
    assert summary.has_fit == (stage in {"fitted", "evaluated"})
    assert summary.recent_operations == [row["operation_id"] for row in session.history[-8:]]
    summary.show()
    text = capsys.readouterr().out
    assert f"{len(session.history)} operation(s)" in text
    assert all(name in text for name in summary.recent_operations)
    for risk in summary.ranked_risks[:10]:
        assert risk.message in text
    preview = session.audit.dry_run(["fit", "evaluate", "not_a_real_operation"])
    preview.show()
    text = capsys.readouterr().out
    assert "[blocked] not_a_real_operation" in text
    assert preview.steps[-1].status == "unknown"
    assert "heuristic" in " ".join(preview.notes)
    json.dumps(summary.to_dict(), allow_nan=False)
    json.dumps(preview.to_dict(), allow_nan=False)
    assert joblib.hash((session.history, session.split_plan, session.fit_result)) == before


@pytest.mark.parametrize("engine", ["pandas", "polars", "duckdb"])
@pytest.mark.parametrize("data_only", [False, True])
def test_native_reattach_retains_checkpoint_rows_and_preprocessing_but_discards_old_outputs(engine, data_only, tmp_path):
    if engine != "pandas":
        pytest.importorskip(engine)
    source = _session().preprocess.scale(columns=["x", "z"], method="standard").with_engine(engine)
    checkpoint = source.checkpoint_save(tmp_path / "source")
    warm = _fitted_classical().with_engine(engine)
    warm.audit.dry_run("evaluate")
    warm.audit.summarize()
    old_connection = warm.dataset.native.connection if engine == "duckdb" else None
    old_dataset = warm.dataset
    identity = id(warm)
    warm.reattach(checkpoint, trusted=not data_only, data_only=data_only)
    assert id(warm) == identity and warm.dataset is not old_dataset
    pd.testing.assert_frame_equal(warm.to_pandas(), source.to_pandas())
    assert warm.fit_result is None and warm.last_cv is None
    assert warm.audit.last_dry_run is None and warm.audit.last_summary is None
    if data_only:
        assert warm.split_plan is None and warm.scale_plan is None
    else:
        assert warm.split_plan.train_indices == source.split_plan.train_indices
        assert joblib.hash(warm.scale_plan) == joblib.hash(source.scale_plan)
        assert warm.dataset.engine == source.dataset.engine
        assert warm.dataset.project(["x", "y"]).to_pandas().shape == (48, 2)
    if old_connection is not None:
        with pytest.raises(Exception, match="closed"):
            old_connection.execute("SELECT 1")
    warm.close_native()
    source.close_native()


@pytest.mark.parametrize("engine", ["pandas", "duckdb"])
def test_refused_untrusted_reattach_preserves_old_predictions_data_and_owned_connection(engine, tmp_path):
    if engine != "pandas":
        pytest.importorskip(engine)
    checkpoint = _session().preprocess.scale(columns=["x", "z"]).checkpoint_save(tmp_path / "untrusted")
    warm = _fitted_classical().with_engine(engine)
    before_frame = warm.to_pandas().copy(deep=True)
    connection = warm.dataset.native.connection if engine == "duckdb" else None
    predicted = np.asarray(warm.classical.predict(partition="test"))
    before = joblib.hash((warm.history, warm.fit_result, warm.split_plan))
    with pytest.raises(ValidationError, match="trusted=True"):
        warm.reattach(checkpoint)
    assert joblib.hash((warm.history, warm.fit_result, warm.split_plan)) == before
    pd.testing.assert_frame_equal(warm.to_pandas(), before_frame)
    np.testing.assert_allclose(warm.classical.predict(partition="test"), predicted, atol=0, rtol=0)
    if connection is not None:
        assert connection.execute("SELECT 17").fetchone()[0] == 17
    warm.close_native()


def test_real_cv_optout_risk_remains_ranked_visible_with_accurate_history_counts(capsys):
    session = _session().preprocess.scale(columns=["x", "z"])
    session.classical.cv_score(Ridge(alpha=1.3), task="regression", cv=3,
                               allow_session_global_preprocess=True)
    before = joblib.hash((session.history, session.last_cv, session.dataset.frame))
    summary = session.audit.summarize()
    assert summary.decision_origins == dict(Counter(row["decision_origin"] for row in session.history))
    assert summary.warning_count == sum(len(row["warnings"]) for row in session.history)
    assert any("preprocess" in str(item).lower() for item in session.last_cv.limitations)
    risk = next(r for r in summary.ranked_risks if r.source == "history.cv_preprocess_scope")
    assert risk.severity == "high" and "Session-global preprocess" in risk.message
    summary.show()
    assert risk.message in capsys.readouterr().out
    preview = session.audit.dry_run(["cv_score", "classical.fit"])
    preview.show()
    assert risk.message in capsys.readouterr().out
    assert joblib.hash((session.history, session.last_cv, session.dataset.frame)) == before
