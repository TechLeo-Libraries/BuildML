"""Multi-task bundles must reject inconsistent fitted contracts before attachment."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.multitask.checkpoint import load_multitask_bundle, save_multitask_bundle


@pytest.fixture(scope="module", params=["regression", "strings", "integers", "booleans"])
def consumer(request):
    rng = np.random.default_rng(436)
    x = rng.normal(size=(80, 2))
    y = np.column_stack([1.8 * x[:, 0] - 0.2 * x[:, 1], -0.4 * x[:, 0] + 1.2 * x[:, 1]])
    if request.param != "regression":
        vocabulary = {"strings": ["001", "010"], "integers": [2, 10], "booleans": [False, True]}[
            request.param
        ]
        y = np.asarray(vocabulary)[(y > 0).astype(int)]
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "a": y[:, 0], "b": y[:, 1]})
    s = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "a": "target", "b": "target"})
        .split(test_size=0.2, random_state=4)
    )
    fit = s.multitask.fit(
        backend="sklearn",
        method="multi_output",
        task="regression" if request.param == "regression" else "classification",
        base_estimator="ridge" if request.param == "regression" else "logistic_regression",
        prefer_reduce_components=False,
    )
    return s, fit


def malformed(plan, kind):
    p = copy.deepcopy(plan)
    if kind == "duplicate_features":
        p.columns = ("x", "x")
    elif kind == "duplicate_targets":
        p.target_columns = ("a", "a")
    elif kind == "missing_head":
        p.target_columns = ("a",)
    elif kind == "missing_model":
        p.estimator_ = None
    elif kind == "unfitted_model":
        p.estimator_ = MultiOutputRegressor(Ridge())
    elif kind == "feature_width":
        p.columns = ("x", "z", "extra")
    elif kind == "zero_rows":
        p.n_train_rows = 0
    elif kind == "invalid_task":
        p.task = "invalid"
    return p


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_features",
        "duplicate_targets",
        "missing_head",
        "missing_model",
        "unfitted_model",
        "feature_width",
        "zero_rows",
        "invalid_task",
    ],
)
@pytest.mark.parametrize("boundary", ["save", "load"])
def test_malformed_model_contract_is_refused_before_files_or_session_change(
    consumer, kind, boundary, tmp_path
):
    s, fit = consumer
    original = joblib.hash(s.multitask_plan)
    path = tmp_path / "bundle"
    bad = malformed(s.multitask_plan, kind)
    if boundary == "save":
        with pytest.raises(ValidationError):
            save_multitask_bundle(path, bad)
        assert not path.exists()
    else:
        save_multitask_bundle(path, s.multitask_plan, fit_result=fit)
        receiver = Session.ingest(s.to_pandas()).set_roles(s.dataset.roles)
        receiver.multitask.load_bundle(path, trusted=True)
        received = joblib.hash(receiver.multitask_plan)
        joblib.dump({"plan": bad}, path / "multitask_plan.joblib")
        with pytest.raises(ValidationError):
            receiver.multitask.load_bundle(path, trusted=True)
        assert joblib.hash(receiver.multitask_plan) == received
    assert joblib.hash(s.multitask_plan) == original


@pytest.mark.parametrize(
    "field", ["columns", "target_columns", "n_train_rows", "n_tasks", "task", "method", "backend"]
)
def test_saved_summary_disagreement_cannot_relabel_the_loaded_model(consumer, field, tmp_path):
    s, fit = consumer
    path = save_multitask_bundle(tmp_path / "bundle", s.multitask_plan, fit_result=fit)
    receiver = Session.ingest(s.to_pandas()).set_roles(s.dataset.roles)
    receiver.multitask.load_bundle(path, trusted=True)
    p = path / "meta.json"
    meta = json.loads(p.read_text())
    if field in {"columns", "target_columns"}:
        meta["plan"][field] = list(reversed(meta["plan"][field]))
    else:
        meta["plan"][field] = 999 if field in {"n_train_rows", "n_tasks"} else "inconsistent"
    p.write_text(json.dumps(meta))
    before = joblib.hash(receiver.multitask_plan)
    with pytest.raises(ValidationError):
        receiver.multitask.load_bundle(path, trusted=True)
    assert joblib.hash(receiver.multitask_plan) == before


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("summaries", [False, True])
def test_valid_current_and_legacy_payloads_preserve_numerical_predictions(
    consumer, direct, summaries, tmp_path
):
    s, fit = consumer
    plan = s.multitask_plan
    path = save_multitask_bundle(tmp_path / "bundle", plan, fit_result=fit if summaries else None)
    if direct:
        joblib.dump(plan, path / "multitask_plan.joblib")
    if not summaries:
        p = path / "meta.json"
        meta = json.loads(p.read_text())
        meta.pop("plan", None)
        meta.pop("fit", None)
        p.write_text(json.dumps(meta))
    restored = load_multitask_bundle(path, trusted=True)
    x = s.partition("test")[["x", "z"]].to_numpy()
    expected = []
    for model in plan.estimator_.estimators_:
        values = x @ model.coef_.T + model.intercept_
        expected.append(
            values.ravel() if plan.task == "regression" else (values.ravel() > 0).astype(int)
        )
    np.testing.assert_allclose(
        restored.estimator_.predict(x), np.column_stack(expected), atol=1e-12
    )


@pytest.mark.parametrize("consumer", ["strings", "integers", "booleans"], indirect=True)
@pytest.mark.parametrize("kind", ["missing_encoder", "wrong_classes"])
def test_classifier_encoder_identity_is_checked_before_save(consumer, kind, tmp_path):
    s, _ = consumer
    plan = copy.deepcopy(s.multitask_plan)
    if kind == "missing_encoder":
        plan.label_encoders_.pop("a")
    else:
        plan.classes_per_task_["a"] = tuple(reversed(plan.classes_per_task_["a"]))
    path = tmp_path / "invalid"
    with pytest.raises(ValidationError):
        save_multitask_bundle(path, plan)
    assert not path.exists()


def test_real_mixed_torch_bundle_replays_independent_numpy_heads(tmp_path):
    torch = pytest.importorskip("torch")
    x = np.linspace(-2, 2, 48)
    frame = pd.DataFrame(
        {"x": x, "z": np.cos(x), "label": np.where(x > 0, "yes", "no"), "amount": x * 2 + 0.3}
    )
    s = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "label": "target", "amount": "target"})
        .split(test_size=0.2, random_state=3)
    )
    fit = s.multitask.fit(
        backend="torch",
        method="shared_trunk_multihead",
        task="auto",
        epochs=2,
        batch_size=12,
        prefer_reduce_components=False,
        device="cpu",
    )
    plan = s.multitask_plan
    module = plan.estimator_.module_
    matrix = s.partition("test")[["x", "z"]].to_numpy(dtype=np.float32)
    shared = matrix
    for layer in module.trunk:
        if isinstance(layer, torch.nn.Linear):
            shared = shared @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
        else:
            shared = np.maximum(shared, 0)
    expected = []
    for target, head in zip(plan.target_columns, module.heads, strict=True):
        raw = shared @ head.weight.detach().numpy().T + head.bias.detach().numpy()
        expected.append(
            raw.argmax(axis=1) if plan.task_kinds_[target] == "classification" else raw.ravel()
        )
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    path = save_multitask_bundle(tmp_path / "torch-bundle", plan, fit_result=fit)
    restored = load_multitask_bundle(path, trusted=True)
    np.testing.assert_allclose(
        restored.estimator_.predict(matrix), np.column_stack(expected), atol=1e-6
    )
    for key, value in before.items():
        torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)

    assert restored.estimator_.device == "cpu"
    assert next(restored.estimator_.module_.parameters()).device.type == "cpu"
    receiver = Session.ingest(frame).set_roles(s.dataset.roles)
    receiver.multitask.load_bundle(path, trusted=True)
    kept = receiver.multitask_plan
    broken = copy.deepcopy(plan)
    # A damaged persisted tensor shape must fail strict reconstruction before
    # the receiver can replace its currently usable learner.
    head = broken.estimator_.module_.heads[0]
    broken.estimator_.module_.heads[0] = torch.nn.Linear(head.in_features, head.out_features + 1)
    joblib.dump({"plan": broken}, path / "multitask_plan.joblib")
    with pytest.raises(ValidationError, match="weights"):
        receiver.multitask.load_bundle(path, trusted=True)
    assert receiver.multitask_plan is kept
