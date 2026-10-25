"""Semi-supervised bundle contracts preserve class identity and missingness counts."""

import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.semi_supervised import LabelPropagation

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.semisupervised.checkpoint import load_semisupervised_bundle, save_semisupervised_bundle
from buildml.semisupervised.predict import predict_semisupervised


@pytest.fixture(
    scope="module",
    params=[
        (method, labels)
        for method in ("label_propagation", "self_training")
        for labels in (("001", "010"), (2, 10), (False, True))
    ],
)
def fitted(request):
    method, labels = request.param
    rng = np.random.default_rng(619)
    x = np.r_[rng.normal(-1.5, 0.4, 36), rng.normal(1.5, 0.4, 36)]
    y = pd.Series([labels[0]] * 36 + [labels[1]] * 36, dtype=object)
    y.iloc[::5] = None
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=72), "y": y})
    s = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "y": "target"})
        .split(test_size=0.2, random_state=3)
    )
    fit = s.semisupervised.fit(method=method, kernel="rbf", prefer_reduce_components=False)
    return s, fit


def corrupt(plan, kind):
    p = copy.deepcopy(plan)
    if kind == "duplicate_features":
        p.columns = ("x", "x")
    elif kind == "missing_target":
        p.target_column = ""
    elif kind == "missing_model":
        p.estimator_ = None
    elif kind == "unfitted_model":
        p.estimator_ = LabelPropagation()
    elif kind == "feature_width":
        p.columns = ("x", "z", "extra")
    elif kind == "zero_rows":
        p.n_train_rows = 0
    elif kind == "negative_labels":
        p.n_labeled_train = -1
    elif kind == "inconsistent_counts":
        p.n_unlabeled_train += 1
    elif kind == "missing_encoder":
        p.label_encoder_ = None
    elif kind == "reversed_classes":
        p.classes_ = tuple(reversed(p.classes_))
    elif kind == "invalid_model_codes":
        p.estimator_.classes_ = np.array([0, 3])
    elif kind == "one_class":
        p.classes_ = p.classes_[:1]
    return p


def receiver(s, path):
    other = Session.ingest(s.to_pandas()).set_roles(s.dataset.roles)
    other.semisupervised.load_bundle(path, trusted=True)
    return other


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_features",
        "missing_target",
        "missing_model",
        "unfitted_model",
        "feature_width",
        "zero_rows",
        "negative_labels",
        "inconsistent_counts",
        "missing_encoder",
        "reversed_classes",
        "invalid_model_codes",
        "one_class",
    ],
)
@pytest.mark.parametrize("boundary", ["save", "load"])
def test_bad_contract_is_refused_before_files_or_receiver_change(fitted, kind, boundary, tmp_path):
    s, fit = fitted
    before = joblib.hash(s.semisupervised_plan)
    path = tmp_path / "bundle"
    bad = corrupt(s.semisupervised_plan, kind)
    if boundary == "save":
        with pytest.raises(ValidationError):
            save_semisupervised_bundle(path, bad)
        assert not path.exists()
    else:
        save_semisupervised_bundle(path, s.semisupervised_plan, fit_result=fit)
        other = receiver(s, path)
        retained = other.semisupervised_plan
        joblib.dump({"plan": bad}, path / "semisupervised_plan.joblib")
        with pytest.raises(ValidationError):
            other.semisupervised.load_bundle(path, trusted=True)
        assert other.semisupervised_plan is retained
    assert joblib.hash(s.semisupervised_plan) == before


@pytest.mark.parametrize(
    "field",
    [
        "columns",
        "target_column",
        "method",
        "backend",
        "modality",
        "n_train_rows",
        "n_labeled_train",
        "n_unlabeled_train",
        "classes",
    ],
)
def test_persisted_summary_disagreement_is_refused_atomically(fitted, field, tmp_path):
    s, fit = fitted
    path = save_semisupervised_bundle(tmp_path / "bundle", s.semisupervised_plan, fit_result=fit)
    other = receiver(s, path)
    retained = other.semisupervised_plan
    p = path / "meta.json"
    meta = json.loads(p.read_text())
    value = meta["plan"][field]
    meta["plan"][field] = (
        list(reversed(value))
        if isinstance(value, list)
        else (999 if isinstance(value, int) else "inconsistent")
    )
    p.write_text(json.dumps(meta))
    with pytest.raises(ValidationError):
        other.semisupervised.load_bundle(path, trusted=True)
    assert other.semisupervised_plan is retained


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("summaries", [False, True])
def test_valid_legacy_and_current_reload_matches_independent_probability_oracle(
    fitted, direct, summaries, tmp_path
):
    s, fit = fitted
    plan = s.semisupervised_plan
    path = save_semisupervised_bundle(
        tmp_path / "bundle", plan, fit_result=fit if summaries else None
    )
    if direct:
        joblib.dump(plan, path / "semisupervised_plan.joblib")
    if not summaries:
        p = path / "meta.json"
        meta = json.loads(p.read_text())
        meta.pop("plan", None)
        meta.pop("fit", None)
        p.write_text(json.dumps(meta))
    restored = load_semisupervised_bundle(path, trusted=True)
    x = s.partition("test")[["x", "z"]].to_numpy()
    model = plan.estimator_
    if plan.method == "label_propagation":
        kernels = np.exp(-model.gamma * np.sum((x[:, None, :] - model.X_[None, :, :]) ** 2, axis=2))
        probabilities = kernels @ model.label_distributions_
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        codes = probabilities.argmax(axis=1)
    else:
        base = getattr(model, "estimator_", getattr(model, "base_estimator_", None))
        logits = (x @ base.coef_.T + base.intercept_).ravel()
        codes = (logits > 0).astype(int)
    expected = np.asarray(plan.classes_)[codes]
    _, result = predict_semisupervised(s.dataset, restored, s.split_plan, partition="test")
    np.testing.assert_array_equal(result.predictions, expected)
    assert restored.n_labeled_train + restored.n_unlabeled_train == restored.n_train_rows


@pytest.mark.parametrize(
    "fitted",
    [("label_propagation", (False, True)), ("self_training", (False, True))],
    indirect=True,
)
@pytest.mark.parametrize("boundary", ["plan", "summary"])
def test_boolean_classes_cannot_silently_become_integer_classes(fitted, boundary, tmp_path):
    s, fit = fitted
    path = tmp_path / "bundle"
    if boundary == "plan":
        plan = copy.deepcopy(s.semisupervised_plan)
        plan.classes_ = (0, 1)
        with pytest.raises(ValidationError, match="vocabulary"):
            save_semisupervised_bundle(path, plan)
        assert not path.exists()
    else:
        save_semisupervised_bundle(path, s.semisupervised_plan, fit_result=fit)
        other = receiver(s, path)
        retained = other.semisupervised_plan
        p = path / "meta.json"
        meta = json.loads(p.read_text())
        meta["plan"]["classes"] = [0, 1]
        p.write_text(json.dumps(meta))
        with pytest.raises(ValidationError, match="classes"):
            other.semisupervised.load_bundle(path, trusted=True)
        assert other.semisupervised_plan is retained
