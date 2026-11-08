"""Train-only blend designs retain class labels and replay numerical consumers."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.ensemble.blending import HoldoutBlendClassifier
from buildml.ensemble.checkpoint import load_ensemble_bundle
from buildml.ensemble.evaluate import build_ensemble_eval_report

VOCABULARIES = {
    "words": ["left", "right"],
    "multiclass": ["amber", "blue", "violet"],
    "numeric": [-7, 11, 35],
    "numeric_text": ["10", "40"],
    "nonfinite_text": ["inf", "nan"],
    "mixed_text": ["10", "right"],
    "mixed_nonfinite_text": ["1", "inf"],
}
FEATURES = ["x", "z", "w"]


def _frame(vocabulary):
    rng = np.random.default_rng(317)
    x = rng.normal(size=(96, 3))
    signal = x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2]
    cuts = [0.0] if len(vocabulary) == 2 else [-0.5, 0.5]
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["target"] = np.asarray(vocabulary)[np.digitize(signal, cuts)]
    frame["weight"] = 0.3 + rng.uniform(size=96)
    return frame


def _session(frame, backing, tmp_path):
    if backing in {"polars", "duckdb"}:
        pytest.importorskip(backing)
    if backing == "memory":
        source, engine, lazy = frame.copy(deep=True), "pandas", False
    else:
        source = tmp_path / (backing + ".parquet")
        frame.to_parquet(source, index=False)
        engine, lazy = ("pandas", False) if backing == "parquet" else (backing, True)
    session = Session.ingest(source, engine=engine, mode="lazy" if lazy else "memory").set_roles(
        {**{name: "feature" for name in FEATURES}, "target": "target", "weight": "weight"}
    )
    return session.inject_split(
        train_indices=list(range(60)),
        validation_indices=list(range(60, 78)),
        test_indices=list(range(78, 96)),
    )


def _linear_output(model, frame):
    scores = frame.to_numpy() @ model.coef_.T + model.intercept_
    if len(model.classes_) == 2:
        positive = 1 / (1 + np.exp(-scores[:, 0]))
        probabilities = np.column_stack([1 - positive, positive])
        labels = np.asarray(model.classes_)[(scores[:, 0] > 0).astype(int)]
    else:
        exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
        labels = np.asarray(model.classes_)[scores.argmax(axis=1)]
    return labels, probabilities


def _design(models, frame, mode, kind, passthrough):
    blocks = []
    for _, model in models:
        labels, probability = _linear_output(model, frame)
        if mode == "predict_proba":
            blocks.append(probability[:, 1:] if len(model.classes_) == 2 else probability)
        elif kind in {
            "words",
            "multiclass",
            "nonfinite_text",
            "mixed_text",
            "mixed_nonfinite_text",
        }:
            codes = {label: i for i, label in enumerate(model.classes_)}
            blocks.append(np.asarray([codes[label] for label in labels]).reshape(-1, 1))
        else:
            # Existing finite numeric and numeric-text class features keep their values.
            blocks.append(np.asarray(labels, dtype=float).reshape(-1, 1))
    if passthrough:
        blocks.append(frame.to_numpy())
    return np.hstack(blocks)


def _f1_weighted(y, prediction):
    total = 0.0
    for label in np.unique(y):
        actual, chosen = y == label, prediction == label
        tp = np.sum(actual & chosen)
        denominator = np.sum(actual) + np.sum(chosen)
        total += np.sum(actual) * (2 * tp / denominator if denominator else 0.0)
    return total / len(y)


@pytest.mark.parametrize("backing", ["memory", "parquet", "polars", "duckdb"])
@pytest.mark.parametrize("kind", list(VOCABULARIES))
@pytest.mark.parametrize("mode", ["predict", "predict_proba", "fallback"])
@pytest.mark.parametrize("passthrough,refit", [(False, False), (True, True)])
def test_class_labels_train_only_meta_design_and_saved_report(
    tmp_path, capsys, backing, kind, mode, passthrough, refit
):
    frame = _frame(VOCABULARIES[kind])
    if passthrough:
        frame.loc[60:, FEATURES] = frame.loc[60:, FEATURES] * 9 + 50
        frame.loc[60:, "target"] = np.roll(frame.loc[60:, "target"].to_numpy(), 5)
    original = frame.copy(deep=True)
    session = _session(frame, backing, tmp_path)
    prototypes = {
        "smooth": LogisticRegression(C=0.4, max_iter=1000, random_state=4),
        "sharp": (
            LinearSVC(C=0.3, dual=False, random_state=5)
            if mode == "fallback"
            else LogisticRegression(C=2.0, max_iter=1000, random_state=5)
        ),
    }
    resolved = "predict" if mode == "fallback" else mode
    requested = "predict_proba" if mode == "fallback" else mode
    xb, xm = train_test_split(
        np.arange(60), test_size=0.25, random_state=9, stratify=frame.loc[:59, "target"].to_numpy()
    )
    training_bases = [
        (
            name,
            clone(model).fit(
                frame.iloc[xb][FEATURES],
                frame.iloc[xb]["target"],
                sample_weight=frame.iloc[xb]["weight"],
            ),
        )
        for name, model in prototypes.items()
    ]
    meta_x = _design(training_bases, frame.iloc[xm][FEATURES], resolved, kind, passthrough)
    reference = LogisticRegression(max_iter=1000).fit(
        meta_x, frame.iloc[xm]["target"], sample_weight=frame.iloc[xm]["weight"]
    )
    session.ensemble.fit_blending(
        prototypes,
        blend_method=requested,
        holdout_fraction=0.25,
        random_state=9,
        passthrough=passthrough,
        refit_bases_on_full_train=refit,
        task="classification",
    )
    plan = session.ensemble_plan
    fitted = plan.estimator_
    assert plan.blend_method == resolved and plan.config["blend_method"] == requested
    assert fitted.blend_holdout_rows_ == 15 and fitted.blend_base_rows_ == 45
    np.testing.assert_allclose(fitted.final_estimator_.coef_, reference.coef_, atol=1e-8)
    np.testing.assert_allclose(fitted.final_estimator_.intercept_, reference.intercept_, atol=1e-8)
    for name, prototype in prototypes.items():
        rows = np.arange(60) if refit else xb
        expected = clone(prototype).fit(
            frame.iloc[rows][FEATURES],
            frame.iloc[rows]["target"],
            sample_weight=frame.iloc[rows]["weight"],
        )
        np.testing.assert_allclose(fitted.named_estimators_[name].coef_, expected.coef_)
        np.testing.assert_allclose(fitted.named_estimators_[name].intercept_, expected.intercept_)
        assert not hasattr(prototype, "coef_")
    weights = {name: model.coef_.copy() for name, model in fitted.estimators_}
    for partition in ["validation", "test"]:
        rows = frame.iloc[60:78] if partition == "validation" else frame.iloc[78:]
        design = _design(fitted.estimators_, rows[FEATURES], resolved, kind, passthrough)
        expected, probability = _linear_output(reference, pd.DataFrame(design))
        np.testing.assert_array_equal(session.predict(partition=partition), expected)
        np.testing.assert_allclose(fitted.predict_proba(rows[FEATURES]), probability, atol=1e-8)
        for position in range(3):
            np.testing.assert_allclose(
                fitted.predict_proba(rows.iloc[[position]][FEATURES]),
                probability[[position]],
                atol=1e-8,
            )
        report = build_ensemble_eval_report(
            session.dataset, session.split_plan, session.fit_result, plan, partition=partition
        )
        assert not report.warnings and report.diversity.n_bases == 2
        predictions = {}
        for contribution, (name, model) in zip(
            report.base_contributions, fitted.estimators_, strict=True
        ):
            prediction, _ = _linear_output(model, rows[FEATURES])
            predictions[name] = prediction
            actual = rows["target"].to_numpy()
            assert contribution.metrics["accuracy"] == pytest.approx(np.mean(actual == prediction))
            assert contribution.metrics["f1_weighted"] == pytest.approx(
                _f1_weighted(actual, prediction)
            )
            assert contribution.agree_with_ensemble == pytest.approx(
                np.mean(prediction == expected)
            )
        disagreement = float(np.mean(predictions["smooth"] != predictions["sharp"]))
        assert report.diversity.mean_pairwise_disagreement == pytest.approx(disagreement)
        report.show()
        assert f"partition={partition}" in capsys.readouterr().out
    bundle = session.ensemble.save_bundle(tmp_path / "blend")
    stored = {p.name: sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file()}
    loaded, fit = load_ensemble_bundle(bundle, trusted=True)
    assert fit.estimator is loaded.estimator_
    destination = _session(frame, backing, tmp_path)
    with pytest.raises(ValidationError, match="trust"):
        destination.ensemble.load_bundle(bundle)
    destination.ensemble.load_bundle(bundle, trusted=True)
    np.testing.assert_array_equal(
        destination.predict(partition="test"), session.predict(partition="test")
    )
    np.testing.assert_allclose(
        loaded.estimator_.predict_proba(frame.iloc[78:][FEATURES]),
        fitted.predict_proba(frame.iloc[78:][FEATURES]),
    )
    for name, model in fitted.estimators_:
        np.testing.assert_array_equal(model.coef_, weights[name])
    assert {
        p.name: sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.is_file()
    } == stored
    pd.testing.assert_frame_equal(session.to_pandas(), original, check_dtype=False)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("backing", ["memory", "parquet", "polars", "duckdb"])
@pytest.mark.parametrize("passthrough", [False, True])
def test_regression_prediction_design_remains_numeric(tmp_path, capsys, backing, passthrough):
    frame = _frame([0, 1])
    frame["target"] = frame["x"] * 2 - frame["z"] * 0.6 + np.sin(frame["w"])
    session = _session(frame, backing, tmp_path)
    session.ensemble.fit_blending(
        {"a": Ridge(alpha=0.3), "b": Ridge(alpha=2.0)},
        passthrough=passthrough,
        random_state=9,
        task="regression",
    )
    model = session.ensemble_plan.estimator_
    rows = frame.iloc[78:]
    columns = [
        rows[FEATURES].to_numpy() @ base.coef_ + base.intercept_ for _, base in model.estimators_
    ]
    design = np.column_stack(columns)
    if passthrough:
        design = np.column_stack([design, rows[FEATURES]])
    expected = design @ model.final_estimator_.coef_ + model.final_estimator_.intercept_
    np.testing.assert_allclose(session.predict(partition="test"), expected)
    report = build_ensemble_eval_report(
        session.dataset,
        session.split_plan,
        session.fit_result,
        session.ensemble_plan,
        partition="test",
    )
    for value, (_, base) in zip(report.base_contributions, model.estimators_, strict=True):
        prediction = rows[FEATURES].to_numpy() @ base.coef_ + base.intercept_
        assert value.metrics["mae"] == pytest.approx(np.mean(np.abs(prediction - rows["target"])))
    report.show()
    assert "regression" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_unfitted_blend_still_refuses_predictions(method):
    with pytest.raises(NotFittedError):
        getattr(HoldoutBlendClassifier(), method)(np.ones((3, 2)))


@pytest.mark.parametrize("kind", ["words", "numeric"])
def test_fitted_margin_meta_model_discloses_missing_probabilities(kind):
    frame = _frame(VOCABULARIES[kind])
    model = HoldoutBlendClassifier(
        estimators=[("a", LogisticRegression(max_iter=1000)), ("b", LogisticRegression(C=2))],
        final_estimator=LinearSVC(dual=False),
        random_state=9,
    ).fit(frame[FEATURES], frame["target"])
    before = deepcopy(model.final_estimator_.coef_)
    with pytest.raises(AttributeError, match="predict_proba"):
        model.predict_proba(frame.iloc[:3][FEATURES])
    np.testing.assert_array_equal(model.final_estimator_.coef_, before)
    assert len(model.predict(frame.iloc[:3][FEATURES])) == 3
