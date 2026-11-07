"""Frozen SSL scoring and head fitting consume only their labeled population."""

from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.selfsupervised.checkpoint import load_ssl_bundle, save_ssl_bundle
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.transform import transform_ssl

METHODS = ["masked_tabular", "simclr_tabular", "byol_tabular", "vicreg_tabular",
           "mae_tabular", "vae_tabular"]


@pytest.fixture(scope="module", params=METHODS)
def pretext(request):
    if request.param != "masked_tabular":
        pytest.importorskip("torch")
    rng = np.random.default_rng(287)
    x = rng.normal(size=(72, 3))
    frame = pd.DataFrame(x, columns=["a", "b", "c"], index=np.arange(72) % 7)
    frame["y"] = np.where(x[:, 0] + .3*x[:, 1] > 0, "approved", "declined")
    session = (Session.ingest(frame).set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
               .split(test_size=.25, validation_size=.25, random_state=7))
    options = dict(method=request.param, latent_dim=3, hidden=(8,), random_state=5,
                   epochs=1, batch_size=16, max_iter=2000)
    if request.param == "masked_tabular":
        with pytest.warns(DeprecationWarning, match="legacy sklearn MLP"):
            session.ssl.fit_pretext(**options)
    else:
        session.ssl.fit_pretext(**options)
    return session, frame


@pytest.fixture(scope="module", params=["logistic_regression", "hist_gradient_boosting"])
def trained(pretext, request, tmp_path_factory):
    session, frame = pretext
    head, _ = finetune_ssl_head(session.dataset, session.split_plan, session._ssl_plan,
                               estimator=request.param, random_state=5)
    root = tmp_path_factory.mktemp("ssl-population")
    save_ssl_bundle(root, session._ssl_plan, head_plan=head)
    return session, frame, session._ssl_plan, head, root


@pytest.fixture(scope="module", params=[False, True])
def fitted(trained, request):
    session, frame, plan, head, root = trained
    if request.param:
        plan, head = load_ssl_bundle(root, trusted=True)
    return session, frame, plan, head


def _dataset(session, frame):
    return Dataset.from_pandas(frame, roles=dict(session.dataset.roles))


def _fingerprint(plan, head=None):
    def arrays(value):
        if isinstance(value, dict):
            return {key: arrays(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [arrays(item) for item in value]
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().copy()
        return value
    if hasattr(plan.encoder_, "state_dict"):
        return joblib.hash((plan.to_dict(), arrays(plan.encoder_.state_dict()), head))
    return joblib.hash((plan, head))


def _metrics(actual, predicted):
    correct = sum(a == b for a, b in zip(actual, predicted, strict=True))
    scores, supports = [], []
    for label in sorted(set(actual) | set(predicted)):
        tp = sum(a == label and b == label for a, b in zip(actual, predicted, strict=True))
        fp = sum(a != label and b == label for a, b in zip(actual, predicted, strict=True))
        fn = sum(a == label and b != label for a, b in zip(actual, predicted, strict=True))
        scores.append(0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
        supports.append(sum(a == label for a in actual))
    return dict(accuracy=correct/len(actual), f1_macro=np.mean(scores),
                f1_weighted=np.average(scores, weights=supports), support_labeled=float(len(actual)))


@pytest.mark.parametrize("partition", ["validation", "test", "all"])
@pytest.mark.parametrize("population", ["valid", "partial_clean", "partial_corrupt",
                                       "unlabeled_corrupt", "marker_corrupt", "nullable_marker"])
def test_frozen_scores_count_only_labeled_rows_and_match_scalar_metrics(fitted, partition, population):
    session, original, plan, head = fitted
    frame = original.copy()
    positions = np.arange(len(frame)) if partition == "all" else np.array(session.split_plan.indices_for(partition))
    marker = "unobserved" if population in {"marker_corrupt", "nullable_marker"} else None
    if population == "nullable_marker":
        frame["y"] = frame.y.astype("string")
    if population != "valid":
        blank = positions if population == "unlabeled_corrupt" else positions[::3]
        frame.iloc[blank, frame.columns.get_loc("y")] = marker
        if population == "nullable_marker":
            frame.iloc[blank[::2], frame.columns.get_loc("y")] = pd.NA
        if population != "partial_clean":
            frame.iloc[blank, frame.columns.get_indexer(["a", "b", "c"])] = np.nan
    selected = frame.iloc[positions]
    labeled = selected.y.notna() & (selected.y != marker)
    observed = selected.loc[labeled]
    expected = {}
    if len(observed):
        embeddings = plan.encoder_.transform(observed[list(plan.columns)].to_numpy())
        if head.estimator_name == "logistic_regression":
            decision = embeddings @ head.estimator_.coef_.T + head.estimator_.intercept_
            codes = (decision[:, 0] > 0).astype(int)
        else:
            codes = head.estimator_.predict(embeddings)
        labels = head.estimator_._buildml_label_encoder_.inverse_transform(codes)
        expected = _metrics(observed.y.tolist(), labels.tolist())
    fingerprint = _fingerprint(plan, head)
    result = evaluate_ssl(_dataset(session, frame), plan, head, session.split_plan,
                          partition=partition, unlabeled_marker=marker)
    assert result.n_rows == len(selected)
    assert result.n_labeled_eval == len(observed)
    assert result.n_unlabeled_eval == len(selected)-len(observed)
    assert result.metrics == pytest.approx(expected)
    assert any("independence" in note for note in result.disclosures)
    if partition == "all":
        assert any("diagnostic" in warning for warning in result.warnings)
    if not len(observed):
        assert any("No labeled" in warning for warning in result.warnings)
    assert _fingerprint(plan, head) == fingerprint


def test_empty_evaluation_returns_counts_without_synthetic_metrics(fitted):
    session, frame, plan, head = fitted
    fingerprint = _fingerprint(plan, head)
    result = evaluate_ssl(_dataset(session, frame.iloc[:0].copy()), plan, head, None, partition="all")
    assert result.n_rows == result.n_labeled_eval == result.n_unlabeled_eval == 0
    assert result.metrics == {}
    assert any("No labeled" in warning for warning in result.warnings)
    assert any("diagnostic" in warning for warning in result.warnings)
    assert _fingerprint(plan, head) == fingerprint


@pytest.mark.parametrize("population", ["partial", "marker", "nullable_marker", "too_few", "single_class"])
def test_head_fit_skips_unobserved_features_and_preserves_frozen_pretext(fitted, population):
    session, original, plan, previous = fitted
    frame = original.copy()
    train = np.array(session.split_plan.train_indices)
    marker = "unobserved" if population in {"marker", "nullable_marker"} else None
    if population == "nullable_marker":
        frame["y"] = frame.y.astype("string")
    blank = train[::3] if population in {"partial", "marker", "nullable_marker"} else train[1:]
    frame.iloc[blank, frame.columns.get_loc("y")] = marker
    if population == "nullable_marker":
        frame.iloc[blank[::2], frame.columns.get_loc("y")] = pd.NA
    frame.iloc[blank, frame.columns.get_indexer(["a", "b", "c"])] = np.nan
    if population == "single_class":
        frame.iloc[train, frame.columns.get_loc("y")] = "approved"
    data = _dataset(session, frame)
    fingerprint = _fingerprint(plan, previous)
    if population in {"too_few", "single_class"}:
        message = "labeled train rows" if population == "too_few" else "at least 2 classes"
        with pytest.raises(ValidationError, match=message):
            finetune_ssl_head(data, session.split_plan, plan, estimator=previous.estimator_name)
    else:
        selected = frame.iloc[train]
        observed = selected.loc[selected.y.notna() & (selected.y != marker)]
        embeddings = np.asarray(plan.encoder_.transform(observed[list(plan.columns)].to_numpy()), dtype=float)
        codes = previous.estimator_._buildml_label_encoder_.transform(observed.y.to_numpy())
        reference = clone(previous.estimator_).fit(embeddings, codes)
        head, report = finetune_ssl_head(data, session.split_plan, plan,
                                        estimator=previous.estimator_name, random_state=5,
                                        unlabeled_marker=marker)
        assert report.n_labeled_train == len(observed)
        assert report.n_unlabeled_skipped == len(selected)-len(observed)
        holdout = original.iloc[list(session.split_plan.test_indices)]
        query = plan.encoder_.transform(holdout[list(plan.columns)].to_numpy())
        np.testing.assert_array_equal(head.estimator_.predict(query), reference.predict(query))
        if head.estimator_name == "logistic_regression":
            np.testing.assert_allclose(head.estimator_.coef_, reference.coef_, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(head.estimator_.intercept_, reference.intercept_, rtol=1e-12, atol=1e-12)
    assert _fingerprint(plan, previous) == fingerprint


@pytest.mark.parametrize("fault", ["head_columns", "unknown_class", "nonfinite"])
def test_invalid_population_refuses_without_changing_frozen_state(fitted, fault):
    session, original, plan, head = fitted
    frame = original.copy()
    indices = np.array(session.split_plan.test_indices)
    if fault == "head_columns":
        head = replace(head, representation_columns=("wrong",)*len(head.representation_columns))
        message = "representation"
    elif fault == "unknown_class":
        frame.iloc[indices, frame.columns.get_loc("y")] = "new-class"
        frame.iloc[indices, frame.columns.get_loc("a")] = np.nan
        message = "class labels"
    else:
        frame.iloc[indices, frame.columns.get_loc("a")] = np.inf
        message = "finite"
    fingerprint = _fingerprint(plan, head)
    with pytest.raises(ValidationError, match=message):
        evaluate_ssl(_dataset(session, frame), plan, head, session.split_plan, partition="test")
    assert _fingerprint(plan, head) == fingerprint


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("attach", [False, True])
def test_empty_representation_export_has_the_saved_width(pretext, tmp_path, replay, attach):
    session, frame = pretext
    plan = session._ssl_plan
    if replay:
        save_ssl_bundle(tmp_path/"empty", plan)
        plan, _ = load_ssl_bundle(tmp_path/"empty", trusted=True)
    data = _dataset(session, frame.iloc[:0].copy())
    fingerprint = _fingerprint(plan)
    attached, report, embeddings = transform_ssl(data, plan, None, partition="all", attach=attach)
    assert embeddings.shape == (0, len(plan.representation_columns))
    assert report.n_rows == 0 and report.attached is attach
    if attach:
        assert attached.frame.empty
        assert set(plan.representation_columns).issubset(attached.frame.columns)
    else:
        assert attached is None
    assert _fingerprint(plan) == fingerprint


@pytest.mark.parametrize("population", ["partial_unlabeled", "unknown_class", "nonfinite"])
def test_session_reports_or_refuses_atomically_after_real_bundle_load(trained, population):
    _, original, _, _, root = trained
    frame = original.copy()
    consumer = (Session.ingest(frame).set_roles({"a": "feature", "b": "feature", "c": "feature", "y": "target"})
                .split(test_size=.25, validation_size=.25, random_state=7))
    consumer.ssl.load_bundle(root, trusted=True)
    positions = np.array(consumer.split_plan.test_indices)
    if population == "partial_unlabeled":
        blank = positions[::3]
        consumer.dataset.frame.iloc[blank, consumer.dataset.frame.columns.get_loc("y")] = None
        consumer.dataset.frame.iloc[blank, consumer.dataset.frame.columns.get_loc("a")] = np.nan
    elif population == "unknown_class":
        consumer.dataset.frame.iloc[positions, consumer.dataset.frame.columns.get_loc("y")] = "novel"
        consumer.dataset.frame.iloc[positions, consumer.dataset.frame.columns.get_loc("a")] = np.nan
    else:
        consumer.dataset.frame.iloc[positions, consumer.dataset.frame.columns.get_loc("a")] = np.inf
    before = list(consumer._history)
    fingerprint = _fingerprint(consumer._ssl_plan, consumer._ssl_head_plan)
    if population == "partial_unlabeled":
        result = consumer.ssl.evaluate(partition="test")
        assert result.n_labeled_eval == len(positions)-len(positions[::3])
        assert consumer._ssl_eval_result is result
        assert len(consumer._history) == len(before)+1
    else:
        message = "class labels" if population == "unknown_class" else "finite"
        with pytest.raises(ValidationError, match=message):
            consumer.ssl.evaluate(partition="test")
        assert consumer._history == before and consumer._ssl_eval_result is None
    assert _fingerprint(consumer._ssl_plan, consumer._ssl_head_plan) == fingerprint
