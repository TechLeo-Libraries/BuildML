"""Real local sentence encoders preserve SSL labeled-population contracts."""

import numpy as np
import pandas as pd
import pytest
import test_offline_local_encoder_contracts as offline
from sklearn.base import clone
from test_ssl_labeled_population_consumers import _dataset, _fingerprint, _metrics

from buildml import Session
from buildml.selfsupervised.checkpoint import load_ssl_bundle, save_ssl_bundle
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.transform import transform_ssl

tiny_bert = offline.tiny_bert
local_sentence_encoder = offline.local_sentence_encoder


@pytest.fixture(scope="module")
def text_pretext(local_sentence_encoder):
    frame = pd.DataFrame({"text": ["red apple", "engine valve", "blue berry fruit", "engine valve"]*12,
                          "y": ["fruit", "machine", "fruit", "machine"]*12},
                         index=np.arange(48) % 5)
    session = (Session.ingest(frame).set_roles({"text": "feature", "y": "target"})
               .split(test_size=.25, validation_size=.25, random_state=7))
    session.ssl.fit_pretext(method="hf_text_ssl", text_column="text", latent_dim=5,
                            epochs=0, hf_model_name=str(local_sentence_encoder),
                            weight_mode="pretrained", batch_size=4)
    return session, frame


@pytest.fixture(scope="module", params=["logistic_regression", "hist_gradient_boosting"])
def text_trained(text_pretext, request, tmp_path_factory):
    session, frame = text_pretext
    head, _ = finetune_ssl_head(session.dataset, session.split_plan, session._ssl_plan,
                               estimator=request.param, random_state=5)
    root = tmp_path_factory.mktemp("ssl-text-population")
    save_ssl_bundle(root, session._ssl_plan, head_plan=head)
    return session, frame, session._ssl_plan, head, root


@pytest.fixture(scope="module", params=[False, True])
def text_fitted(text_trained, request):
    session, frame, plan, head, root = text_trained
    if request.param:
        plan, head = load_ssl_bundle(root, trusted=True)
    return session, frame, plan, head


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("population", ["nullable_marker", "unlabeled"])
def test_local_text_scores_and_counts_match_frozen_consumer(text_fitted, partition, population):
    session, original, plan, head = text_fitted
    frame = original.copy()
    frame["y"] = frame.y.astype("string")
    positions = np.array(session.split_plan.indices_for(partition))
    blank = positions if population == "unlabeled" else positions[::2]
    frame.iloc[blank, frame.columns.get_loc("y")] = "unobserved"
    frame.iloc[blank[::2], frame.columns.get_loc("y")] = pd.NA
    frame.iloc[blank, frame.columns.get_loc("text")] = None
    observed = frame.iloc[positions].loc[lambda data: data.y.notna() & (data.y != "unobserved")]
    expected = {}
    if len(observed):
        embeddings = plan.encoder_.transform(observed.text.tolist())
        predictions = head.estimator_._buildml_label_encoder_.inverse_transform(head.estimator_.predict(embeddings))
        expected = _metrics(observed.y.tolist(), predictions.tolist())
    fingerprint = _fingerprint(plan, head)
    result = evaluate_ssl(_dataset(session, frame), plan, head, session.split_plan,
                          partition=partition, unlabeled_marker="unobserved")
    assert result.metrics == pytest.approx(expected)
    assert result.n_labeled_eval == len(observed)
    assert result.n_unlabeled_eval == len(positions)-len(observed)
    assert any("independence" in note for note in result.disclosures)
    assert _fingerprint(plan, head) == fingerprint


def test_text_head_fit_matches_real_labeled_only_sklearn_fit(text_fitted):
    session, original, plan, previous = text_fitted
    frame = original.copy()
    frame["y"] = frame.y.astype("string")
    train = np.array(session.split_plan.train_indices)
    blank = train[::3]
    frame.iloc[blank, frame.columns.get_loc("y")] = "unobserved"
    frame.iloc[blank[::2], frame.columns.get_loc("y")] = pd.NA
    frame.iloc[blank, frame.columns.get_loc("text")] = None
    observed = frame.iloc[train].loc[lambda data: data.y.notna() & (data.y != "unobserved")]
    embeddings = plan.encoder_.transform(observed.text.tolist())
    codes = previous.estimator_._buildml_label_encoder_.transform(observed.y.to_numpy())
    reference = clone(previous.estimator_).fit(embeddings, codes)
    fingerprint = _fingerprint(plan, previous)
    head, report = finetune_ssl_head(_dataset(session, frame), session.split_plan, plan,
                                    estimator=previous.estimator_name, random_state=5,
                                    unlabeled_marker="unobserved")
    assert report.n_labeled_train == len(observed)
    assert report.n_unlabeled_skipped == len(train)-len(observed)
    query = plan.encoder_.transform(original.text.tolist())
    np.testing.assert_array_equal(head.estimator_.predict(query), reference.predict(query))
    assert _fingerprint(plan, previous) == fingerprint


def test_empty_text_evaluation_keeps_counts_without_encoder_failure(text_fitted):
    session, original, plan, head = text_fitted
    result = evaluate_ssl(_dataset(session, original.iloc[:0].copy()), plan, head, None, partition="all")
    assert result.n_rows == result.n_labeled_eval == result.n_unlabeled_eval == 0
    assert result.metrics == {}


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("attach", [False, True])
def test_empty_text_export_retains_fitted_width(text_pretext, tmp_path, replay, attach):
    session, frame = text_pretext
    plan = session._ssl_plan
    if replay:
        save_ssl_bundle(tmp_path/"empty-text", plan)
        plan, _ = load_ssl_bundle(tmp_path/"empty-text", trusted=True)
    fingerprint = _fingerprint(plan)
    data, report, embeddings = transform_ssl(_dataset(session, frame.iloc[:0].copy()),
                                             plan, None, partition="all", attach=attach)
    assert embeddings.shape == (0, 5)
    assert report.n_rows == 0 and report.attached is attach
    assert (data is not None) is attach
    assert _fingerprint(plan) == fingerprint
