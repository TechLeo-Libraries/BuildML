"""Real vision encoders exclude unobserved targets before decoding image cells."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from test_ssl_labeled_population_consumers import _dataset, _fingerprint, _metrics

from buildml import Session
from buildml.selfsupervised.checkpoint import load_ssl_bundle
from buildml.selfsupervised.evaluate import evaluate_ssl
from buildml.selfsupervised.finetune import finetune_ssl_head
from buildml.selfsupervised.transform import transform_ssl


@pytest.fixture(scope="module")
def vision_trained(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(45)
    images = []
    directory = tmp_path_factory.mktemp("ssl-vision-images")
    for number in range(24):
        path = directory/f"image-{number}.png"
        image.fromarray(rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)).save(path)
        images.append(str(path))
    frame = pd.DataFrame({"image": images, "y": ["yes", "no"]*12}, index=np.arange(24) % 5)
    session = (Session.ingest(frame).set_roles({"image": "feature", "y": "target"})
               .split(test_size=.25, validation_size=.25, random_state=7))
    session.ssl.fit_pretext(method="vision_ssl", image_column="image", epochs=0,
                            weight_mode="mock", batch_size=4, latent_dim=4, random_state=5)
    session.ssl.finetune_head()
    root = session.ssl.save_bundle(tmp_path_factory.mktemp("ssl-vision-population"))
    return session, frame, root


@pytest.fixture(scope="module", params=[False, True])
def vision_fitted(vision_trained, request):
    session, frame, root = vision_trained
    plan, head = session._ssl_plan, session._ssl_head_plan
    if request.param:
        plan, head = load_ssl_bundle(root, trusted=True)
    return session, frame, plan, head


def test_unlabeled_images_are_not_decoded_for_evaluation(vision_fitted):
    session, original, plan, head = vision_fitted
    frame = original.copy()
    positions = np.array(session.split_plan.test_indices)
    blank = positions[::2]
    frame.iloc[blank, frame.columns.get_loc("y")] = None
    frame.iloc[blank, frame.columns.get_loc("image")] = None
    observed = frame.iloc[positions].loc[lambda data: data.y.notna()]
    embeddings = plan.encoder_.transform(observed.image.tolist())
    labels = head.estimator_._buildml_label_encoder_.inverse_transform(head.estimator_.predict(embeddings))
    expected = _metrics(observed.y.tolist(), labels.tolist())
    fingerprint = _fingerprint(plan, head)
    result = evaluate_ssl(_dataset(session, frame), plan, head, session.split_plan, partition="test")
    assert result.metrics == pytest.approx(expected)
    assert result.n_labeled_eval == len(observed)
    assert result.n_unlabeled_eval == len(blank)
    assert _fingerprint(plan, head) == fingerprint


def test_vision_head_fits_only_observed_image_rows(vision_fitted):
    session, original, plan, previous = vision_fitted
    frame = original.copy()
    frame["y"] = frame.y.astype("string")
    train = np.array(session.split_plan.train_indices)
    blank = train[::3]
    frame.iloc[blank, frame.columns.get_loc("y")] = "unobserved"
    frame.iloc[blank[::2], frame.columns.get_loc("y")] = pd.NA
    frame.iloc[blank, frame.columns.get_loc("image")] = None
    observed = frame.iloc[train].loc[lambda data: data.y.notna() & (data.y != "unobserved")]
    embeddings = np.asarray(plan.encoder_.transform(observed.image.tolist()), dtype=float)
    codes = previous.estimator_._buildml_label_encoder_.transform(observed.y.to_numpy())
    reference = clone(previous.estimator_).fit(embeddings, codes)
    fingerprint = _fingerprint(plan, previous)
    head, report = finetune_ssl_head(_dataset(session, frame), session.split_plan, plan,
                                    unlabeled_marker="unobserved")
    assert report.n_labeled_train == len(observed)
    assert report.n_unlabeled_skipped == len(blank)
    query = plan.encoder_.transform(original.iloc[list(session.split_plan.test_indices)].image.tolist())
    np.testing.assert_array_equal(head.estimator_.predict(query), reference.predict(query))
    np.testing.assert_allclose(head.estimator_.coef_, reference.coef_, rtol=1e-12, atol=1e-12)
    assert _fingerprint(plan, previous) == fingerprint


def test_empty_vision_evaluation_keeps_zero_counts(vision_fitted):
    session, original, plan, head = vision_fitted
    fingerprint = _fingerprint(plan, head)
    result = evaluate_ssl(_dataset(session, original.iloc[:0].copy()), plan, head, None, partition="all")
    assert result.n_rows == result.n_labeled_eval == result.n_unlabeled_eval == 0
    assert result.metrics == {}
    assert _fingerprint(plan, head) == fingerprint


def test_empty_vision_export_keeps_actual_backbone_width(vision_fitted):
    session, original, plan, head = vision_fitted
    fingerprint = _fingerprint(plan, head)
    _, report, embeddings = transform_ssl(_dataset(session, original.iloc[:0].copy()),
                                          plan, None, partition="all")
    assert report.n_rows == 0
    assert embeddings.shape == (0, plan.latent_dim) == (0, 512)
    assert _fingerprint(plan, head) == fingerprint
