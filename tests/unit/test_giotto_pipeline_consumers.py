"""Real Giotto consumers with independent neighbors, topology and frozen heads."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.data.splits import SplitPlan


def require_stack():
    pytest.importorskip('gtda')
    pytest.importorskip('ripser')
    pytest.importorskip('persim')


def reference_features(train, queries, vectorizer=None, kind='betti_curve'):
    """Use direct Giotto batches; select clouds by independent Euclidean sorting."""
    from gtda.diagrams import BettiCurve, PersistenceImage, PersistenceLandscape
    from gtda.homology import VietorisRipsPersistence

    clouds = np.stack([train[np.argsort(np.sum((train - row) ** 2, axis=1))[:5]] for row in queries])
    diagrams = VietorisRipsPersistence(homology_dimensions=(0, 1), collapse_edges=True).fit_transform(clouds)
    if vectorizer is None:
        factories = dict(betti_curve=lambda: BettiCurve(n_bins=5),
                         persistence_image=lambda: PersistenceImage(n_bins=5),
                         landscape=lambda: PersistenceLandscape(n_bins=5, n_layers=2))
        vectorizer = factories[kind]()
        features = vectorizer.fit_transform(diagrams)
    else:
        features = vectorizer.transform(diagrams)
    return features.reshape(len(queries), -1), vectorizer


@pytest.mark.parametrize('kind', ['betti_curve', 'persistence_image', 'landscape'])
@pytest.mark.parametrize('mode', ['classification', 'regression', 'none'])
@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('cap', [12, 40])
def test_giotto_pipeline_matches_direct_batches_and_replays_frozen_state(kind, mode, standardize, cap, tmp_path):
    require_stack()
    rng = np.random.default_rng(727)
    raw = rng.normal(size=(32, 3)) * [1, 2, .4]
    raw[20:] += [4, -3, 2]
    labels = ['001' if i % 2 == 0 else '010' for i in range(32)]
    target = labels if mode != 'regression' else raw[:, 0] - .3 * raw[:, 1] + rng.normal(size=32)
    frame = pd.DataFrame(raw, columns=['a', 'b', 'c'], index=np.arange(32)[::-1] + 100)
    frame['target'] = target
    source = Session.ingest(frame).set_roles({'target': 'target'})
    split = SplitPlan('injected', .1875, .1875, None, None,
                      tuple(range(20)), tuple(range(20, 26)), tuple(range(26, 32)))
    source._split_plan = split
    head = 'none' if mode == 'none' else 'ridge' if mode == 'regression' else 'logistic_regression'
    mapper = kind == 'betti_curve' and mode == 'none' and not standardize
    result = source.tda.fit(backend='giotto', vectorization=kind, knn=5, n_bins=5,
                            n_layers=2, head=head, task=None if mode == 'none' else mode,
                            standardize=standardize, max_points_guard=cap,
                            subsample_strategy='random', mapper=mapper, random_state=7)
    chosen = np.sort(np.random.default_rng(7).choice(20, size=cap, replace=False)) if cap < 20 else np.arange(20)
    train = raw[chosen]
    mean = train.mean(axis=0) if standardize else np.zeros(3)
    scale = train.std(axis=0) if standardize else np.ones(3)
    normalized = (train - mean) / scale
    expected_train, vectorizer = reference_features(normalized, normalized, kind=kind)
    plan = source.tda_plan
    assert result.n_train_rows == len(chosen)
    np.testing.assert_allclose(plan.train_x_, normalized, atol=1e-12)
    np.testing.assert_allclose(plan.train_tda_features_, expected_train, rtol=1e-6, atol=1e-7)
    assert plan.feature_dim == expected_train.shape[1] == len(set(plan.feature_names))
    if standardize:
        np.testing.assert_allclose(plan.mean_, mean)
        np.testing.assert_allclose(plan.scale_, scale)
    else:
        assert plan.mean_ is plan.scale_ is None
    if mapper:
        if cap < 20:
            assert plan.mapper_summary_ is None
            assert any('Mapper' in warning for warning in result.warnings)
        else:
            assert plan.mapper_summary_['n_train_mapper_points'] == 20
            assert plan.mapper_summary_['n_mapper_nodes'] >= 0
    if mode == 'classification':
        reference = LogisticRegression(max_iter=500, random_state=7).fit(expected_train, chosen % 2)
    elif mode == 'regression':
        reference = Ridge(random_state=7).fit(expected_train, np.asarray(target)[chosen])
    root = source.tda.save_bundle(tmp_path / 'frozen')
    replay = Session(dataset=source.dataset, split_plan=split)
    replay.tda.load_bundle(root, trusted=True)
    for consumer in [source, replay]:
        saved = consumer.tda_plan
        fingerprint = joblib.hash((saved.train_x_, saved.nn_._fit_X, saved.vectorizer_state_,
                                   saved.mean_, saved.scale_, saved.head_estimator_))
        for partition, positions in [('validation', range(20, 26)), ('test', range(26, 32)), ('all', range(32))]:
            expected, _ = reference_features(normalized, (raw[list(positions)] - mean) / scale, vectorizer)
            actual = consumer.tda.transform(partition=partition)
            np.testing.assert_allclose(actual.features, expected, rtol=1e-6, atol=1e-7)
            assert actual.n_rows == len(positions)
            if mode == 'none':
                with pytest.raises(ValidationError, match='head'):
                    consumer.tda.predict(partition=partition)
                continue
            predictions = consumer.tda.predict(partition=partition).predictions
            direct = reference.predict(expected)
            if mode == 'classification':
                wanted = ["001" if code == 0 else "010" for code in direct]
                assert list(predictions) == wanted
                accuracy = sum(a == labels[i] for a, i in zip(wanted, positions, strict=True)) / len(positions)
                assert consumer.tda.evaluate(partition=partition).metrics['accuracy'] == pytest.approx(accuracy)
            else:
                np.testing.assert_allclose(predictions, direct, atol=1e-9, rtol=1e-7)
                residuals = np.asarray(target)[list(positions)] - direct
                metrics = consumer.tda.evaluate(partition=partition).metrics
                assert metrics['mae'] == pytest.approx(np.mean(np.abs(residuals)))
                assert metrics['rmse'] == pytest.approx(np.sqrt(np.mean(residuals ** 2)))
        assert joblib.hash((saved.train_x_, saved.nn_._fit_X, saved.vectorizer_state_,
                            saved.mean_, saved.scale_, saved.head_estimator_)) == fingerprint


@pytest.mark.parametrize('cloud', [np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                                   np.array([[0., 0.], [2., 0.], [5., 0.], [9., 0.]])])
def test_real_giotto_h0_matches_prim_mst_and_square_h1(cloud):
    require_stack()
    from buildml.tda.adapters.giotto import compute_giotto_diagrams

    diagrams = compute_giotto_diagrams(cloud, homology_dims=(0, 1), maxdim=1)
    reached = {0}
    edges = []
    while len(reached) < len(cloud):
        weight, destination = min((float(np.linalg.norm(cloud[i] - cloud[j])), j)
                                  for i in reached for j in range(len(cloud)) if j not in reached)
        reached.add(destination)
        edges.append(weight)
    np.testing.assert_allclose(np.sort(diagrams[0][:, 1]), sorted(edges), atol=1e-6)
    np.testing.assert_array_equal(diagrams[0][:, 0], 0)
    if np.max(cloud[:, 1]) > 0:
        np.testing.assert_allclose(diagrams[1], [[1., np.sqrt(2)]], atol=1e-6)
    else:
        assert diagrams[1].shape == (0, 2)
