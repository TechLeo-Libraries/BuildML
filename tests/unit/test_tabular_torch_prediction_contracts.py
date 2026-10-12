"""Real tabular Torch estimators preserve prediction and device contracts."""
import numpy as np
import pytest

from buildml.semisupervised.adapters.torch_consistency import TabularConsistencyClassifier
from buildml.symbolic.adapters.torch_neuro import TabularConceptBottleneck, TabularNeuralAdditive


@pytest.mark.parametrize('method', ['fixmatch_tabular', 'mixmatch_tabular'])
def test_consistency_probabilities_are_repeatable_and_match_labels(method):
    pytest.importorskip('torch')
    rng = np.random.default_rng(27)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    labels = np.where(x[:, 0] > 0, 7, 3)
    labels[::3] = -1
    estimator = TabularConsistencyClassifier(method=method, epochs=2, batch_size=16, threshold=0)
    estimator.fit(x, labels)
    weights = {key: value.clone() for key, value in estimator.module_.state_dict().items()}
    probabilities = estimator.predict_proba(x)
    np.testing.assert_array_equal(estimator.predict_proba(x), probabilities)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(estimator.predict(x), estimator.classes_[probabilities.argmax(axis=1)])
    if method == "fixmatch_tabular":
        assert estimator.n_pseudo_labels_ > 0
    for key, value in estimator.module_.state_dict().items():
        assert value.equal(weights[key])


@pytest.mark.parametrize('estimator_type', [TabularConceptBottleneck, TabularNeuralAdditive])
@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_neuro_estimator_predicts_on_training_device_and_scores_independently(estimator_type, task, device):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware/runtime unavailable')
    rng = np.random.default_rng(52)
    x = rng.normal(size=(32, 2)).astype(np.float32)
    y = np.where(x[:, 0] > 0, 'yes', 'no') if task == 'classification' else (2 * x[:, 0] - x[:, 1])
    estimator = estimator_type(task=task, device=device, epochs=2, batch_size=16)
    estimator.fit(x, y)
    assert next(estimator.module_.parameters()).device.type == device
    prediction = estimator.predict(x)
    assert prediction.shape == (32,)
    np.testing.assert_array_equal(estimator.predict(x), prediction)
    if task == 'classification':
        assert set(prediction).issubset({'yes', 'no'})
        expected = np.mean(prediction == y)
    else:
        assert np.isfinite(prediction).all()
        expected = 1 - np.sum((prediction - y) ** 2) / np.sum((y - y.mean()) ** 2)
    assert estimator.score(x, y) == pytest.approx(float(expected), abs=1e-6)
