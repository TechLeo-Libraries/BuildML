"""Invalid SSL requests cannot produce an untrained or non-finite fitted model."""
import numpy as np
import pytest

from buildml.core.errors import ValidationError
from buildml.selfsupervised.torch.trainer import SSLTrainConfig, train_tabular_ssl

BAD_BUDGETS = [
    ('epochs', 0), ('epochs', -1), ('epochs', 1.5), ('epochs', True),
    ('batch_size', 0), ('batch_size', 1), ('batch_size', 2.5), ('batch_size', True),
    ('learning_rate', 0), ('learning_rate', -1), ('learning_rate', float('nan')),
    ('learning_rate', float('inf')), ('learning_rate', True),
    ('temperature', 0), ('temperature', -1), ('temperature', float('nan')),
    ('temperature', float('inf')), ('temperature', True),
]


@pytest.mark.parametrize('name,value', BAD_BUDGETS)
def test_invalid_budget_refused_before_encoder_construction(name, value, monkeypatch):
    pytest.importorskip('torch')
    from buildml.selfsupervised.torch import models

    constructed = []

    def unexpected_model(*args, **kwargs):
        constructed.append(True)
        pytest.fail('Invalid configuration constructed a model')

    monkeypatch.setattr(models, 'build_simclr_module', unexpected_model)
    config = SSLTrainConfig(method='simclr_tabular', **{name: value})
    with pytest.raises(ValidationError, match=name):
        train_tabular_ssl(np.ones((8, 3)), method='simclr_tabular', latent_dim=2,
                          hidden=(4,), config=config)
    assert not constructed


@pytest.mark.parametrize('name,value', BAD_BUDGETS)
def test_public_encoder_does_not_coerce_invalid_budgets(name, value):
    from buildml.selfsupervised.torch.encoder import TorchTabularSSLEncoder

    with pytest.raises(ValidationError, match=name):
        TorchTabularSSLEncoder(**{name: value})


@pytest.mark.parametrize('matrix', [
    np.ones((4, 0)), np.ones((3, 2)), np.ones(8),
    np.full((8, 2), np.nan), np.full((8, 2), np.inf),
])
def test_invalid_feature_matrix_does_not_return_a_fitted_model(matrix):
    pytest.importorskip('torch')
    with pytest.raises(ValidationError):
        train_tabular_ssl(matrix, method='mae_tabular', latent_dim=2, hidden=(4,))
