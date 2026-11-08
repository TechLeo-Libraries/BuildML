"""Real Torch SSL sidecars must agree with their persisted representation contract."""
import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.selfsupervised.checkpoint import load_ssl_bundle

METHODS = ('simclr_tabular', 'byol_tabular', 'vicreg_tabular', 'mae_tabular', 'vae_tabular')


def session(frame):
    return Session.ingest(frame).set_roles({'y': 'target'}).split(test_size=.25, random_state=3)


@pytest.fixture(scope='module', params=METHODS)
def fitted(request):
    pytest.importorskip('torch')
    rng = np.random.default_rng(771)
    x = rng.normal(size=(48, 3))
    frame = pd.DataFrame(x, columns=['a', 'b', 'c'])
    frame['y'] = (x[:, 0] > 0).astype(int)
    s = session(frame)
    s.ssl.fit_pretext(method=request.param, latent_dim=2, hidden=(8,), epochs=1,
                      batch_size=16, random_state=3)
    return s, frame


def affine_network(layers, x):
    for layer in layers:
        if hasattr(layer, 'weight'):
            x = x @ layer.weight.detach().cpu().numpy().T + layer.bias.detach().cpu().numpy()
        else:
            assert type(layer).__name__ == 'ReLU'
            x = np.maximum(x, 0)
    return x


@pytest.mark.parametrize('layout', ['v2', 'v1', 'direct_v2', 'direct_v1'])
def test_restored_embeddings_match_numpy_affine_network(fitted, tmp_path, layout):
    s, frame = fitted
    root = s.ssl.save_bundle(tmp_path/'valid')
    payload = joblib.load(root/'ssl_plan.joblib')
    plan = payload['plan']
    module = s._ssl_plan.encoder_._train_result.module
    x = frame[list(plan.columns)].to_numpy(dtype=np.float32, copy=True)
    if plan.method == 'vae_tabular':
        expected = affine_network([module.mu], affine_network(module.encoder_body, x))
    else:
        expected = affine_network(module.encoder.net, x)
    if layout.startswith('direct'):
        joblib.dump(plan, root/'ssl_plan.joblib')
    if layout.endswith('v1'):
        meta = json.loads((root/'meta.json').read_text())
        meta['format'] = 'buildml.selfsupervised_bundle.v1'
        meta.pop('torch_state_saved')
        (root/'meta.json').write_text(json.dumps(meta))
    restored, head = load_ssl_bundle(root, trusted=True)
    assert head is None
    np.testing.assert_allclose(restored.encoder_.transform(x), expected, rtol=2e-5, atol=2e-6)
    assert restored.encoder_.method == plan.method
    assert restored.bundle_format == ('buildml.selfsupervised_bundle.v1' if layout.endswith('v1') else 'buildml.ssl_bundle.v2')
    consumer = session(frame)
    consumer.ssl.load_bundle(root, trusted=True)
    np.testing.assert_allclose(consumer._ssl_plan.encoder_.transform(x), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('fault', ['method', 'latent_dim', 'columns', 'representation_width',
                                  'duplicate_representation', 'state_method', 'state_not_mapping', 'state_weights', 'state_latent'])
def test_mismatched_sidecar_is_refused_before_session_replacement(fitted, tmp_path, fault):
    torch = pytest.importorskip('torch')
    s, frame = fitted
    good = s.ssl.save_bundle(tmp_path/'good')
    consumer = session(frame)
    consumer.ssl.load_bundle(good, trusted=True)
    snapshot = vars(consumer).copy()
    root = s.ssl.save_bundle(tmp_path/'corrupt')
    payload = joblib.load(root/'ssl_plan.joblib')
    plan = payload['plan']
    changes = {
        'method': {'method': 'vicreg_tabular' if plan.method != 'vicreg_tabular' else 'simclr_tabular'},
        'latent_dim': {'latent_dim': plan.latent_dim+1},
        'columns': {'columns': plan.columns+('extra',)},
        'representation_width': {'representation_columns': ('one',)},
        'duplicate_representation': {'representation_columns': ('dup', 'dup')},
    }
    if fault in changes:
        payload['plan'] = replace(plan, **changes[fault])
        joblib.dump(payload, root/'ssl_plan.joblib')
    else:
        state = torch.load(root/'encoder_torch.pt', weights_only=False)
        if fault == 'state_method':
            state['method'] = 'hf_text_ssl'
        elif fault == 'state_weights':
            state['module'] = {}
        elif fault == 'state_latent':
            state['latent_dim'] = True
        else:
            state = []
        torch.save(state, root/'encoder_torch.pt')
    with pytest.raises(ValidationError):
        consumer.ssl.load_bundle(root, trusted=True)
    assert vars(consumer).keys() == snapshot.keys()
    assert all(vars(consumer)[key] is value for key, value in snapshot.items())

