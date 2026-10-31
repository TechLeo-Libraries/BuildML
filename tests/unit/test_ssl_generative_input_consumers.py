"""Generative Torch SSL obeys matrix contracts and independent frozen forward sums."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.selfsupervised.checkpoint import load_ssl_bundle
from buildml.selfsupervised.fit import fit_ssl_pretext
from buildml.selfsupervised.torch.encoder import TorchTabularSSLEncoder
from buildml.selfsupervised.transform import transform_ssl

torch = pytest.importorskip("torch")
BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
METHODS = ("mae_tabular", "vae_tabular")


def _configured(method, hidden=(4,)):
    return TorchTabularSSLEncoder(method=method, latent_dim=2, hidden=hidden,
        epochs=2, batch_size=8, mask_ratio=.15, random_state=17, device="cpu")


def _session(backing, directory, poison=False):
    x = np.random.default_rng(52).normal(size=(40, 3))
    if poison:
        x[24:] += 8
    frame = pd.DataFrame(x, columns=["x", "z", "q"])
    frame["target"] = np.where(np.arange(40) % 2, "yes", "no")
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "features.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="memory" if engine == "pandas" else "lazy")
    session.set_roles({"x": "feature", "z": "feature", "q": "feature", "target": "target"}).inject_split(
        train_indices=list(range(24)), validation_indices=list(range(24, 32)), test_indices=list(range(32, 40)))
    return session, x


def _weights(encoder):
    return {name: value.detach().cpu().numpy().copy()
        for name, value in encoder._train_result.module.state_dict().items()}


def _equal_weights(encoder, before):
    after = _weights(encoder)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])


def _linear(x, layer):
    return x@layer.weight.detach().cpu().numpy().astype(float).T + layer.bias.detach().cpu().numpy()


def _mlp(x, layers):
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            x = _linear(x, layer)
        else:
            assert isinstance(layer, torch.nn.ReLU)
            x = np.maximum(x, 0)
    return x


def _expected(encoder, x):
    module = encoder._train_result.module
    x = np.asarray(x, dtype=np.float32).astype(float)
    if encoder.method == "mae_tabular":
        latent = _mlp(x, module.encoder.net)
        reconstructed = _mlp(latent, module.decoder.decoder)
    else:
        # Eval-mode VAE uses the latent Gaussian mean, without sampling noise.
        latent = _linear(_mlp(x, module.encoder_body), module.mu)
        reconstructed = _mlp(latent, module.decoder)
    return latent, reconstructed


def _assert_outputs(encoder, x):
    expected_latent, expected_reconstruction = _expected(encoder, x)
    before = _weights(encoder)
    np.testing.assert_allclose(encoder.transform(x), expected_latent, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(encoder.reconstruct(x), expected_reconstruction, rtol=2e-5, atol=1e-6)
    _equal_weights(encoder, before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("hidden", ((4,), (5, 3)), ids=("one-hidden", "two-hidden"))
@pytest.mark.parametrize("poison", (False, True))
def test_train_only_generative_encoders_and_real_saved_consumers_match_numpy_forward_sums(backing, method, hidden, poison, tmp_path):
    session, x = _session(backing, tmp_path, poison)
    frame = session.to_pandas().copy(deep=True)
    session.ssl.fit_pretext(method=method, latent_dim=2, hidden=hidden, epochs=2,
        batch_size=8, random_state=17, device="cpu")
    plan = session.ssl_plan
    assert plan.n_train_rows == 24 and plan.columns == ("x", "z", "q")
    encoder = plan.encoder_
    explicit_train = _configured(method, hidden).fit(x[:24])
    _equal_weights(encoder, _weights(explicit_train))
    _assert_outputs(encoder, x[32:])
    _assert_outputs(encoder, np.empty((0, 3)))
    restored = TorchTabularSSLEncoder.from_state_dict(encoder.state_dict())
    _assert_outputs(restored, x[32:])
    path = session.ssl.save_bundle(tmp_path / "trusted-ssl")
    loaded, head = load_ssl_bundle(path, trusted=True)
    assert head is None
    weights = _weights(loaded.encoder_)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}
    _assert_outputs(loaded.encoder_, x[32:])
    _, report, embeddings = transform_ssl(session.dataset, loaded, session.split_plan, partition="test")
    np.testing.assert_allclose(embeddings, _expected(encoder, x[32:])[0], rtol=2e-5, atol=1e-6)
    assert report.n_rows == 8
    _equal_weights(loaded.encoder_, weights)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()} == files
    pd.testing.assert_frame_equal(session.to_pandas(), frame)


PROBLEMS = ("nan", "positive-inf", "negative-inf", "overflow", "huge-int", "text", "complex", "rank-one", "rank-three", "features")


def _bad(problem, operation):
    if problem in {"nan", "positive-inf", "negative-inf", "overflow"}:
        return np.full((8, 3), {"nan": np.nan, "positive-inf": np.inf, "negative-inf": -np.inf, "overflow": 1e100}[problem])
    if problem == "huge-int":
        return [[10**1000]*3]*8
    if problem == "text":
        return [["not-a-number"]*3]*8
    if problem == "complex":
        return np.full((8, 3), 1+2j)
    if problem == "rank-one":
        return np.ones(3)
    if problem == "rank-three":
        return np.ones((2, 4, 3))
    # Refit may legitimately choose a new feature width; zero features are invalid.
    return np.empty((8, 0)) if operation == "fit" else np.ones((8, 2))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("route", ("live", "loaded"))
@pytest.mark.parametrize("operation", ("fit", "transform", "reconstruct"))
@pytest.mark.parametrize("problem", PROBLEMS)
def test_invalid_matrices_raise_domain_errors_before_changing_fitted_state_or_rng(method, route, operation, problem, tmp_path):
    session, x = _session("pandas", tmp_path)
    session.ssl.fit_pretext(method=method, latent_dim=2, hidden=(4,), epochs=2,
        batch_size=8, random_state=17, device="cpu")
    encoder = session.ssl_plan.encoder_
    if route == "loaded":
        path = session.ssl.save_bundle(tmp_path / "trusted-ssl")
        encoder = load_ssl_bundle(path, trusted=True)[0].encoder_
    result, width, weights = encoder._train_result, encoder.n_features_in_, _weights(encoder)
    result.module.train()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValidationError):
        getattr(encoder, operation)(_bad(problem, operation))
    assert encoder._train_result is result and encoder.n_features_in_ == width
    assert result.module.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    _equal_weights(encoder, weights)
    _assert_outputs(encoder, x[32:])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation", ("transform", "reconstruct"))
def test_unfitted_generative_encoder_refuses_before_inference(method, operation):
    encoder = _configured(method)
    with pytest.raises(ValidationError, match="not fitted"):
        getattr(encoder, operation)(np.ones((2, 3)))
    assert encoder._train_result is None


@pytest.mark.parametrize("method", ("simclr_tabular", "byol_tabular", "vicreg_tabular"))
def test_real_contrastive_encoder_does_not_claim_a_reconstruction_head(method):
    x = np.random.default_rng(3).normal(size=(12, 3))
    encoder = _configured(method).fit(x)
    before = _weights(encoder)
    with pytest.raises(ValidationError, match="only defined"):
        encoder.reconstruct(x)
    _equal_weights(encoder, before)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation", ("fit", "transform"))
@pytest.mark.parametrize("route", ("session", "module"))
def test_dataframe_boundary_does_not_discard_complex_features_before_encoder_validation(method, operation, route, tmp_path):
    session, _ = _session("pandas", tmp_path)
    session.ssl.fit_pretext(method=method, latent_dim=2, hidden=(4,), epochs=2,
        batch_size=8, random_state=17, device="cpu")
    path = session.ssl.save_bundle(tmp_path / "trusted-ssl")
    frame = session.to_pandas().copy(deep=True)
    frame["x"] = frame["x"].astype(complex)+2j
    receiver = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "q": "feature", "target": "target"}).inject_split(
        train_indices=list(range(24)), validation_indices=list(range(24, 32)), test_indices=list(range(32, 40)))
    receiver.ssl.load_bundle(path, trusted=True)
    plan = receiver.ssl_plan
    before = _weights(plan.encoder_)
    plan.encoder_._train_result.module.train()
    rng = torch.random.get_rng_state().clone()
    history = list(receiver.history)
    with pytest.raises(ValidationError):
        if operation == "fit":
            if route == "session":
                receiver.ssl.fit_pretext(method=method, latent_dim=2, hidden=(4,), epochs=2,
                    batch_size=8, random_state=17, device="cpu")
            else:
                fit_ssl_pretext(receiver.dataset, receiver.split_plan, method=method,
                    latent_dim=2, hidden=(4,), epochs=2, batch_size=8, random_state=17, device="cpu")
        elif route == "session":
            receiver.ssl.transform(partition="test")
        else:
            transform_ssl(receiver.dataset, plan, receiver.split_plan, partition="test")
    assert receiver.ssl_plan is plan and receiver.history == history
    assert plan.encoder_._train_result.module.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    _equal_weights(plan.encoder_, before)
    pd.testing.assert_frame_equal(receiver.to_pandas(), frame)
