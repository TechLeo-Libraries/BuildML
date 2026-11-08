"""Native local backbones retain truthful controls, gradients and JSON provenance."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.zoo import (
    attach_backbone_head,
    freeze_module,
    load_audio_backbone,
    load_speech_backbone,
    load_vision_backbone,
)

FAMILIES = ["resnet18", "wav2vec2_base", "hubert_base", "whisper_tiny_encoder"]


def _digest(module):
    return hashlib.sha256(
        b"".join(
            name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
            for name, tensor in module.state_dict().items()
        )
    ).hexdigest()


@pytest.fixture(scope="module", params=FAMILIES)
def native(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    name = request.param
    torch.manual_seed(619)
    if name == "resnet18":
        pytest.importorskip("torchvision")
        return {
            "torch": torch,
            "name": name,
            "modality": "vision",
            "path": None,
            "samples": torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32),
        }
    transformers = pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp(name)
    if name == "whisper_tiny_encoder":
        config = transformers.WhisperConfig(
            d_model=8,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=16,
            decoder_ffn_dim=16,
            max_source_positions=16,
            max_target_positions=8,
            num_mel_bins=4,
            vocab_size=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            apply_spec_augment=False,
        )
        producer = transformers.WhisperModel(config)
        expected = _digest(producer.get_encoder())
        modality = "speech"
        samples = torch.linspace(-1, 1, 2 * 4 * 32).reshape(2, 4, 32)
    else:
        config_cls = (
            transformers.Wav2Vec2Config if name == "wav2vec2_base" else transformers.HubertConfig
        )
        model_cls = (
            transformers.Wav2Vec2Model if name == "wav2vec2_base" else transformers.HubertModel
        )
        config = config_cls(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(4, 4),
            conv_stride=(2, 2),
            conv_kernel=(3, 3),
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            feat_proj_dropout=0.0,
            final_dropout=0.0,
            feat_extract_norm="layer",
        )
        producer = model_cls(config)
        expected = _digest(producer)
        modality = "audio"
        samples = torch.linspace(-1, 1, 128).reshape(2, 64)
    # These are native random initializations, not published pretrained weights.
    producer.save_pretrained(directory)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }
    return {
        "torch": torch,
        "name": name,
        "modality": modality,
        "path": directory,
        "samples": samples,
        "producer_digest": expected,
        "files": files,
    }


def _session():
    return Session.ingest(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]}))


def _load(native, route="direct", session=None, **controls):
    values = dict(weights="none", freeze=True, seed=17)
    values.update(controls)
    if native["path"] is not None:
        values["model_id"] = str(native["path"])
    if route == "session":
        return session.dl.load_backbone(native["modality"], native["name"], **values)
    if native["modality"] == "vision":
        return load_vision_backbone(native["name"], **values)
    if native["modality"] == "audio":
        return load_audio_backbone(native["name"], **values)
    return load_speech_backbone(native["name"], **values)


def _files_unchanged(native):
    if native["path"] is not None:
        assert {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in native["path"].iterdir()
            if p.is_file()
        } == native["files"]


@pytest.mark.parametrize("route", ["direct", "session"])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("seed", [17, np.int64(17)], ids=["python-seed", "numpy-seed"])
def test_native_random_backbone_head_and_gradients_match_independent_sums(
    native, route, frozen, seed, tmp_path
):
    _check_native_head(native, route, frozen, seed, "none", tmp_path)


@pytest.mark.parametrize("route", ["direct", "session"])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("seed", [17, np.int64(17)], ids=["python-seed", "numpy-seed"])
@pytest.mark.parametrize("native", FAMILIES[1:], indirect=True)
def test_actual_local_checkpoint_head_and_gradients_match_independent_sums(
    native, route, frozen, seed, tmp_path
):
    _check_native_head(native, route, frozen, seed, "pretrained", tmp_path)


def _check_native_head(native, route, frozen, seed, mode, directory):
    torch = native["torch"]
    session = _session()
    backbone = _load(native, route, session, weights=mode, freeze=frozen, seed=seed)
    assert backbone.frozen is frozen
    assert all(p.requires_grad is not frozen for p in backbone.module.parameters())
    record = backbone.to_dict()
    assert type(record["meta"]["seed"]) is int
    assert json.loads(json.dumps(record, allow_nan=False))["meta"]["seed"] == 17
    assert record["weight_mode"] == mode
    if mode == "pretrained":
        assert _digest(backbone.module) == native["producer_digest"]
    head = (
        session.dl.attach_head(n_classes=3)
        if route == "session"
        else attach_backbone_head(backbone, n_classes=3)
    )
    head.module.eval()
    samples = native["samples"]
    with torch.no_grad():
        output = backbone.module(samples)
        features = (
            output.last_hidden_state.detach().numpy()
            if hasattr(output, "last_hidden_state")
            else output.numpy()
        )
        pooled = features.mean(axis=1) if features.ndim == 3 else features
        weight = head.module.head.weight.detach().numpy().copy()
        bias = head.module.head.bias.detach().numpy().copy()
        expected = pooled @ weight.T + bias
        np.testing.assert_allclose(head.module(samples).numpy(), expected, rtol=2e-5, atol=2e-6)
        if native["modality"] != "vision":
            np.testing.assert_allclose(
                head.module(samples, return_dict=False).numpy(), expected, rtol=2e-5, atol=2e-6
            )
    logits = expected.astype(np.float64)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    delta = probabilities - np.eye(3)[[0, 2]]
    delta /= 2
    expected_gradient = delta.T @ pooled
    expected_bias_gradient = delta.sum(axis=0)
    before = _digest(backbone.module)
    loss = torch.nn.functional.cross_entropy(head.module(samples), torch.tensor([0, 2]))
    loss.backward()
    np.testing.assert_allclose(
        head.module.head.weight.grad.numpy(), expected_gradient, rtol=3e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        head.module.head.bias.grad.numpy(), expected_bias_gradient, rtol=3e-5, atol=2e-6
    )
    assert (
        all(p.grad is None for p in backbone.module.parameters())
        if frozen
        else any(
            p.grad is not None and bool(torch.any(p.grad != 0))
            for p in backbone.module.parameters()
        )
    )
    optimizer = torch.optim.SGD([p for p in head.module.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    assert (_digest(backbone.module) != before) is (not frozen)
    state = directory / "native-head.pt"
    torch.save(head.module.state_dict(), state)
    restored = torch.load(state, weights_only=True)
    for name, value in head.module.state_dict().items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
    _files_unchanged(native)


@pytest.mark.parametrize("route", ["direct", "session"])
@pytest.mark.parametrize("mode", ["pretained", [], {}], ids=["typo", "list", "mapping"])
def test_invalid_weight_modes_do_not_publish_a_random_backbone(native, route, mode):
    _refuse_without_state_change(native, route, weights=mode)


@pytest.mark.parametrize("route", ["direct", "session"])
@pytest.mark.parametrize(
    "freeze", ["false", 0, np.bool_(True)], ids=["string", "integer", "numpy-bool"]
)
def test_invalid_freeze_flags_do_not_change_parameter_or_session_state(native, route, freeze):
    _refuse_without_state_change(native, route, freeze=freeze)


@pytest.mark.parametrize("route", ["direct", "session"])
@pytest.mark.parametrize(
    "seed",
    [True, 1.75, "3", 2**64, -(2**63) - 1],
    ids=["bool", "fraction", "string", "overflow", "underflow"],
)
def test_invalid_mock_seeds_refuse_loss_and_raw_overflow_before_loading(native, route, seed):
    _refuse_without_state_change(native, route, weights="mock", seed=seed)


def _refuse_without_state_change(native, route, **controls):
    torch = native["torch"]
    session = _session()
    prior = _load(native, "session", session, freeze=False)
    head = session.dl.attach_head(n_classes=3)
    fingerprint = _digest(prior.module)
    head_fingerprint = _digest(head.module)
    flags = [p.requires_grad for p in prior.module.parameters()]
    state = torch.get_rng_state().clone()
    with pytest.raises(ValidationError):
        _load(native, route, session, **controls)
    assert session._dl_backbone is head.backbone
    assert session._dl_backbone_head is head
    assert _digest(prior.module) == fingerprint
    assert _digest(head.module) == head_fingerprint
    assert [p.requires_grad for p in prior.module.parameters()] == flags
    assert torch.equal(torch.get_rng_state(), state)
    with torch.no_grad():
        head.module.eval()
        assert torch.isfinite(head.module(native["samples"])).all()
    _files_unchanged(native)


@pytest.mark.parametrize(
    "freeze", ["false", 0, np.bool_(True)], ids=["string", "integer", "numpy-bool"]
)
def test_freeze_module_refuses_invalid_flags_before_parameter_changes(native, freeze):
    backbone = _load(native, freeze=False)
    original = _digest(backbone.module)
    with pytest.raises(ValidationError, match="freeze"):
        freeze_module(backbone.module, freeze=freeze)
    assert all(p.requires_grad for p in backbone.module.parameters())
    assert _digest(backbone.module) == original


@pytest.mark.parametrize("native", ["wav2vec2_base", "hubert_base"], indirect=True)
@pytest.mark.parametrize("seed", [-(2**63), -1, 0, 2**64 - 1])
def test_documented_torch_seed_boundaries_replay_native_mock_weights(native, seed):
    first = _load(native, weights="mock", seed=seed)
    second = _load(native, weights="mock", seed=seed)
    assert _digest(first.module) == _digest(second.module)
    assert json.loads(json.dumps(first.to_dict(), allow_nan=False))["meta"]["seed"] == seed
    generator = native["torch"].Generator().manual_seed(seed)
    assert generator.initial_seed() == seed % 2**64
    _files_unchanged(native)
