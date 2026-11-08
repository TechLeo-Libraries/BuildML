"""Speech training, resumed bundles and TorchScript agree with NumPy CNN math."""

import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.speech import build_speech_classifier


def _session(labels):
    waves = [
        (
            0.2 * np.sin(np.arange(45 + i % 7, dtype=np.float32) * (0.12 + i * 0.003)) + i * 0.015
        ).astype(np.float32)
        for i in range(40)
    ]
    frame = pd.DataFrame({"clip": waves, "target": [labels[i % 2] for i in range(40)]})
    return (
        Session.ingest(frame)
        .set_roles({"clip": "feature", "target": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=7)
    )


def _loader_args(normalize):
    return dict(
        audio_column="clip",
        batch_size=6,
        sample_rate=8_000,
        max_samples=64,
        encoder_dim=8,
        normalize_audio=normalize,
        seed=11,
    )


def _check_loaders(session, normalize, labels):
    """Compute train statistics, repeat padding and label codes from raw clips."""
    bundle = session._torch_loaders
    raw_train = np.concatenate(session.partition("train")["clip"].to_list()).astype(np.float64)
    mean, std = float(raw_train.mean()), float(raw_train.std())
    contract = bundle.speech_contract
    assert contract.class_labels == tuple(labels)
    if normalize:
        assert contract.audio_mean == pytest.approx(mean, abs=1e-12)
        assert contract.audio_std == pytest.approx(std, abs=1e-12)
    else:
        assert contract.audio_mean is contract.audio_std is None
    for partition in ("train", "validation", "test"):
        frame = session.partition(partition)
        padded = np.stack([np.resize(clip, 64) for clip in frame["clip"]])[:, None, :]
        if normalize:
            padded = ((padded - float(np.float32(mean))) / float(np.float32(std))).astype(
                np.float32
            )
        values, targets = bundle.loaders[partition].dataset.tensors
        np.testing.assert_allclose(values.numpy(), padded, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(targets.numpy(), [labels.index(y) for y in frame.target])
        assert bundle.loaders[partition].batch_size == 6


def _cnn_logits(module, values):
    """Evaluate three strided cross-correlations without Torch or BuildML ops."""
    state = {
        key: tensor.detach().cpu().numpy().astype(np.float64)
        for key, tensor in module.state_dict().items()
    }
    x = values.astype(np.float64)
    for layer, padding in ((0, 4), (2, 3), (4, 2)):
        weight, bias = state[f"encoder.net.{layer}.weight"], state[f"encoder.net.{layer}.bias"]
        padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, weight.shape[-1], axis=-1)
        x = np.einsum("bctk,ock->bot", windows[:, :, ::2, :], weight) + bias[None, :, None]
        x = np.maximum(x, 0)
    return x.mean(axis=-1) @ state["head.weight"].T + state["head.bias"]


def _check_predictions(torch, session):
    bundle = session._torch_loaders
    values, targets = bundle.loaders["test"].dataset.tensors
    module = session.dl_train_result.module.eval()
    expected = _cnn_logits(module, values.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(module(values).numpy(), expected, rtol=3e-5, atol=3e-6)
    evaluated = session.dl.evaluate(partition="test", device="cpu")
    codes = expected.argmax(axis=1)
    truth = targets.numpy()
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (truth, codes), 1)
    assert evaluated.n_rows == len(truth)
    assert evaluated.class_labels == bundle.contract.class_labels
    assert evaluated.confusion_matrix == confusion.tolist()
    assert evaluated.metrics["accuracy"] == pytest.approx(float(np.mean(truth == codes)))
    assert evaluated.metrics["balanced_accuracy"] == pytest.approx(
        float(np.mean([np.mean(codes[truth == label] == label) for label in (0, 1)]))
    )
    return values, expected


@pytest.mark.parametrize("labels", [(0, 1), (10, 30)], ids=["contiguous", "original-ids"])
@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "train-normalized"])
@pytest.mark.parametrize("path", ["generic", "speech", "speech-frozen", "domain-default"])
def test_speech_training_bundle_resume_and_export_consumers(path, normalize, labels, tmp_path):
    torch = pytest.importorskip(
        "torch", reason="real speech CNN training and TorchScript require Torch"
    )
    torch.set_num_threads(1)
    frozen = path in {"speech-frozen", "domain-default"}
    args = _loader_args(normalize)
    torch.manual_seed(91)
    initial = build_speech_classifier(
        n_classes=2, embed_dim=8, sample_rate=8_000, freeze_encoder=frozen
    )
    encoder_initial = {
        name: value.detach().clone() for name, value in initial.encoder.state_dict().items()
    }
    torch.manual_seed(91)
    session = _session(labels)
    if path == "generic":
        session.dl.make_speech_loaders(**args)
        session.dl.fit(epochs=2, learning_rate=0.01, device="cpu")
    elif path == "domain-default":
        # Exercise the public default that freezes the tiny encoder; no pretrained claim.
        session.dl.domain_adapt_speech(epochs=2, learning_rate=0.01, device="cpu", **args)
        record = session.history[-1]
        assert record["operation_id"] == "domain_adapt_speech_torch"
        assert any("not continued pretrain" in warning for warning in record["warnings"])
    else:
        session.dl.fit_speech(
            epochs=2, learning_rate=0.01, device="cpu", freeze_encoder=frozen, **args
        )
    _check_loaders(session, normalize, labels)
    result = session.dl_train_result
    assert result.n_epochs_ran == 2
    module = result.module
    assert module.encoder.sample_rate == 8_000 and module.embed_dim == 8
    assert all(parameter.requires_grad != frozen for parameter in module.encoder.parameters())
    encoder_changed = any(
        not torch.equal(value, encoder_initial[name])
        for name, value in module.encoder.state_dict().items()
    )
    assert encoder_changed != frozen
    assert not torch.equal(module.head.weight, initial.head.weight)
    values, expected = _check_predictions(torch, session)

    exported = session.dl.export(tmp_path / "active.pt")
    consumer = torch.jit.load(str(exported.path)).eval()
    # Different batch sizes from the trace, using the actual serialized consumer.
    with torch.no_grad():
        for count in (1, len(values)):
            np.testing.assert_allclose(
                consumer(values[:count]).numpy(), expected[:count], rtol=3e-5, atol=3e-6
            )
    destination = session.dl.save_bundle(tmp_path / "trainer")
    restored = _session(labels)
    shell = build_speech_classifier(
        n_classes=2, embed_dim=8, sample_rate=8_000, freeze_encoder=frozen
    )
    restored.dl.load_bundle(destination, shell, map_location="cpu", trusted=True)
    assert restored._torch_loaders is None
    assert restored.dl_train_result.contract == result.contract
    assert restored.dl_train_result.n_epochs_ran == 2
    for name, value in module.state_dict().items():
        assert torch.equal(value, shell.state_dict()[name])
    explicit = restored.dl.export(tmp_path / "explicit.pt", example_input=values[:2])
    with torch.no_grad():
        np.testing.assert_allclose(
            torch.jit.load(str(explicit.path))(values).numpy(), expected, rtol=3e-5, atol=3e-6
        )
    for call in (
        lambda: restored.dl.evaluate(partition="test"),
        lambda: restored.dl.fit(epochs=1, device="cpu", resume=True),
    ):
        with pytest.raises(ValidationError, match="active speech loaders"):
            call()
        assert restored._torch_loaders is None

    # Rebuild BOTH loaders with identical public arguments and shuffle state.
    # A resumed serialized optimizer must produce the same continuation as the live run.
    for current in (session, restored):
        current.dl.make_speech_loaders(**args)
        current.dl.fit(epochs=1, learning_rate=0.01, device="cpu", resume=True)
        assert current.dl_train_result.resumed_from_epochs == 2
        assert current.dl_train_result.n_epochs_ran == 3
        assert [row["epoch"] for row in current.dl_train_result.history] == [1, 2, 3]
        assert current.dl_train_result.module is (module if current is session else shell)
    for name, value in module.state_dict().items():
        np.testing.assert_allclose(
            value.detach().numpy(), shell.state_dict()[name].detach().numpy(), rtol=1e-6, atol=1e-7
        )
    _check_predictions(torch, restored)
    if frozen:
        for name, value in module.encoder.state_dict().items():
            assert torch.equal(value, encoder_initial[name])
    print(
        "SPEECH_NUMERICAL",
        json.dumps(
            {
                "path": path,
                "normalize": normalize,
                "labels": labels,
                "logits": expected.tolist(),
                "epochs_after_resume": 3,
            }
        ),
    )

    # A fresh restored consumer has no loaders; refusal must precede tabular rebuilding.
    missing = _session(labels)
    missing.dl.load_bundle(
        destination,
        build_speech_classifier(n_classes=2, embed_dim=8, sample_rate=8_000, freeze_encoder=frozen),
        trusted=True,
    )
    with pytest.raises(ValidationError, match="active speech loaders or an explicit example_input"):
        missing.dl.export(tmp_path / "refused.pt")
    assert missing._torch_loaders is None
    assert missing._dl_export_result is None
    assert not (tmp_path / "refused.pt").exists()
