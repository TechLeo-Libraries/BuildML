"""Regression target conversion must preserve finite float32 training labels."""

import copy

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.cv import cross_validate_torch
from buildml.dl.dataset import arrays_to_tensor_dataset
from buildml.dl.image import decode_image_cell

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

MODALITIES = (
    "tabular",
    "numeric_text",
    "numeric_image",
    "numeric_audio",
    "numeric_text_image",
    "all",
)
BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")
INVALID = (np.nan, np.inf, -np.inf, 1e100)


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    directory = tmp_path_factory.mktemp("regression_media")
    paths = []
    for i in range(12):
        path = directory / f"pixels-{i}.png"
        Image.fromarray((np.arange(16).reshape(4, 4) + i).astype(np.uint8)).save(path)
        paths.append(str(path))
    return pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": 1 + 0.4 * np.arange(12),
            "words": [f"known token {i % 3}" for i in range(12)],
            "image": paths,
            "audio": [np.linspace(-0.5, 0.5, 8 + i % 4).tolist() for i in range(12)],
            "record": [f"row-{i}" for i in range(12)],
        }
    )


def _session(frame, modality, backing, directory):
    columns = ["x", "y", "record"]
    if modality in {"numeric_text", "numeric_text_image", "all"}:
        columns.append("words")
    if modality in {"numeric_image", "numeric_text_image", "all"}:
        columns.append("image")
    if modality in {"numeric_audio", "all"}:
        columns.append("audio")
    data = frame[columns].copy(deep=True)
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing != "pandas":
        path = directory / "targets.parquet"
        data.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    else:
        session = Session.ingest(data)
    session.set_roles({"x": "feature", "y": "target", "record": "id"})
    return session.inject_split(
        train_indices=list(range(8)), validation_indices=[8, 9], test_indices=[10, 11]
    )


def _load(session, modality):
    kwargs = dict(task="regression", normalize=False, shuffle_train=False, batch_size=12)
    if modality == "tabular":
        return session.dl.make_loaders(**kwargs)
    return session.dl.make_multimodal_loaders(
        numeric_columns=["x"],
        text_column="words" if modality in {"numeric_text", "numeric_text_image", "all"} else None,
        image_column="image"
        if modality in {"numeric_image", "numeric_text_image", "all"}
        else None,
        audio_column="audio" if modality in {"numeric_audio", "all"} else None,
        image_size=(4, 4),
        image_channels=1,
        max_len=4,
        audio_sample_rate=8000,
        audio_source_sample_rate=8000,
        audio_max_samples=16,
        normalize_images=False,
        normalize_audio=False,
        **kwargs,
    )


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("partition,position", (("train", 0), ("validation", 8), ("test", 10)))
@pytest.mark.parametrize(
    "bad", INVALID, ids=("nan", "positive-inf", "negative-inf", "float32-overflow")
)
def test_invalid_target_population_does_not_publish_loaders(
    frame, backing, modality, partition, position, bad, tmp_path
):
    data = frame.copy(deep=True)
    data.loc[position, "y"] = bad
    session = _session(data, modality, backing, tmp_path)
    before = session.dataset.to_pandas().copy(deep=True)
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="[Tt]arget"):
        _load(session, modality)
    assert session._torch_loaders is None and session.dl_train_result is None
    assert session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_actual_batches_match_independent_mse_and_sgd_gradient(frame, backing, modality, tmp_path):
    session = _session(frame, modality, backing, tmp_path)
    bundle = _load(session, modality)
    for partition, positions in [
        ("train", list(range(8))),
        ("validation", [8, 9]),
        ("test", [10, 11]),
    ]:
        batch = next(iter(bundle.loaders[partition]))
        x, labels = batch[0], batch[-1]
        assert labels.dtype == torch.float32 and tuple(labels.shape) == (len(positions), 1)
        assert torch.isfinite(labels).all()
        expected = frame.y.iloc[positions].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(labels.numpy().reshape(-1), expected)
        torch.manual_seed(17)
        model = torch.nn.Linear(1, 1)
        w, b = float(model.weight.detach()[0, 0]), float(model.bias.detach()[0])
        numeric = x.numpy().reshape(-1).astype(float)
        residual = w * numeric + b - expected.astype(float)
        expected_mse = np.mean(residual**2)
        loss = torch.nn.MSELoss()(model(x), labels)
        assert torch.isfinite(loss) and float(loss.detach()) == pytest.approx(
            expected_mse, rel=2e-6
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()
        optimizer.step()
        np.testing.assert_allclose(
            float(model.weight.detach()[0, 0]), w - 0.02 * np.mean(residual * numeric), atol=2e-6
        )
        np.testing.assert_allclose(
            float(model.bias.detach()[0]), b - 0.02 * np.mean(residual), atol=2e-6
        )


@pytest.mark.parametrize("column_shape", (False, True))
@pytest.mark.parametrize(
    "bad", INVALID, ids=("nan", "positive-inf", "negative-inf", "float32-overflow")
)
def test_direct_tensor_conversion_cannot_hide_target_overflow(column_shape, bad):
    targets = np.asarray([1.0, bad])
    if column_shape:
        targets = targets[:, None]
    with pytest.raises(ValidationError, match="[Tt]arget"):
        arrays_to_tensor_dataset(np.asarray([[0.0], [1.0]]), targets, task="regression")


@pytest.mark.parametrize(
    "bad", INVALID, ids=("nan", "positive-inf", "negative-inf", "float32-overflow")
)
def test_cv_refuses_invalid_targets_before_constructing_any_fold_model(bad):
    data = pd.DataFrame({"x": np.arange(18, dtype=float), "y": 1 + 0.4 * np.arange(18)})
    data.loc[0, "y"] = bad
    dataset = Session.ingest(data).set_roles({"x": "feature", "y": "target"}).dataset
    models = []

    def factory(n_features, task, n_classes):
        model = torch.nn.Linear(n_features, 1)
        models.append(model)
        return model

    with pytest.raises(ValidationError, match="[Tt]arget"):
        cross_validate_torch(
            dataset,
            n_folds=3,
            task="regression",
            module_factory=factory,
            epochs=1,
            batch_size=24,
            device="cpu",
            normalize=False,
        )
    assert not models


def test_classification_indices_stay_integer_and_keep_cross_entropy_meaning():
    targets = np.asarray([0, 1])
    dataset = arrays_to_tensor_dataset(np.asarray([[0.0], [1.0]]), targets, task="classification")
    labels = dataset.tensors[-1]
    assert labels.dtype == torch.long and tuple(labels.shape) == (2,)
    logits = np.asarray([[0.2, 0.8], [0.5, 0.1]], dtype=np.float32)
    expected = np.mean(
        np.log(np.exp(logits.astype(float)).sum(axis=1)) - logits[np.arange(2), targets]
    )
    actual = torch.nn.CrossEntropyLoss()(torch.from_numpy(logits), labels)
    assert float(actual) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("path_kind", ("string", "path"))
@pytest.mark.parametrize("channels", (1, 3))
def test_grayscale_png_pixels_have_a_real_conv2d_channel(frame, path_kind, channels):
    from pathlib import Path

    path = Path(frame.image.iloc[0])
    pixels = np.arange(16, dtype=np.uint8).reshape(4, 4)
    expected = np.repeat(pixels.astype(np.float32)[None] / 255, channels, axis=0)
    from_array = decode_image_cell(pixels, size=(4, 4), channels=channels)
    np.testing.assert_array_equal(from_array, expected)
    decoded = decode_image_cell(
        str(path) if path_kind == "string" else path, size=(4, 4), channels=channels
    )
    assert decoded.dtype == np.float32 and decoded.shape == (channels, 4, 4)
    np.testing.assert_array_equal(decoded, expected)
    torch.manual_seed(19)
    model = torch.nn.Conv2d(channels, 2, kernel_size=1)
    weights = model.weight.detach().numpy()[:, :, 0, 0]
    biases = model.bias.detach().numpy()
    reference = np.einsum("oc,chw->ohw", weights, expected)[None] + biases[None, :, None, None]
    with torch.inference_mode():
        actual = model(torch.from_numpy(decoded[None])).numpy()
    np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-6)
