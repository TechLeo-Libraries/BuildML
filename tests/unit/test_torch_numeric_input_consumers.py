"""Finite numeric tensors must survive loader, CV and normalization boundaries."""

import copy

import numpy as np
import pandas as pd
import pytest
import test_torch_regression_target_consumers as target_consumers

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.cv import cross_validate_torch
from buildml.dl.dataset import arrays_to_tensor_dataset

torch = pytest.importorskip("torch")
frame = target_consumers.frame


def _load(session, modality, normalize):
    kwargs = dict(task="regression", normalize=normalize, shuffle_train=False, batch_size=12)
    if modality == "tabular":
        return session.dl.make_loaders(**kwargs)
    return session.dl.make_multimodal_loaders(
        numeric_columns=["x"],
        text_column="words" if modality in {"numeric_text", "numeric_text_image", "all"} else None,
        image_column="image" if modality in {"numeric_image", "numeric_text_image", "all"} else None,
        audio_column="audio" if modality in {"numeric_audio", "all"} else None,
        image_size=(4, 4), image_channels=1, max_len=4,
        audio_sample_rate=8000, audio_source_sample_rate=8000, audio_max_samples=16,
        normalize_images=False, normalize_audio=False, **kwargs,
    )


INVALID = [
    (bad, normalize, partition, position)
    for bad in (np.inf, -np.inf, 1e100)
    for normalize in (False, True)
    for partition, position in (("train", 0), ("validation", 8), ("test", 10))
    if not (bad == 1e100 and normalize and partition == "train")
]


@pytest.mark.parametrize("backing", target_consumers.BACKINGS)
@pytest.mark.parametrize("modality", target_consumers.MODALITIES)
@pytest.mark.parametrize("bad,normalize,partition,position", INVALID)
def test_nonfinite_numeric_batches_are_refused_before_session_publication(
    frame, backing, modality, bad, normalize, partition, position, tmp_path
):
    data = frame.copy(deep=True)
    data.loc[position, "x"] = bad
    session = target_consumers._session(data, modality, backing, tmp_path)
    before = session.dataset.to_pandas().copy(deep=True)
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="[Ff]eature|[Nn]umeric"):
        _load(session, modality, normalize)
    assert session._torch_loaders is None and session.dl_train_result is None
    assert session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("backing", target_consumers.BACKINGS)
@pytest.mark.parametrize("modality", target_consumers.MODALITIES)
def test_large_finite_train_values_remain_valid_when_normalization_rescales_them(
    frame, backing, modality, tmp_path
):
    data = frame.copy(deep=True)
    data.loc[0, "x"] = 1e100
    session = target_consumers._session(data, modality, backing, tmp_path)
    bundle = _load(session, modality, True)
    train = data.x.iloc[:8].to_numpy(dtype=np.float64)
    mean, std = float(train.mean()), float(train.std())
    assert np.isfinite(mean) and np.isfinite(std)
    for partition, positions in (("train", range(8)), ("validation", [8, 9]), ("test", [10, 11])):
        batch = next(iter(bundle.loaders[partition]))
        numeric, labels = batch[0], batch[-1]
        expected = ((data.x.iloc[list(positions)].to_numpy(dtype=np.float64) - mean) / std).astype(np.float32)
        np.testing.assert_allclose(numeric.numpy().reshape(-1), expected, rtol=1e-6)
        assert torch.isfinite(numeric).all()
        torch.manual_seed(83)
        model = torch.nn.Linear(1, 1)
        w, b = float(model.weight.detach()[0, 0]), float(model.bias.detach()[0])
        residual = w * expected.astype(float) + b - labels.numpy().reshape(-1).astype(float)
        loss = torch.nn.MSELoss()(model(numeric), labels)
        assert torch.isfinite(loss) and float(loss.detach()) == pytest.approx(np.mean(residual**2), rel=2e-6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()
        optimizer.step()
        assert float(model.weight.detach()[0, 0]) == pytest.approx(w - .02*np.mean(residual*expected), abs=2e-6)
        assert float(model.bias.detach()[0]) == pytest.approx(b - .02*np.mean(residual), abs=2e-6)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf, 1e100))
def test_direct_numeric_tensor_conversion_refuses_nonfinite_representation(bad):
    with pytest.raises(ValidationError, match="[Ff]eature|[Nn]umeric"):
        arrays_to_tensor_dataset(np.asarray([[0.0], [bad]]), np.asarray([0, 1]), task="classification")


@pytest.mark.parametrize("bad,normalize", [(bad, normalize) for bad in (np.nan, np.inf, -np.inf, 1e100) for normalize in (False, True) if not (bad == 1e100 and normalize)])
def test_cv_refuses_invalid_numeric_population_before_constructing_a_fold_model(bad, normalize):
    data = pd.DataFrame({"x": np.arange(18, dtype=float), "y": 1+.4*np.arange(18)})
    data.loc[0, "x"] = bad
    dataset = Session.ingest(data).set_roles({"x": "feature", "y": "target"}).dataset
    models = []

    def factory(n_features, task, n_classes):
        model = torch.nn.Linear(n_features, 1)
        models.append(model)
        return model

    with pytest.raises(ValidationError, match="[Ff]eature|[Nn]umeric"):
        cross_validate_torch(dataset, task="regression", normalize=normalize, n_folds=3,
            module_factory=factory, epochs=1, device="cpu", batch_size=24)
    assert not models
