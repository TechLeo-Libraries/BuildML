"""Multimodal loaders keep labels out of fresh and replayed features."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.multimodal import MultimodalLoaderConfig, make_multimodal_loaders


@pytest.fixture
def session():
    pytest.importorskip("torch")
    x = np.array([0, 2, 4, 6, 8, 10, 32, 34], dtype=float)
    return (
        Session.ingest(
            pd.DataFrame({"x": x, "words": ["red apple", "green pear"] * 4, "y": 3 * x - 4})
        )
        .set_roles({"x": "feature", "words": "feature", "y": "target"})
        .inject_split(train_indices=list(range(6)), test_indices=[6, 7])
    )


def _make_session_loaders(session, **inputs):
    return session.dl.make_multimodal_loaders(
        **{"text_column": "words", "numeric_columns": ["x"], **inputs},
        task="regression",
        batch_size=8,
        shuffle_train=False,
        normalize=True,
    )


def _assert_holdout(bundle):
    numeric, _, labels = next(iter(bundle.loaders["test"]))
    train = np.arange(0, 12, 2, dtype=float)
    holdout = np.array([32, 34], dtype=float)
    np.testing.assert_allclose(
        numeric.numpy().ravel(), (holdout - train.mean()) / train.std(), rtol=1e-6
    )
    np.testing.assert_array_equal(labels.numpy().ravel(), 3 * holdout - 4)


@pytest.mark.parametrize(
    "inputs",
    [
        {"numeric_columns": ["x", "y"]},
        {"text_column": "y"},
        {"image_column": "y"},
        {"audio_column": "y"},
    ],
    ids=["numeric", "text", "image", "audio"],
)
def test_target_cannot_be_an_explicit_multimodal_input(session, inputs):
    prior = _make_session_loaders(session)
    history = list(session.history)
    with pytest.raises(ValidationError, match="Target column 'y'.*multimodal input"):
        _make_session_loaders(session, **inputs)
    assert session._torch_loaders is prior
    assert session.history == history
    _assert_holdout(prior)


@pytest.mark.parametrize("override", [{}, {"target_column": "x"}, {"numeric_columns": ["y"]}])
def test_frozen_multimodal_contract_respects_the_dataset_target(session, override):
    original = _make_session_loaders(session)
    contract = {**original.multimodal_contract.to_dict(), **override}
    config = MultimodalLoaderConfig(batch_size=8, shuffle_train=False)

    def replay():
        return make_multimodal_loaders(
            session.dataset,
            session.split_plan,
            config=config,
            preprocess=contract,
            task="regression",
        )

    if override:
        with pytest.raises(ValidationError, match="[Tt]arget column"):
            replay()
    else:
        _assert_holdout(replay())
