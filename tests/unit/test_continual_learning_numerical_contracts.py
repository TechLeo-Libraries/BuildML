"""Actual CPU continual updates obey seed, Fisher, and stream boundaries."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

if find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)

import torch  # noqa: E402

from buildml.core.errors import ValidationError  # noqa: E402
from buildml.online.adapters.torch_continual import ContinualTabularClassifier  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread_torch():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def _data():
    x = np.random.default_rng(27).normal(size=(12, 2)).astype(np.float32)
    return x, np.where(x[:, 0] > 0, 7, 3)


def _model(method="replay_mlp", **kwargs):
    return ContinualTabularClassifier(
        method=method,
        buffer_size=8,
        hidden_dim=4,
        epochs_per_update=2,
        batch_size=4,
        random_state=13,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["replay_mlp", "ewc_mlp"])
def test_seed_controls_initialization_without_consuming_ambient_torch_rng(method):
    x, y = _data()
    observed = []
    for ambient in (29, 71):
        torch.manual_seed(ambient)
        before = torch.random.get_rng_state().clone()
        model = _model(method).partial_fit(x, y, classes=[3, 7])
        assert torch.equal(torch.random.get_rng_state(), before)
        observed.append(
            {key: value.detach().clone() for key, value in model.module_.state_dict().items()}
        )
    for key in observed[0]:
        torch.testing.assert_close(observed[0][key], observed[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("weight", [0.0, 50.0])
def test_ewc_consolidates_learned_weights_with_empirical_per_example_fisher(weight):
    x, y = _data()
    torch.manual_seed(53)
    replay = _model().partial_fit(x, y, classes=[3, 7])
    torch.manual_seed(53)
    model = _model("ewc_mlp", ewc_lambda=weight).partial_fit(x, y, classes=[3, 7])
    params = [p for p in model.module_.parameters() if p.requires_grad]
    for learned, anchor, reference in zip(
        params, model.anchor_params_, replay.module_.parameters()
    ):
        torch.testing.assert_close(learned, reference, rtol=0, atol=0)
        torch.testing.assert_close(anchor, learned, rtol=0, atol=0)
    assert model._ewc_penalty(torch).item() == pytest.approx(0.0, abs=1e-12)
    tensor_x = torch.tensor(np.stack(model.buffer_x_))
    tensor_y = torch.tensor(list(model.buffer_y_), dtype=torch.long)
    gradients = []
    for row, target in zip(tensor_x, tensor_y):
        loss = torch.nn.functional.cross_entropy(model.module_(row[None]), target[None])
        gradients.append(torch.autograd.grad(loss, params))
    expected_fisher = [
        torch.stack([row[i] ** 2 for row in gradients]).mean(0) for i in range(len(params))
    ]
    for actual, expected in zip(model.fisher_diag_, expected_fisher):
        torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        for parameter in params:
            parameter.add_(0.1)
    expected_penalty = weight * sum((f * 0.01).sum().item() for f in expected_fisher)
    assert model._ewc_penalty(torch).item() == pytest.approx(expected_penalty, abs=1e-6)


@pytest.mark.parametrize("method", ["replay_mlp", "ewc_mlp"])
@pytest.mark.parametrize(
    "failure",
    ["shape", "length", "unknown_label", "changed_classes", "nonfinite", "fractional_label"],
)
def test_rejected_updates_preserve_weights_buffer_and_fitted_contract(method, failure):
    x, y = _data()
    model = _model(method).partial_fit(x, y, classes=[3, 7])
    weights = {key: value.clone() for key, value in model.module_.state_dict().items()}
    buffer_x, buffer_y = np.stack(model.buffer_x_).copy(), list(model.buffer_y_)
    before = (model.n_updates_, model.n_features_, model.classes_.copy())
    bad_x, bad_y, classes = x.copy(), y.copy(), [3, 7]
    if failure == "shape":
        bad_x = np.column_stack([bad_x, np.ones(len(bad_x))])
    elif failure == "length":
        bad_y = bad_y[:-1]
    elif failure == "unknown_label":
        bad_y[0] = 99
    elif failure == "changed_classes":
        classes = [3, 9]
    elif failure == "nonfinite":
        bad_x[0, 0] = np.nan
    else:
        bad_y = bad_y.astype(float)
        bad_y[0] = 3.5
    with pytest.raises(ValidationError):
        model.partial_fit(bad_x, bad_y, classes=classes)
    assert model.n_updates_ == before[0] and model.n_features_ == before[1]
    np.testing.assert_array_equal(model.classes_, before[2])
    np.testing.assert_array_equal(np.stack(model.buffer_x_), buffer_x)
    assert list(model.buffer_y_) == buffer_y
    for key, value in model.module_.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_replay_is_fifo_copies_input_and_decodes_noncanonical_class_codes():
    x, y = _data()
    model = _model().partial_fit(x, y, classes=[3, 7])
    saved = x[-8:].copy()
    x[:] = 1000
    np.testing.assert_array_equal(np.stack(model.buffer_x_), saved)
    fresh = np.array([[30.0, 1.0], [31.0, 1.0], [32.0, 1.0], [33.0, 1.0]], dtype=np.float32)
    model.partial_fit(fresh, np.array([3, 7, 3, 7]))
    np.testing.assert_array_equal(np.stack(model.buffer_x_), np.vstack([saved[-4:], fresh]))
    np.testing.assert_array_equal(
        list(model.buffer_y_), np.r_[np.where(y[-4:] == 3, 0, 1), [0, 1, 0, 1]]
    )
    assert len(model.buffer_y_) == 8 and model.n_updates_ == 2
    assert set(model.predict(fresh)) <= {3, 7}


@pytest.mark.parametrize("failure", ["empty", "one_class", "duplicate_classes"])
def test_invalid_initial_population_leaves_unfitted_state_pristine(failure):
    x, y = _data()
    model = _model()
    classes = [3, 7]
    if failure == "empty":
        x, y = x[:0], y[:0]
    elif failure == "one_class":
        classes, y = [3], np.full(len(y), 3)
    else:
        classes = [3, 3]
    with pytest.raises(ValidationError):
        model.partial_fit(x, y, classes=classes)
    assert model.module_ is None and model.classes_.size == 0
    assert model.n_updates_ == model.n_features_ == 0
    assert not model.buffer_x_ and not model.buffer_y_


def test_ewc_waits_for_usable_buffer_and_consolidates_after_learning():
    x, y = _data()
    model = _model("ewc_mlp").partial_fit(x[:1], y[:1], classes=[3, 7])
    assert model.fisher_diag_ is None and not model.anchor_params_
    model.partial_fit(x[1:4], y[1:4])
    assert model.fisher_diag_ is not None
    for learned, anchor in zip(model.module_.parameters(), model.anchor_params_):
        torch.testing.assert_close(learned, anchor, rtol=0, atol=0)


def test_later_updates_preserve_first_consolidation_reference():
    x, y = _data()
    model = _model("ewc_mlp").partial_fit(x, y, classes=[3, 7])
    anchors = [value.clone() for value in model.anchor_params_]
    fisher = [value.clone() for value in model.fisher_diag_]
    model.partial_fit(-x, y)
    for actual, expected in zip(model.anchor_params_, anchors):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(model.fisher_diag_, fisher):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.n_updates_ == 2
    assert torch.isfinite(model._ewc_penalty(torch))


@pytest.mark.parametrize(
    "field,value",
    [
        ("buffer_size", 1),
        ("epochs_per_update", 0),
        ("batch_size", 1),
        ("learning_rate", -1.0),
        ("ewc_lambda", -1.0),
        ("method", "unknown"),
        ("random_state", -1),
    ],
)
def test_invalid_training_configuration_is_refused_before_model_or_buffer_creation(field, value):
    x, y = _data()
    model = _model()
    setattr(model, field, value)
    with pytest.raises(ValidationError):
        model.partial_fit(x, y, classes=[3, 7])
    assert model.module_ is None and model.classes_.size == 0
    assert model.n_updates_ == model.n_features_ == 0
    assert not model.buffer_x_ and not model.buffer_y_
